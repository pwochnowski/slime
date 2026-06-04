import gc
import logging
from collections.abc import Iterable

import psutil
import torch
import torch.distributed as dist

from gcr import device_mem_get_info
from gcr import log_gpu_memory as log_gpu_memory  # re-exported for callers

logger = logging.getLogger(__name__)


def _collect_cuda_storages(roots: Iterable) -> list[torch.UntypedStorage]:
    """Collect all unique CUDA storages reachable from model/optimizer objects."""
    seen_ptrs: set[int] = set()
    storages: list[torch.UntypedStorage] = []

    def _visit_tensor(t: torch.Tensor):
        if not t.is_cuda:
            return
        s = t.untyped_storage()
        ptr = s.data_ptr()
        if ptr != 0 and ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            storages.append(s)

    def _visit(obj, depth=0):
        if depth > 8:
            return
        if isinstance(obj, torch.Tensor):
            _visit_tensor(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _visit(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _visit(v, depth + 1)

    for root in roots:
        if hasattr(root, "parameters"):
            for p in root.parameters():
                _visit_tensor(p)
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    _visit_tensor(p.main_grad)
                if hasattr(p, "main_param") and isinstance(p.main_param, torch.Tensor):
                    _visit_tensor(p.main_param)
        if hasattr(root, "buffers"):
            for b in root.buffers():
                _visit_tensor(b)
        if hasattr(root, "state"):
            _visit(root.state)
        if hasattr(root, "param_groups"):
            _visit(root.param_groups)
        # Megatron DistributedOptimizer: walk contiguous param/grad buffers
        if hasattr(root, "buffers") and hasattr(root, "gbuf_ranges"):
            for buf in root.buffers:
                if hasattr(buf, "param_data") and buf.param_data is not None:
                    _visit_tensor(buf.param_data)
                if hasattr(buf, "grad_data") and buf.grad_data is not None:
                    _visit_tensor(buf.grad_data)
        # Megatron DistributedOptimizer: walk shard groups
        for attr in ("shard_fp32_from_float16_groups", "shard_fp32_groups", "shard_float16_groups"):
            if hasattr(root, attr):
                _visit(getattr(root, attr))
        # Inner optimizer (e.g. Adam) state
        if hasattr(root, "optimizer") and hasattr(root.optimizer, "state"):
            _visit(root.optimizer.state)

    return storages


def offload_cuda_storages(roots: Iterable) -> list[tuple[torch.UntypedStorage, torch.UntypedStorage]]:
    """Move all CUDA storages from model/optimizer to CPU, freeing GPU memory.

    Returns a list of (gpu_storage, cpu_copy) pairs for later restoration.
    """
    storages = _collect_cuda_storages(roots)
    saved = []
    total_bytes = 0
    for gpu_storage in storages:
        total_bytes += gpu_storage.nbytes()
        cpu_copy = gpu_storage.cpu()
        saved.append((gpu_storage, cpu_copy))

    for gpu_storage, _ in saved:
        gpu_storage.resize_(0)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(
        f"[Rank {dist.get_rank()}] Offloaded {len(saved)} CUDA storages "
        f"({total_bytes / 1024**3:.2f} GB) to CPU"
    )
    return saved


def onload_cuda_storages(saved: list[tuple[torch.UntypedStorage, torch.UntypedStorage]]):
    """Restore CUDA storages from CPU copies produced by offload_cuda_storages."""
    total_bytes = 0
    for gpu_storage, cpu_copy in saved:
        gpu_storage.resize_(cpu_copy.nbytes())
        gpu_storage.copy_(cpu_copy)
        total_bytes += cpu_copy.nbytes()

    torch.cuda.synchronize()
    logger.info(
        f"[Rank {dist.get_rank()}] Onloaded {len(saved)} CUDA storages "
        f"({total_bytes / 1024**3:.2f} GB) back to GPU"
    )


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = device_mem_get_info(device)
    vm = psutil.virtual_memory()
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
        "host_total_GB": _byte_to_gb(vm.total),
        "host_available_GB": _byte_to_gb(vm.available),
        "host_used_GB": _byte_to_gb(vm.used),
        "host_free_GB": _byte_to_gb(vm.free),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def dump_cuda_segments(label: str = ""):
    """Dump PyTorch CUDA caching allocator segments (VA, size, type, expandable).

    Cross-reference these VAs against GCR hook log cuMemCreate VAs to see whether
    PyTorch's pool sits on top of GCR-tracked VMM regions.
    """
    import os
    snap = torch.cuda.memory._snapshot()
    rank = dist.get_rank() if dist.is_initialized() else 0
    backend = torch.cuda.get_allocator_backend()
    cuda_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
    alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "<unset>")
    launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "<unset>")
    lines = [
        f"[pytorch-segments rank={rank}] {label}",
        f"  [allocator] backend={backend} PYTORCH_CUDA_ALLOC_CONF={cuda_conf} PYTORCH_ALLOC_CONF={alloc_conf}",
        f"  [env] CUDA_LAUNCH_BLOCKING={launch_blocking} pid={os.getpid()}",
    ]
    for seg in snap.get("segments", []):
        lines.append(
            f"  addr=0x{seg['address']:x} "
            f"total={seg['total_size']} "
            f"allocated={seg['allocated_size']} "
            f"active={seg.get('active_size', 0)} "
            f"type={seg.get('segment_type', '?')} "
            f"expandable={seg.get('is_expandable', False)}"
        )
    print("\n".join(lines), flush=True)


def print_train_mem(label: str):
    device = torch.cuda.current_device()
    rank = dist.get_rank() if dist.is_initialized() else 0
    free, total = device_mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    G = 2**30
    print(
        f"[train-mem rank={rank} gpu={device}] {label}: "
        f"free={free / G:.2f}G alloc={allocated / G:.2f}G reserved={reserved / G:.2f}G total={total / G:.2f}G",
        flush=True,
    )


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
