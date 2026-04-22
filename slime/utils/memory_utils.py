import gc
import logging

import psutil
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = _device_mem_get_info(device)
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


def _device_mem_get_info(device: int) -> tuple[int, int]:
    """Return (free, total) bytes for ``device``, device-wide.

    Fast path: ``torch.cuda.mem_get_info`` (in-process CUDA driver call).
    Fallback: ``nvidia-smi`` subprocess — works even when the CUDA context is
    suspended (e.g. after GCR checkpoint) because it spawns a fresh process
    with its own driver context.
    """
    import os
    import subprocess

    try:
        return torch.cuda.mem_get_info(device)
    except Exception:
        pass

    try:
        env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits", "-i", str(device)],
            capture_output=True, text=True, timeout=5, env=env,
        )
        if result.returncode == 0:
            free_mib, total_mib = (int(x) for x in result.stdout.strip().split(", "))
            return free_mib * 1024 * 1024, total_mib * 1024 * 1024
    except Exception:
        pass

    return 0, 0


def log_gpu_memory(label: str = "") -> dict:
    """Log a compact GPU memory summary to stderr (always visible).

    Reports four views of GPU memory:
      - alloc: memory held by live PyTorch tensors
      - reserved: memory held by PyTorch's caching allocator (alloc + cached free blocks)
      - driver: total GPU memory in use (from CUDA driver, matches nvidia-smi)
      - gcr: memory tracked by GCR (gpu_bytes on device, host_bytes dumped to host)

    Key comparisons:
      - reserved vs alloc → PyTorch's cached free blocks
      - driver vs reserved → memory outside PyTorch (CUDA ctx, NCCL, etc.)
      - gcr.gpu_bytes vs driver → memory not tracked by GCR (leaked or non-GCR allocs)
    """
    import os
    import sys

    device = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free, total = _device_mem_get_info(device)
    device_used = total - free

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Query GCR stats for this process
    gcr_str = ""
    gcr_stats = None
    try:
        from slime.utils.gcr import query_stats
        gcr_stats = query_stats(os.getpid())
        if gcr_stats:
            gcr_host = gcr_stats.get("total_physical_bytes_host") or 0
            gcr_va = gcr_stats.get("total_va_bytes") or 0
            gcr_n = gcr_stats.get("num_allocations", 0)
            gcr_cycles = gcr_stats.get("num_suspend_cycles", 0)
            gcr_state = gcr_stats.get("state", "?")
            gcr_str = (
                f"  gcr=[host={gcr_host / 2**30:.2f}G  "
                f"va={gcr_va / 2**30:.2f}G  n={gcr_n}  "
                f"cycles={gcr_cycles}  state={gcr_state}]"
            )
    except Exception:
        pass

    print(
        f"[gpu-mem][rank {rank}] {label}: "
        f"alloc={alloc / 2**30:.2f}G  "
        f"reserved={reserved / 2**30:.2f}G  "
        f"device={device_used / 2**30:.2f}G  "
        f"(gap={max(0, device_used - reserved) / 2**30:.2f}G  "
        f"free={free / 2**30:.2f}G)"
        f"{gcr_str}",
        file=sys.stderr,
        flush=True,
    )
    result = {"alloc": alloc, "reserved": reserved, "device_used": device_used, "free": free, "total": total}
    if gcr_stats:
        result["gcr"] = gcr_stats
    return result


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
