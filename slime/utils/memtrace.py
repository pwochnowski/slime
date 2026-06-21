"""Per-phase GPU memory tracing for colocated RL runs.

Emits raw, tagged memory numbers (no in-framework attribution) for offline
processing into stacked-bar figures. Two data sources are combined offline:

1. Torch caching-allocator snapshot (``torch.cuda.memory._snapshot()``) written
   to a per-(rank, snapshot) pickle sidecar. Stack-trace attribution to semantic
   buckets (weights / optimizer / gradients / activations) is done offline.
2. Device ground truth (``nvml_used``) = total device footprint including memory
   torch cannot see (NCCL buffers, CUDA context, raw cudaMalloc). The remainder
   bucket is derived offline as ``nvml_used - torch_reserved``.

This module is the *training-side* writer and the shared on-disk format. The
inference side (SGLang server subprocess) writes the same schema independently
(see sglang scheduler ``dump_memory_snapshot``), because torch allocator stats
are per-process and the SGLang model lives in a different process.

All gated behind ``SLIME_MEMTRACE=1`` so there is zero overhead by default.
Recording must be enabled (``_record_memory_history``) at process startup before
any large allocation, on every rank.
"""

import json
import logging
import os
import pickle

logger = logging.getLogger(__name__)

_ENABLED = None
_DIR = None
_RECORDING_STARTED = False
_NVML_INITED = None
_ROLE = None


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("SLIME_MEMTRACE", "0") == "1"
    return _ENABLED


def warmup_count() -> int:
    try:
        return int(os.environ.get("SLIME_MEMTRACE_WARMUP", "2"))
    except ValueError:
        return 2


def trace_dir() -> str:
    global _DIR
    if _DIR is None:
        d = os.environ.get("SLIME_MEMTRACE_DIR") or os.path.join(os.getcwd(), "mem_trace")
        os.makedirs(d, exist_ok=True)
        _DIR = d
    return _DIR


def _init_nvml() -> bool:
    global _NVML_INITED
    if _NVML_INITED is not None:
        return _NVML_INITED
    try:
        import pynvml

        pynvml.nvmlInit()
        _NVML_INITED = True
    except Exception as e:  # pragma: no cover - depends on driver
        logger.warning(f"[memtrace] pynvml unavailable, falling back to mem_get_info: {e}")
        _NVML_INITED = False
    return _NVML_INITED


def _physical_index(torch_index: int) -> int:
    """Map a torch device index to a physical NVML index via CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return torch_index
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    if 0 <= torch_index < len(visible):
        return visible[torch_index]
    return torch_index


def nvml_used(device: int | None = None) -> int:
    """Device-wide used bytes (ground-truth ceiling), via NVML when available.

    Falls back to ``cudaMemGetInfo`` (total - free), which is the same
    device-wide quantity scoped to the current CUDA context.
    """
    import torch

    if device is None:
        device = torch.cuda.current_device()
    if _init_nvml():
        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(_physical_index(device))
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        except Exception:
            pass
    free, total = torch.cuda.mem_get_info(device)
    return int(total - free)


def init_recording(role: str) -> None:
    """Enable allocation-history recording once per process. No-op if disabled.

    Must run before large allocations so every block carries an alloc stack.
    """
    global _RECORDING_STARTED, _ROLE
    if not enabled():
        return
    _ROLE = role
    _init_nvml()
    if _RECORDING_STARTED:
        return
    import torch

    try:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        _RECORDING_STARTED = True
        logger.info(f"[memtrace] recording enabled (role={role}, dir={trace_dir()})")
    except Exception as e:  # pragma: no cover
        logger.warning(f"[memtrace] failed to enable _record_memory_history: {e}")


def _default_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", 0))


def _append(role: str, rank: int, line: dict) -> None:
    path = os.path.join(trace_dir(), f"mem_trace_{role}_rank{rank}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")


def capture(
    snapshot: str,
    iter: int,
    idle_comm: str,
    role: str | None = None,
    rank: int | None = None,
    warmup: bool | None = None,
    extra: dict | None = None,
) -> None:
    """Capture one (rank, snapshot): scalars jsonl line + torch _snapshot sidecar.

    ``snapshot`` is one of train_peak / post_train_idle / gen_peak / post_gen_idle.
    ``idle_comm`` is which communicator group is idle now ("train"/"inference"/"none").
    ``iter`` is the monotonic iteration index (rollout_id); no wall-clock used.
    """
    if not enabled():
        return
    import torch

    role = role or _ROLE or "train"
    if rank is None:
        rank = _default_rank()
    if warmup is None:
        warmup = iter < warmup_count()

    device = torch.cuda.current_device()
    reserved = int(torch.cuda.memory_reserved(device))
    allocated = int(torch.cuda.memory_allocated(device))
    used = nvml_used(device)

    snap_file = f"torchsnap_{role}_rank{rank}_{snapshot}_iter{iter}.pkl"
    try:
        snap = torch.cuda.memory._snapshot()
        with open(os.path.join(trace_dir(), snap_file), "wb") as f:
            pickle.dump(snap, f)
    except Exception as e:  # pragma: no cover
        logger.warning(f"[memtrace] _snapshot dump failed ({snapshot} rank{rank}): {e}")
        snap_file = ""

    line = {
        "role": role,
        "rank": rank,
        "snapshot": snapshot,
        "iter": iter,
        "idle_comm": idle_comm,
        "nvml_used": used,
        "torch_reserved": reserved,
        "torch_allocated": allocated,
        "snap_file": snap_file,
        "warmup": bool(warmup),
    }
    if extra:
        line.update(extra)
    _append(role, rank, line)
    logger.info(
        f"[memtrace] {snapshot} iter={iter} rank={rank} role={role} "
        f"nvml_used={used} reserved={reserved} allocated={allocated} warmup={warmup}"
    )


def nccl_delta(
    comm: str,
    before: int,
    after: int,
    role: str | None = None,
    rank: int | None = None,
    iter: int = -1,
) -> None:
    """Record nvml_used immediately before/after a communicator init.

    Cross-checks the offline remainder (nvml_used - torch_reserved) attribution.
    """
    if not enabled():
        return
    role = role or _ROLE or "train"
    if rank is None:
        rank = _default_rank()
    line = {
        "type": "nccl_init_delta",
        "role": role,
        "rank": rank,
        "comm": comm,
        "iter": iter,
        "nvml_before": int(before),
        "nvml_after": int(after),
        "nccl_init_delta": int(after - before),
    }
    _append(role, rank, line)
    logger.info(f"[memtrace] nccl_init_delta comm={comm} rank={rank} delta={after - before}")
