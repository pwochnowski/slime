"""Minimal test: does GCR track PyTorch's GPU allocations?

Allocates tensors on GPU under GCR interposition and checks whether
GCR's control shm sees them.  No Ray, no Megatron — just torch + GCR.

Run:
  GCR_HOME=/root/GCR LD_PRELOAD=$GCR_HOME/GCR/libpreload.so \
    python tests/utils/test_gcr_alloc_tracking.py

Expects the `cr` binary on PATH.
"""

import json
import os
import subprocess
import sys

import torch


def gcr_query() -> dict | None:
    pid = os.getpid()
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    try:
        r = subprocess.run(
            ["cr", "-q", "-p", str(pid)],
            capture_output=True, text=True, timeout=5, env=env,
        )
        if r.returncode == 0:
            return json.loads(r.stdout)
    except Exception as e:
        print(f"cr -q failed: {e}", file=sys.stderr)
    return None

def _device_mem_get_info(device: int) -> tuple[int, int]:
    """Return (free, total) bytes for ``device``, device-wide.

    Fast path: ``torch.cuda.mem_get_info`` (in-process CUDA driver call).
    Fallback: ``nvidia-smi`` subprocess — works even when the CUDA context is
    suspended (e.g. after GCR checkpoint) because it spawns a fresh process
    with its own driver context.
    """
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
    except Exception as e:
        pass

    return 0, 0

def report(label: str):
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    free, total = _device_mem_get_info(dev)

    gcr = gcr_query()
    gcr_va = gcr.get("total_va_bytes") or 0 if gcr else 0
    gcr_n = gcr.get("num_allocations", 0) if gcr else 0

    G = 2**30
    print(
        f"[{label}]  "
        f"alloc={alloc/G:.3f}G  reserved={reserved/G:.3f}G  " +
        (f"device={(total-free)/G:.3f}G  " if free and total else "") +
        f"gcr_va={gcr_va/G:.3f}G  gcr_n={gcr_n}"
    )


def main():
    torch.cuda.set_device(0)

    # Show PyTorch allocator config
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)")
    print(f"PYTORCH_CUDA_ALLOC_CONF = {alloc_conf}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PID: {os.getpid()}")
    print()

    report("baseline")

    # Phase 1: small allocation
    a = torch.randn(1024, 1024, device="cuda")  # ~4 MB
    report("after 4MB tensor")

    # Phase 2: larger allocation
    b = torch.randn(8192, 8192, device="cuda")  # ~256 MB
    report("after 256MB tensor")

    # Phase 3: simulate training (matmul + backward)
    x = torch.randn(4096, 4096, device="cuda", requires_grad=True)
    y = torch.randn(4096, 4096, device="cuda")
    loss = (x @ y).sum()
    loss.backward()
    report("after matmul+backward")

    # Phase 4: free intermediate tensors
    del loss, x, y
    torch.cuda.synchronize()
    report("after del (before empty_cache)")

    torch.cuda.empty_cache()
    report("after empty_cache")

    # Phase 5: suspend and resume
    print("\n--- Suspend ---")
    from gcr import suspend, resume
    suspend([os.getpid()])
    report("after suspend")
    print("--- Resume ---")
    resume([os.getpid()])
    report("after suspend+resume")

    # Phase 6: allocate again — does it reuse or double?
    del b
    torch.cuda.empty_cache()
    c = torch.randn(8192, 8192, device="cuda")  # ~256 MB
    report("after 256MB post-resume")

    # Cleanup
    del a, c
    torch.cuda.empty_cache()
    report("final cleanup")

    print("\nDone.")


if __name__ == "__main__":
    main()