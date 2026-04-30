"""
Reproduce the post-C/R NCCL launch failure observed in slime production.

Topology (mirroring slime colocate mode):
  - 2 producer workers (TP=2 NCCL group), one per GPU.  Allocate VMM, cat
    flat tensors into it, send fd to the paired consumer over UDS, free
    after consumer done.  Mirrors megatron_utils/.../update_weight_from_tensor.
  - 2 consumer workers, one per GPU, pinned to same GPU as paired producer.
    Receive fd, import_vmm_buffer, wrap, do many copy_() reads into separate
    "model parameter" tensors (mimics model.load_weights), sync, free.

Per round (chunk): producer all_gather; alloc+cat+sync; send fd; consumer
imports/uses/frees; producer joins fd thread, frees.  N_CHUNKS rounds per phase.

PHASE A: pre-C/R, no extra syncs.
Coordinator: cr -d / cr -r on all 4 PIDs.
PHASE B: post-C/R, sync_each=True so sticky cuda 719/700 surfaces at the
actual offending kernel rather than the next NCCL collective.

Run:
  GCR_HOME=/root/GCR python tests/test_vmm_ipc_after_cr.py
"""

import argparse
import os
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

SENTINEL_DIR = Path("/tmp/slime_vmm_after_cr")
N_CHUNKS = 8           # chunks per phase (slime does ~10 per update_weights)
N_PARAMS_PER_CHUNK = 16  # "named tensors" per chunk (consumer does N copy_ reads)
PARAM_NUMEL = 1024 * 1024  # 4 MiB float32 per param  → ~64 MiB per chunk
TENSOR_NUMEL = 256 * 1024  # 1 MiB float32 for TP all_gather
WORLD_SIZE = 2
PROD = "producer"
CONS = "consumer"

# Persistent VMM allocations that survive all C/R cycles.  Mimic slime's
# pre-existing GCR-tracked mappings (SGLang KV cache + master_daemon shared
# regions).  The 1777463001 log showed ~14GB / ~500 allocations resident
# across the C/R boundary; we approximate at smaller scale.
N_PERSISTENT_VMM = 32
PERSISTENT_VMM_SIZE = 64 * 1024 * 1024  # 64 MiB each → 2 GiB total per process

# Number of back-to-back C/R cycles between PHASE A and PHASE B.  Slime hits
# 2-3 cycles per training iteration; the post-canary crashes at iter 1 train
# follow ~3 cycles total.
N_CR_CYCLES = 3


def _log(role: str, rank: int, msg: str) -> None:
    print(f"[{role[0]}{rank}] {msg}", flush=True)


def _alloc_persistent_vmm(device: int, n: int, size: int):
    """Allocate N VMM buffers that the caller keeps alive forever (until
    process exit).  Mimics SGLang's KV cache and other mappings that survive
    every C/R cycle in slime.  Returns (allocs, bufs); both must be retained
    by the caller — drop either and the mapping is unmapped.
    """
    from sglang.srt.weight_sync.vmm_ipc import alloc_vmm_buffer, wrap_as_torch_uint8
    import torch

    allocs = []
    bufs = []
    for i in range(n):
        alloc = alloc_vmm_buffer(size, device)
        buf = wrap_as_torch_uint8(alloc)
        # Touch every page so the mapping is actually backed before C/R.
        buf[::1024 * 1024].fill_(0xAB)
        allocs.append(alloc)
        bufs.append(buf)
    torch.cuda.synchronize()
    return allocs, bufs


# ---------------------------------------------------------------------------
# Producer worker
# ---------------------------------------------------------------------------


def producer_main(rank: int):
    import torch
    import torch.distributed as dist
    from sglang.srt.weight_sync.vmm_ipc import (
        alloc_vmm_buffer,
        free_vmm_buffer,
        open_sidecar_listener,
        send_fd,
        wrap_as_torch_uint8,
    )

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    _log(PROD, rank, f"pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    torch.cuda.set_device(0)
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    dist.init_process_group(backend="nccl", rank=rank, world_size=WORLD_SIZE)
    _log(PROD, rank, "NCCL initialized")

    device = 0
    dev = "cuda:0"

    # Held for the lifetime of the worker so the mappings survive every
    # C/R cycle.  Dropping either list unmaps the VMM.
    _persistent = _alloc_persistent_vmm(device, N_PERSISTENT_VMM, PERSISTENT_VMM_SIZE)
    _log(PROD, rank, f"persistent VMM: {len(_persistent[0])} bufs × {PERSISTENT_VMM_SIZE} bytes")

    # The UDS path is fixed per (producer rank, chunk index) so the consumer
    # knows where to connect without coordinator round-trip.
    def uds_path(chunk_idx: int) -> str:
        return f"/tmp/slime_vmm_repro-prod{rank}-chunk{chunk_idx}.sock"

    def run_chunk(label: str, chunk_idx: int, sync_each: bool) -> None:
        # Step 1: TP all_gather (mimics gather_from_tp_ranks).  This is the
        # collective that surfaces sticky errors first in production.
        x = torch.full((TENSOR_NUMEL,), float(rank + 1), device=dev, dtype=torch.float32)
        gathered = [torch.empty_like(x) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, x)
        if sync_each:
            torch.cuda.synchronize()

        # Step 2: build flat_parts (N_PARAMS_PER_CHUNK separate tensors), then
        # alloc_vmm + cat into the VMM-mapped buffer (matches real producer).
        flat_parts = []
        total_bytes = 0
        param_meta = []
        for p in range(N_PARAMS_PER_CHUNK):
            t = torch.full((PARAM_NUMEL,), float(rank + p + 1), device=dev, dtype=torch.float32)
            flat = t.flatten().view(torch.uint8)
            param_meta.append((total_bytes, total_bytes + flat.numel()))
            flat_parts.append(flat)
            total_bytes += flat.numel()

        alloc = alloc_vmm_buffer(total_bytes, device)
        if sync_each:
            torch.cuda.synchronize()
        buf = wrap_as_torch_uint8(alloc)
        torch.cat(flat_parts, out=buf[:total_bytes])
        torch.cuda.synchronize(device)
        if sync_each:
            _log(PROD, rank, f"{label} chunk={chunk_idx} cat+sync ok")
        del flat_parts

        # Step 3: serve fd to consumer over UDS.
        path = uds_path(chunk_idx)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        listener = open_sidecar_listener(path)

        served = threading.Event()

        def _serve():
            conn, _ = listener.accept()
            send_fd(conn, alloc.fd)
            # Wait for consumer to ack done before producer frees.
            try:
                conn.recv(8)
            except Exception:
                pass
            conn.close()
            listener.close()
            served.set()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()

        # Hand over (rank, chunk_idx, total_bytes, n_params, params_meta) via
        # a small "ready" file the consumer polls.  Consumer reads it then
        # connects to the UDS to get the fd.
        ready_path = SENTINEL_DIR / f"chunk_ready_{rank}_{chunk_idx}"
        with open(ready_path, "wb") as f:
            payload = struct.pack(
                "QQ",
                total_bytes,
                N_PARAMS_PER_CHUNK,
            )
            f.write(payload)

        # Step 4: wait for consumer to finish (signals via served event).
        if not served.wait(timeout=120):
            raise RuntimeError(f"consumer never connected for chunk {chunk_idx}")
        thread.join(timeout=10)
        if sync_each:
            torch.cuda.synchronize()

        # Step 5: producer free.
        del buf
        free_vmm_buffer(alloc, close_fd=True)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        if sync_each:
            torch.cuda.synchronize()

        # Sanity-check on the all_gather result so we don't optimize it away.
        _ = torch.cat(gathered).sum().item()

    # PHASE A
    for i in range(N_CHUNKS):
        run_chunk("phaseA", i, sync_each=False)
    _log(PROD, rank, "PHASE A done; waiting for resume")

    (SENTINEL_DIR / f"prod_ready_{rank}").touch()
    deadline = time.monotonic() + 600
    while not (SENTINEL_DIR / f"go_prod_{rank}").exists():
        if time.monotonic() > deadline:
            sys.exit(f"[p{rank}] timed out waiting for go")
        time.sleep(0.1)
    _log(PROD, rank, "received go; entering PHASE B")

    # Clean up phase A sentinels so phase B fresh.
    for f in SENTINEL_DIR.glob(f"chunk_ready_{rank}_*"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass

    # PHASE B
    for i in range(N_CHUNKS):
        run_chunk("phaseB", i, sync_each=True)
    _log(PROD, rank, "PHASE B done")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Consumer worker
# ---------------------------------------------------------------------------


def consumer_main(rank: int):
    """Pinned to the same GPU as producer rank.  Imports the fd, runs many
    copy_() reads into "model param" target tensors, sync, free.
    """
    import torch
    from sglang.srt.weight_sync.vmm_ipc import (
        free_vmm_buffer,
        import_vmm_buffer,
        open_sidecar_client,
        recv_fd,
        wrap_as_torch_uint8,
    )

    _log(CONS, rank, f"pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    torch.cuda.set_device(0)
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    device = 0
    dev = "cuda:0"

    # Held for the lifetime of the worker so the mappings survive every
    # C/R cycle.  Dropping either list unmaps the VMM.
    _persistent = _alloc_persistent_vmm(device, N_PERSISTENT_VMM, PERSISTENT_VMM_SIZE)
    _log(CONS, rank, f"persistent VMM: {len(_persistent[0])} bufs × {PERSISTENT_VMM_SIZE} bytes")

    # Pre-allocate "model parameters" — separate cudaMalloc'd tensors that the
    # consumer copies into (mimics model.load_weights's parameter tensors).
    model_params = [
        torch.empty(PARAM_NUMEL, dtype=torch.float32, device=dev)
        for _ in range(N_PARAMS_PER_CHUNK)
    ]

    def consume_chunk(label: str, chunk_idx: int, sync_each: bool) -> None:
        # Wait for producer's ready file.
        ready_path = SENTINEL_DIR / f"chunk_ready_{rank}_{chunk_idx}"
        deadline = time.monotonic() + 120
        while not ready_path.exists():
            if time.monotonic() > deadline:
                raise RuntimeError(f"producer never produced chunk {chunk_idx}")
            time.sleep(0.01)
        with open(ready_path, "rb") as f:
            total_bytes, n_params = struct.unpack("QQ", f.read(16))
        try:
            ready_path.unlink()
        except FileNotFoundError:
            pass

        uds_path = f"/tmp/slime_vmm_repro-prod{rank}-chunk{chunk_idx}.sock"
        sock = open_sidecar_client(uds_path)
        fd = recv_fd(sock)

        alloc = import_vmm_buffer(fd, total_bytes, device)
        buf = wrap_as_torch_uint8(alloc)
        if sync_each:
            torch.cuda.synchronize()

        # Reconstruct N "named tensor" views and copy_ into model_params.
        param_byte_size = PARAM_NUMEL * 4
        views = []
        for p in range(int(n_params)):
            start = p * param_byte_size
            end = start + param_byte_size
            v = buf[start:end].view(torch.float32)
            views.append(v)
            model_params[p].copy_(v)
        torch.cuda.synchronize(device)
        if sync_each:
            _log(CONS, rank, f"{label} chunk={chunk_idx} copy_+sync ok")

        # Consumer free (mirrors model_runner.py:1537-1538: del views, buf, free).
        del views, v, buf
        free_vmm_buffer(alloc, close_fd=True)
        if sync_each:
            torch.cuda.synchronize()

        # Tell producer we're done so it can free.
        try:
            sock.send(b"\0" * 8)
        except Exception:
            pass
        sock.close()

    # PHASE A
    for i in range(N_CHUNKS):
        consume_chunk("phaseA", i, sync_each=False)
    _log(CONS, rank, "PHASE A done; waiting for resume")

    (SENTINEL_DIR / f"cons_ready_{rank}").touch()
    deadline = time.monotonic() + 600
    while not (SENTINEL_DIR / f"go_cons_{rank}").exists():
        if time.monotonic() > deadline:
            sys.exit(f"[c{rank}] timed out waiting for go")
        time.sleep(0.1)
    _log(CONS, rank, "received go; entering PHASE B")

    # PHASE B
    for i in range(N_CHUNKS):
        consume_chunk("phaseB", i, sync_each=True)
    _log(CONS, rank, "PHASE B done")


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def _spawn(role: str, rank: int, preload_lib: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_PRELOAD"] = preload_lib
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(WORLD_SIZE)
    env["CUDA_VISIBLE_DEVICES"] = str(rank)
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = env.get("MASTER_PORT", "29500")
    env["NCCL_CUMEM_ENABLE"] = "1"
    env["NCCL_SHM_DISABLE"] = "1"
    env["NCCL_NET_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_NVLS_ENABLE"] = "0"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    return subprocess.Popen(
        [sys.executable, "-u", __file__, "--worker", "--role", role, "--rank", str(rank)],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _wait_ready(workers: dict, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    pending = set(workers)
    while pending:
        for key in list(pending):
            role, rank = key
            sentinel = SENTINEL_DIR / f"{'prod' if role == PROD else 'cons'}_ready_{rank}"
            if sentinel.exists():
                pending.discard(key)
                _log("coord", -1, f"{role} {rank} ready (pid={workers[key].pid})")
        for key, w in workers.items():
            if w.poll() is not None and key in pending:
                sys.exit(f"[coord] {key} exited rc={w.returncode} before barrier")
        if time.monotonic() > deadline:
            sys.exit("[coord] timed out waiting for workers")
        time.sleep(0.1)


def coordinator_main():
    gcr_home = os.environ.get("GCR_HOME", "/root/GCR")
    preload_lib = f"{gcr_home}/GCR/libpreload.so:{gcr_home}/GCR/libcuda.so"
    cr_bin = f"{gcr_home}/GCR/cr"

    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    for f in SENTINEL_DIR.iterdir():
        f.unlink()

    workers: dict = {}
    for rank in range(WORLD_SIZE):
        workers[(PROD, rank)] = _spawn(PROD, rank, preload_lib)
    for rank in range(WORLD_SIZE):
        workers[(CONS, rank)] = _spawn(CONS, rank, preload_lib)
    _log("coord", -1, f"spawned: {[(k, w.pid) for k, w in workers.items()]}")

    try:
        _wait_ready(workers)
        _log("coord", -1, f"all workers ready; running {N_CR_CYCLES} C/R cycles")

        cr_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        cmd_d = [cr_bin, "-d"]
        cmd_r = [cr_bin, "-r"]
        for w in workers.values():
            cmd_d.extend(["-p", str(w.pid)])
            cmd_r.extend(["-p", str(w.pid)])

        for cycle in range(N_CR_CYCLES):
            t0 = time.monotonic()
            rc = subprocess.run(cmd_d, env=cr_env, timeout=180).returncode
            _log("coord", -1, f"cycle {cycle}: cr -d rc={rc} ({time.monotonic()-t0:.1f}s)")
            if rc != 0:
                sys.exit(f"[coord] cycle {cycle}: cr -d failed rc={rc}")

            t0 = time.monotonic()
            rc = subprocess.run(cmd_r, env=cr_env, timeout=180).returncode
            _log("coord", -1, f"cycle {cycle}: cr -r rc={rc} ({time.monotonic()-t0:.1f}s)")
            if rc != 0:
                sys.exit(f"[coord] cycle {cycle}: cr -r failed rc={rc}")

        _log("coord", -1, "all C/R cycles done; releasing workers")
        # Producers and consumers must enter PHASE B together — release in same step.
        for rank in range(WORLD_SIZE):
            (SENTINEL_DIR / f"go_prod_{rank}").touch()
            (SENTINEL_DIR / f"go_cons_{rank}").touch()

        rcs = {key: w.wait(timeout=600) for key, w in workers.items()}
        _log("coord", -1, f"exit codes: {rcs}")
        if any(rc != 0 for rc in rcs.values()):
            sys.exit("[coord] FAIL: at least one worker did not exit cleanly")
        _log("coord", -1, "PASS")
    finally:
        for w in workers.values():
            if w.poll() is None:
                w.terminate()
                try:
                    w.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    w.kill()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--role", choices=[PROD, CONS])
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()

    if args.worker:
        if args.role == PROD:
            producer_main(args.rank)
        else:
            consumer_main(args.rank)
    else:
        coordinator_main()


if __name__ == "__main__":
    main()