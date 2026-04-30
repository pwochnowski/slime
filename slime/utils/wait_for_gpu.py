"""Block until specified GPUs have no compute processes outside our Ray cluster.

Enabled by setting SLIME_WAIT_FOR_GPU_IDS (e.g. "0,1"). Intended to be called
from worker init paths just before any CUDA op, so that Ray placement /
worker spawn / code import can complete during preemption windows without
holding the GPU.
"""

import logging
import os
import subprocess
import time

import psutil

logger = logging.getLogger(__name__)


def _our_cluster_pids() -> set[int]:
    """Return PIDs that belong to our Ray cluster (raylet + all descendants).

    Used to distinguish "external preemptor" GPU users from sibling slime
    workers that may have already touched CUDA on a colocated GPU.
    """
    pids: set[int] = set()
    try:
        me = psutil.Process()
        pids.add(me.pid)

        raylet = None
        cur = me
        while cur is not None:
            try:
                name = cur.name()
                cmd = cur.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if "raylet" in name or any("raylet" in c for c in cmd):
                raylet = cur
                break
            try:
                cur = cur.parent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

        if raylet is None:
            for p in psutil.process_iter(["name", "cmdline"]):
                try:
                    name = p.info["name"] or ""
                    cmd = p.info["cmdline"] or []
                    if "raylet" in name or any("raylet" in c for c in cmd):
                        raylet = p
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        if raylet is not None:
            pids.add(raylet.pid)
            try:
                for child in raylet.children(recursive=True):
                    pids.add(child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        logger.warning(f"_our_cluster_pids: falling back to self-only: {e}")
    return pids


def _external_pids_on_gpu(gpu_id: int, our_pids: set[int]) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    for tok in result.stdout.split():
        tok = tok.strip()
        if not tok:
            continue
        try:
            pid = int(tok)
        except ValueError:
            continue
        if pid not in our_pids:
            pids.append(pid)
    return pids


def wait_for_gpus_free(
    gpu_ids: list[int] | None = None,
    poll_interval: float = 2.0,
    log_interval: float = 30.0,
    label: str = "",
) -> None:
    """Block until all specified GPUs are free of external compute processes.

    If gpu_ids is None, reads SLIME_WAIT_FOR_GPU_IDS (comma-separated). If
    that is unset or empty, returns immediately.
    """
    if gpu_ids is None:
        env = os.environ.get("SLIME_WAIT_FOR_GPU_IDS", "").strip()
        if not env:
            return
        gpu_ids = [int(x) for x in env.split(",") if x.strip()]
    if not gpu_ids:
        return

    tag = f"[wait_for_gpu{(' ' + label) if label else ''}]"
    logger.info(f"{tag} Watching GPUs {gpu_ids} until external compute processes exit")
    last_log = 0.0
    start = time.monotonic()
    while True:
        our_pids = _our_cluster_pids()
        busy: dict[int, list[int]] = {}
        for gid in gpu_ids:
            ext = _external_pids_on_gpu(gid, our_pids)
            if ext:
                busy[gid] = ext
        if not busy:
            elapsed = time.monotonic() - start
            logger.info(f"{tag} GPUs {gpu_ids} free (waited {elapsed:.1f}s)")
            return
        now = time.monotonic()
        if now - last_log >= log_interval:
            logger.info(f"{tag} still waiting; busy={busy}")
            last_log = now
        time.sleep(poll_interval)