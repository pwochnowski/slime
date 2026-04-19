"""GCR (GPU Checkpoint/Restore) orchestrator.

Thin wrapper around the ``cr`` CLI tool that suspends and resumes all CUDA
device memory for a set of processes.  GCR operates via LD_PRELOAD
interposition on the CUDA driver API — it is external to the application.

Control plane: shared memory at ``/mnt/huge/control-<PID>`` + ``SIGUSR1``.
"""

import logging
import os
import shutil
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)

HUGE_PAGES_DIR = "/mnt/huge"


class GCRError(RuntimeError):
    """Raised when a GCR suspend or resume operation fails."""


def check_available() -> None:
    """Verify that the ``cr`` binary is on PATH and hugetlbfs is mounted.

    Call once at startup to fail fast with a clear error message.
    """
    if shutil.which("cr") is None:
        raise GCRError("'cr' binary not found on PATH. Is GCR installed?")
    if not os.path.isdir(HUGE_PAGES_DIR):
        raise GCRError(f"{HUGE_PAGES_DIR} does not exist. Is hugetlbfs mounted?")


def suspend(pids: Sequence[int], timeout_s: float = 120.0) -> None:
    """Suspend (dump) GPU memory for one or more processes.

    Calls ``cr -d -p <PID1> -p <PID2> ...`` and blocks until all processes
    have completed suspension.
    """
    _run_cr("-d", pids, timeout_s)


def resume(pids: Sequence[int], timeout_s: float = 120.0) -> None:
    """Resume (restore) GPU memory for one or more processes.

    Calls ``cr -r -p <PID1> -p <PID2> ...`` and blocks until all processes
    have completed restoration.  Virtual addresses are preserved.
    """
    _run_cr("-r", pids, timeout_s)


def _run_cr(flag: str, pids: Sequence[int], timeout_s: float) -> None:
    if not pids:
        return
    cmd = ["cr", flag]
    for pid in pids:
        cmd.extend(["-p", str(pid)])
    logger.info("GCR: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            raise GCRError(f"cr {flag} failed (rc={result.returncode}): {result.stderr.strip()}")
    except subprocess.TimeoutExpired as e:
        raise GCRError(f"cr {flag} timed out after {timeout_s}s for pids {list(pids)}") from e
