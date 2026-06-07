import subprocess, os

def _gpu_processes() -> list[str]:
    """Snapshot GPU compute processes via nvidia-smi.

    Works even when the CUDA context is suspended (e.g. after GCR checkpoint)
    because it spawns a fresh process with its own driver context.  Returns
    lines like ``gpu=2 pid=881338 sglang::scheduler_TP0 602MiB``.
    """
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, env=env,
        )
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, env=env,
        )
    except Exception:
        return []
    if gpus.returncode != 0 or procs.returncode != 0:
        return []
    uuid_to_index: dict[str, str] = {}
    for line in gpus.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2:
            uuid_to_index[parts[1]] = parts[0]
    out = []
    for line in procs.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        uuid, pid, name, mem = parts
        out.append(f"gpu={uuid_to_index.get(uuid, '?')} pid={pid} {name} {mem}MiB")
    return out

def device_mem_get_info(device: int) -> tuple[int, int]:
    """Return (free, total) bytes for ``device``, device-wide.

    Fast path: ``torch.cuda.mem_get_info`` (in-process CUDA driver call).
    Fallback: ``nvidia-smi`` subprocess — works even when the CUDA context is
    suspended (e.g. after GCR checkpoint) because it spawns a fresh process
    with its own driver context.
    """
    # try:
    #     import torch
    #     return torch.cuda.mem_get_info(device)
    # except Exception:
    #     pass

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

def log_phase_memory(label: str, num_devices: int):
    G = 2**30
    mem = {}
    for dev in range(num_devices):
        free, total = device_mem_get_info(dev)
        mem[dev] = (free, total)
    procs_by_gpu: dict[int, list[str]] = {dev: [] for dev in range(num_devices)}
    for proc in _gpu_processes():
        parts = proc.split(" ", 1)
        try:
            gpu_id = int(parts[0].split("=")[1])
            rest = parts[1] if len(parts) > 1 else proc
            procs_by_gpu.setdefault(gpu_id, []).append(rest)
        except (IndexError, ValueError):
            procs_by_gpu.setdefault(-1, []).append(proc)
    lines = [f"[phase-mem] {label}"]
    for dev in range(num_devices):
        free, total = mem[dev]
        used = total - free
        proc_strs = ",\t ".join(procs_by_gpu.get(dev, []))
        lines.append(f"  gpu {dev}: free={free / G:.2f}G, used={used / G:.2f}G\nprocs: {proc_strs}")
    print("\n".join(lines), flush=True)
