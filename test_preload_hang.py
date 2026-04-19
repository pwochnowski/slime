"""Minimal repro for GCR LD_PRELOAD + deferred activation with Ray actors.

LD_PRELOAD interposes dlsym/sendmsg/recvmsg from process start.
The dlsym hook is always active but harmless until libcuda.so.1 is loaded.
The sendmsg/recvmsg hooks are passthrough until gcr_activate() is called
from init_CR() after cuInit succeeds — no application changes needed.
"""
import ray
import sys

GCR_HOME = "/root/GCR"
PRELOAD_SO = f"{GCR_HOME}/GCR/libpreload.so"
LIBCUDA_SO = f"{GCR_HOME}/GCR/libcuda.so"

ray.init()

@ray.remote
class Canary:
    def ping(self):
        import torch
        torch.zeros(1, device='cuda')  # trigger CUDA init → cuInit → init_CR → gcr_activate
        return "ok"

# LD_PRELOAD both libraries; hooks are gated until cuInit calls gcr_activate
ActorCls = Canary.options(num_gpus=1, runtime_env={
    "env_vars": {"GCR_HOME": GCR_HOME, "LD_PRELOAD": f"{PRELOAD_SO}:{LIBCUDA_SO}"},
})
actor = ActorCls.remote()
try:
    result = ray.get(actor.ping.remote(), timeout=15)
    print(f"PASS: {result}")
except ray.exceptions.GetTimeoutError:
    print("HANG: timed out after 15s")
    ray.kill(actor)
