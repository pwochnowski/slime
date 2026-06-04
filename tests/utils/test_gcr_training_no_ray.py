"""No-Ray GCR offload_tag failure repro.

Initializes the Megatron actor (so model + optimizer get allocated under the
MT_MODEL_OPTIM_TAG=1 region), then calls ``gcr.offload_tag([pid], [1])`` to
unmap those VMM allocations *without* notifying PyTorch's caching allocator.

Then probes:
  (a) The model tensor's data_ptr — same integer, but is it still mapped?
  (b) ``torch.cuda.synchronize()`` — does the unmap leave the context in a
      sticky error state? (Case 1)
  (c) A fresh ``torch.empty(8, device='cuda')`` — does PyTorch's allocator
      hand out a pointer into an unmapped segment? (Case 2)

Toggle ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` to widen Case 2.

Run:  python tests/utils/test_gcr_training_no_ray.py
"""

import os
import sys

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_PATH = f"/root/models/{MODEL_NAME}"
SEQ_LEN = 128
BATCH_SIZE = 4


def _ensure_gcr_preload():
    """Re-exec with GCR LD_PRELOAD if not already active."""
    gcr_home = os.environ.get("GCR_HOME", "/root/GCR")
    preload = f"{gcr_home}/GCR/libpreload.so:{gcr_home}/GCR/libcuda.so"
    if preload not in os.environ.get("LD_PRELOAD", ""):
        os.environ["LD_PRELOAD"] = preload
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


def _get_free_port():
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _patch_ray_out():
    """Replace the two Ray call-sites in the train path with direct access."""
    import slime.ray.train_actor as _ta

    _ta.get_local_gpu_id = lambda: 0

    import slime.utils.data as _data
    from slime.utils.timer import Timer

    def _process_no_ray(args, rollout_data_ref, dp_rank, dp_size):
        assert len(rollout_data_ref) == dp_size
        rollout_data = rollout_data_ref[dp_rank].inner
        partition = rollout_data.pop("partition")
        total_lengths = rollout_data["total_lengths"]
        Timer().seq_lens = total_lengths
        rollout_data["total_lengths"] = [total_lengths[i] for i in partition]
        return rollout_data

    _data.process_rollout_data = _process_no_ray


def setup():
    _ensure_gcr_preload()

    sys.path.insert(0, "/root/Megatron-LM")
    os.environ["PYTHONPATH"] = "/root/Megatron-LM/"
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    # os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Surface async CUDA errors at the actual offending call.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ["NCCL_NET_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    # os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_get_free_port())
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["no_proxy"] = "127.0.0.1"
    os.environ.setdefault("GCR_HOME", "/root/GCR")

    _patch_ray_out()

    sys.argv = [
        "test_gcr_no_ray.py",
        # -- model architecture (Qwen2.5-0.5B) --
        "--swiglu",
        "--num-layers", "24",
        "--hidden-size", "896",
        "--ffn-hidden-size", "4864",
        "--num-attention-heads", "14",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000",
        "--group-query-attention",
        "--num-query-groups", "2",
        "--vocab-size", "151936",
        # -- checkpoint --
        "--hf-checkpoint", MODEL_PATH,
        # -- training --
        "--rollout-batch-size", str(BATCH_SIZE),
        "--global-batch-size", str(BATCH_SIZE),
        "--advantage-estimator", "grpo",
        "--eps-clip", "0.2",
        "--optimizer", "adam",
        "--lr", "1e-6",
        "--lr-decay-style", "constant",
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", "4096",
        # -- parallelism (single GPU, no TP/PP) --
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        # -- mode --
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", "1",
        "--colocate",
        "--debug-train-only",
        "--num-rollout", "2",
        "--megatron-to-hf-mode", "bridge",
        "--log-level", "warning",
    ]


_CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
_MEMORY_TYPE_NAMES = {0: "unset", 1: "HOST", 2: "DEVICE", 3: "ARRAY", 4: "UNIFIED"}


def probe_ptr(ptr: int):
    """Query the driver about a VA without dereferencing it.

    Returns (rc, mem_type_name). rc==0 means the VA is mapped; nonzero
    (typically 1 = CUDA_ERROR_INVALID_VALUE) means unmapped/invalid.
    """
    import ctypes

    cuda = ctypes.CDLL("libcuda.so")
    out = ctypes.c_int()
    rc = cuda.cuPointerGetAttribute(
        ctypes.byref(out), _CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ctypes.c_uint64(ptr)
    )
    return rc, _MEMORY_TYPE_NAMES.get(out.value, f"?{out.value}")


def _pick_model_tensor(actor):
    """Largest model parameter — most likely to be in a tag=1 VMM region."""
    best = None
    for p in actor.model.parameters() if hasattr(actor, "model") else []:
        if best is None or p.numel() > best.numel():
            best = p
    if best is None:
        for module in (getattr(actor, "model_chunks", []) or []):
            for p in module.parameters():
                if best is None or p.numel() > best.numel():
                    best = p
    assert best is not None, "could not find a model parameter on the actor"
    return best


def _try(label: str, fn):
    """Run a CUDA op and report whether it raised."""
    try:
        result = fn()
        print(f"  {label}: ok ({result})", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {label}: FAILED -> {type(e).__name__}: {e}", flush=True)
        return False


def main():
    setup()

    import torch

    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    from slime.utils.arguments import parse_args

    args = parse_args()

    actor = MegatronTrainRayActor(
        world_size=1,
        rank=0,
        master_addr="127.0.0.1",
        master_port=int(os.environ["MASTER_PORT"]),
    )

    try:
        print("Initializing actor (allocations under MT_MODEL_OPTIM_TAG=1)...", flush=True)
        actor.init(args, role="actor")
        actor.log_memory("after init")

        print(
            f"\n[allocator] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')} "
            f"PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF', '<unset>')}",
            flush=True,
        )
        print(f"[allocator] torch.cuda.get_allocator_backend() = {torch.cuda.get_allocator_backend()}", flush=True)
        actor.dump_segments("after init")

        import gcr

        pid = os.getpid()
        param = _pick_model_tensor(actor)
        ptr_before = param.data_ptr()
        rc_before, type_before = probe_ptr(ptr_before)
        print(
            f"\n[BEFORE offload_tag] tensor numel={param.numel()} dtype={param.dtype}\n"
            f"  data_ptr={ptr_before:#x}  cuPointerGetAttribute: rc={rc_before} type={type_before}",
            flush=True,
        )

        print(f"\nCalling gcr.offload_tag(pids=[{pid}], tags=[1]) ...", flush=True)
        gcr.offload_tag([pid], [1])
        print("offload_tag returned\n", flush=True)

        ptr_after = param.data_ptr()
        rc_after, type_after = probe_ptr(ptr_after)
        same_va = ptr_after == ptr_before
        print(
            f"[AFTER offload_tag]\n"
            f"  data_ptr={ptr_after:#x}  same_va={same_va}\n"
            f"  cuPointerGetAttribute: rc={rc_after} type={type_after}\n",
            flush=True,
        )

        print("Probing CUDA context state:", flush=True)
        sync_ok = _try("torch.cuda.synchronize() (sticky error?)", lambda: (torch.cuda.synchronize(), "no sticky error")[1])

        print("\nProbing PyTorch caching allocator:", flush=True)
        torch.cuda.empty_cache()
        empty_ok = _try(
            "torch.empty(8, device='cuda').fill_(0); sync()",
            lambda: (
                lambda x: (x.fill_(0), torch.cuda.synchronize(), f"ptr={x.data_ptr():#x}")[-1]
            )(torch.empty(8, device="cuda")),
        )

        print("\nProbing the real failure (CPU→GPU byte copy, same as all_gather_object):", flush=True)
        repro_ok = _try(
            "torch.ByteTensor(b'hello').to('cuda'); sync()",
            lambda: (
                lambda t: (torch.cuda.synchronize(), f"ptr={t.data_ptr():#x}")[-1]
            )(torch.ByteTensor(list(b"hello")).to("cuda")),
        )

        print("\n=== Diagnosis ===", flush=True)
        if rc_after != 0 and not sync_ok:
            print("Case 1: unmap put the context into a sticky error state.", flush=True)
        elif rc_after != 0 and sync_ok and not empty_ok:
            print("Case 2: allocator handed out a pointer into an unmapped segment.", flush=True)
        elif rc_after != 0 and sync_ok and empty_ok and not repro_ok:
            print(
                "Case 1 (deferred): VA is unmapped, fresh allocs are fine, but the .to(device)\n"
                "byte-copy path that crashed in production reproduces here.",
                flush=True,
            )
        elif rc_after == 0:
            print(f"VA still mapped after offload_tag — GCR didn't unmap it. Check tag wiring.", flush=True)
        else:
            print("No failure reproduced — try toggling PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", flush=True)
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
