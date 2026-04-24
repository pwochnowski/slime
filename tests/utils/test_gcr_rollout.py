"""Minimal GCR rollout-only test.

Spins up SGLang rollout engines with GCR interposition, runs generation
on simple math prompts, then verifies suspend (dump) and resume (restore)
round-trips successfully.

No Megatron training — just SGLang + GCR.

Run:  python tests/utils/test_gcr_rollout.py
"""

import json
import os
import sys
import tempfile

import ray

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_PATH = f"/root/models/{MODEL_NAME}"
NUM_GPUS = 2
BATCH_SIZE = 4


def create_fake_prompts(path, n=8):
    """Create a small JSONL file with simple math prompts."""
    prompts = [
        {"messages": [{"role": "user", "content": f"What is {i} + {i}?"}], "label": str(i + i)}
        for i in range(n)
    ]
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")


def setup(prompt_data_path):
    sys.path.insert(0, "/root/Megatron-LM")
    os.environ["PYTHONPATH"] = "/root/Megatron-LM/"
    # os.environ["RAY_SILENT_MODE"] = "1"
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_NET_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["no_proxy"] = "127.0.0.1"
    os.environ.setdefault("GCR_HOME", "/root/GCR")
    gcr_home = os.environ["GCR_HOME"]
    os.environ.setdefault(
        "GCR_PRELOAD_PATH",
        f"{gcr_home}/GCR/libpreload.so:{gcr_home}/GCR/libcuda.so",
    )

    ray.init(logging_level=1, runtime_env={
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_CUMEM_ENABLE": "1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_NET_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "no_proxy": "127.0.0.1",
            "GCR_HOME": os.environ["GCR_HOME"],
            "GCR_PRELOAD_PATH": os.environ["GCR_PRELOAD_PATH"],
        },
    })
    sys.argv = [
        "test_gcr_rollout.py",
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
        # -- rollout config --
        "--prompt-data", prompt_data_path,
        "--input-key", "messages",
        "--label-key", "label",
        "--apply-chat-template",
        "--rm-type", "math",
        "--rollout-batch-size", str(BATCH_SIZE),
        "--n-samples-per-prompt", "2",
        "--rollout-max-response-len", "64",
        "--rollout-temperature", "0.8",
        "--global-batch-size", str(BATCH_SIZE),
        "--num-rollout", "2",
        # -- sglang --
        "--rollout-num-gpus-per-engine", str(NUM_GPUS),
        "--sglang-mem-fraction-static", "0.6",
        "--sglang-cuda-graph-max-bs", "16",
        "--sglang-enable-gcr", "true",
        # -- training args (needed by parser) --
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
        # -- parallelism --
        "--tensor-model-parallel-size", "1",
        "--sequence-parallel",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        # -- mode --
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(NUM_GPUS),
        "--colocate",
        "--debug-rollout-only",
        "--megatron-to-hf-mode", "bridge",
        "--log-level", "warning",
    ]


def main():
    tmpdir = tempfile.mkdtemp(prefix="gcr_rollout_test_")
    prompt_path = os.path.join(tmpdir, "prompts.jsonl")
    create_fake_prompts(prompt_path)

    setup(prompt_path)

    from slime.utils.arguments import parse_args
    from slime.ray.placement_group import create_placement_groups, create_rollout_manager

    args = parse_args()

    try:
        print("[GCR-DEBUG] Creating placement groups...", flush=True)
        pgs = create_placement_groups(args)

        print("[GCR-DEBUG] Creating rollout manager (starts SGLang engines)...", flush=True)
        rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])

        # Rollout step 0
        print("Step 0: generating rollout data", flush=True)
        ray.get(rollout_manager.generate.remote(0))

        # Suspend — GCR dumps all CUDA memory from SGLang engines
        print("GCR: suspending rollout engines", flush=True)
        ray.get(rollout_manager.gcr_suspend.remote())

        # Resume — GCR restores CUDA memory
        print("GCR: resuming rollout engines", flush=True)
        ray.get(rollout_manager.gcr_resume.remote())

        # Rollout step 1 — verify engines still work after restore
        print("Step 1: generating rollout data (post-restore)", flush=True)
        ray.get(rollout_manager.generate.remote(1))

        print("PASS: GCR rollout checkpoint/restore round-trip succeeded", flush=True)
    finally:
        ray.get(rollout_manager.dispose.remote())
        ray.shutdown()


if __name__ == "__main__":
    main()
