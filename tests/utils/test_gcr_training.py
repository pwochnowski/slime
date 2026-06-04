"""Minimal GCR training-only test for Qwen2.5-7B.

Spins up a Megatron actor with GCR interposition, trains on fake data.
No sglang, no rollout manager — just MT + GCR.

Run:  python tests/utils/test_gcr_training.py
"""

import sys
import os
import ray

MODEL_NAME = "Qwen2.5-7B-Instruct"
MODEL_PATH = f"/root/models/{MODEL_NAME}"
NUM_GPUS = 4
TP = 4
PP = 1
SEQ_LEN = 512
BATCH_SIZE = 4


def setup():
    sys.path.insert(0, "/root/Megatron-LM")
    os.environ["PYTHONPATH"] = "/root/Megatron-LM/"
    os.environ["RAY_SILENT_MODE"] = "1"
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    os.environ["NCCL_NET_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["no_proxy"] = "127.0.0.1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("GCR_HOME", "/root/GCR")
    ray.init(logging_level=1, runtime_env={
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "NCCL_CUMEM_ENABLE": "1",
            "NCCL_NET_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "no_proxy": "127.0.0.1",
            "GCR_HOME": os.environ["GCR_HOME"],
            "PYTHONUNBUFFERED": "1",
        },
    })
    sys.argv = [
        "test_gcr_training.py",
        # -- model architecture (Qwen2.5-7B) --
        "--swiglu",
        "--num-layers", "28",
        "--hidden-size", "3584",
        "--ffn-hidden-size", "18944",
        "--num-attention-heads", "28",
        "--group-query-attention",
        "--num-query-groups", "4",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000",
        "--vocab-size", "152064",
        "--untie-embeddings-and-output-weights",
        # -- checkpoint --
        "--hf-checkpoint", MODEL_PATH,
        "--ref-load", MODEL_PATH,
        # -- training --
        "--rollout-batch-size", str(BATCH_SIZE),
        "--global-batch-size", str(BATCH_SIZE),
        "--advantage-estimator", "grpo",
        "--use-kl-loss",
        "--kl-loss-coef", "0.00",
        "--kl-loss-type", "low_var_kl",
        "--entropy-coef", "0.00",
        "--eps-clip", "0.2",
        "--eps-clip-high", "0.28",
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
        "--max-tokens-per-gpu", "512",
        "--recompute-granularity", "full",
        # -- parallelism --
        "--tensor-model-parallel-size", str(TP),
        "--sequence-parallel",
        "--pipeline-model-parallel-size", str(PP),
        "--context-parallel-size", "1",
        # -- mode --
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(NUM_GPUS),
        "--colocate",
        "--debug-train-only",
        "--num-rollout", "2",
        "--megatron-to-hf-mode", "bridge",
        "--log-level", "warning",
    ]


def make_fake_rollout_data(dp_size):
    """Build a minimal rollout_data_ref list (one Box per DP rank)."""
    import random
    from slime.utils.misc import Box

    prompt_len = 32
    resp_len = SEQ_LEN - prompt_len
    total_lengths = [SEQ_LEN] * BATCH_SIZE

    refs = []
    for dp_rank in range(dp_size):
        partition = list(range(dp_rank, BATCH_SIZE, dp_size))
        n = len(partition)
        refs.append(Box(ray.put({
            "partition": partition,
            "tokens": [[random.randint(0, 1000) for _ in range(SEQ_LEN)] for _ in range(n)],
            "loss_masks": [[1] * resp_len for _ in range(n)],
            "response_lengths": [resp_len] * n,
            "total_lengths": total_lengths,
            "rewards": [1.0] * n,
            "truncated": [0] * n,
        })))
    return refs


def main():
    setup()

    from slime.utils.arguments import parse_args
    from slime.ray.placement_group import allocate_train_group, create_placement_groups

    args = parse_args()
    dp_size = NUM_GPUS // (TP * PP)

    try:
        print("[test] Creating placement groups...", flush=True)
        pgs = create_placement_groups(args)
        print("[test] Allocating train group...", flush=True)
        actor_model = allocate_train_group(args, args.actor_num_nodes, args.actor_num_gpus_per_node, pgs["actor"])
        print("[test] Calling async_init...", flush=True)
        ray.get(actor_model.async_init(args, role="actor", with_ref=True))
        print("[test] async_init DONE", flush=True)

        actor_model.log_memory("after init")

        print("Step 0: training on fake data", flush=True)
        data_ref = make_fake_rollout_data(dp_size)
        ray.get(actor_model.async_train(0, data_ref))
        actor_model.log_memory("after train step 0")

        print("GCR: suspending MT", flush=True)
        actor_model.gcr_suspend()

        print("GCR: resuming MT", flush=True)
        actor_model.gcr_resume()
        actor_model.log_memory("after resume")

        print("Step 1: training on fake data (post-restore)", flush=True)
        data_ref = make_fake_rollout_data(dp_size)
        ray.get(actor_model.async_train(1, data_ref))
        actor_model.log_memory("after train step 1")

        print("PASS", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()