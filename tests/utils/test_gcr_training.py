"""Minimal GCR checkpoint/restore test.

Spins up a Megatron actor with GCR interposition, trains on fake data
to create real GPU allocations, then verifies suspend (dump) and
resume (restore) round-trips successfully.

No sglang, no rollout manager — just MT + GCR.

Run:  python tests/utils/test_gcr.py
"""

import sys
import os
import ray

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_PATH = f"/root/models/{MODEL_NAME}"
NUM_GPUS = 1
SEQ_LEN = 128
BATCH_SIZE = 4


def setup():
    sys.path.insert(0, "/root/Megatron-LM")
    os.environ["PYTHONPATH"] = "/root/Megatron-LM/"
    os.environ["RAY_SILENT_MODE"] = "1"
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["no_proxy"] = "127.0.0.1"
    os.environ.setdefault("GCR_HOME", "/root/GCR")
    ray.init(logging_level=1, runtime_env={
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_CUMEM_ENABLE": "1",
            "MASTER_ADDR": "127.0.0.1",
            # "RAY_LD_PRELOAD_ON_WORKERS": "1",
            "no_proxy": "127.0.0.1",
            "GCR_HOME": os.environ["GCR_HOME"],
        },
    })
    sys.argv = [
        "test_gcr.py",
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

    try:
        print("[GCR-DEBUG] Creating placement groups...", flush=True)
        pgs = create_placement_groups(args)
        print("[GCR-DEBUG] Placement groups created. Allocating train group...", flush=True)
        actor_model = allocate_train_group(args, args.actor_num_nodes, args.actor_num_gpus_per_node, pgs["actor"])
        print("[GCR-DEBUG] Train group allocated (actors created). Calling async_init...", flush=True)
        ray.get(actor_model.async_init(args, role="actor"))
        print("[GCR-DEBUG] async_init DONE", flush=True)

        dp_size = 1  # NUM_GPUS with TP=1, PP=1

        actor_model.log_memory("after init")

        # Train step 0 — creates real GPU allocations
        print("Step 0: training on fake data", flush=True)
        data_ref = make_fake_rollout_data(dp_size)
        ray.get(actor_model.async_train(0, data_ref))
        actor_model.log_memory("after train step 0")

        # Suspend — GCR dumps all CUDA memory to hugepages
        print("GCR: suspending MT", flush=True)
        actor_model.gcr_suspend()

        # Resume — GCR restores CUDA memory
        print("GCR: resuming MT", flush=True)
        actor_model.gcr_resume()
        actor_model.log_memory("after resume")

        # Train step 1 — verify the model still works after restore
        print("Step 1: training on fake data (post-restore)", flush=True)
        data_ref = make_fake_rollout_data(dp_size)
        ray.get(actor_model.async_train(1, data_ref))
        actor_model.log_memory("after train step 1")

        print("PASS: GCR checkpoint/restore round-trip succeeded", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
