import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-7B-Instruct"
MODEL_TYPE = "qwen2.5-7B"
# MODEL_NAME = "Qwen3-4B"
# MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8
TP = 4
PP = 2
DP = NUM_GPUS // (TP * PP)  # megatron training data-parallel degree (4)

# sglang per-engine dp-attention degree. Must divide gpus-per-engine (=TP).
# 1 => pure TP inference, dp-attention off. Engines = NUM_GPUS // TP.
SGLANG_DP = 1

WORKER_ENV_VARS = {
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NET_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    # "PYTORCH_ALLOC_CONF": "expandable_segments:False",
    "PYTHONUNBUFFERED": "1",
    "NCCL_DEBUG": "WARN",
    "NCCL_TIMEOUT": "60",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",

    # Per-phase GPU memory tracing (offline stacked-bar analysis).
    # Goes into the ray-job runtime_env -> reaches the driver (train.py phase
    # gating) and the megatron train actors. The sglang subprocess is covered
    # separately via slime/ray/rollout.py env_vars.
    # "SLIME_MEMTRACE": "1",
    # "SLIME_MEMTRACE_DIR": f"/tmp/mem_trace_{MODEL_TYPE}",
    # "SLIME_MEMTRACE_WARMUP": "2",
}


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    model_path = f"/root/models/{MODEL_NAME}"
    assert os.path.isdir(model_path), f"Model not found at {model_path} (expected mounted in /root/models)"
    # U.hf_download_dataset("zhuzilin/gsm8k")
    os.environ["RAY_SILENT_MODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 6 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--global-batch-size 64 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 64 "
        # don't drop zero-std samples
        # "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
    )

    perf_args = (
        f"--tensor-model-parallel-size {TP} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {PP} "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {8192}"
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {TP} "
        f"--sglang-data-parallel-size {SGLANG_DP} "
        f"{'--sglang-enable-dp-attention ' if SGLANG_DP > 1 else ''}"
        f"--sglang-server-concurrency 128 "
        f"--sglang-mem-fraction-static 0.85 "
        f"--sglang-cuda-graph-max-bs 16 "
        # "--sglang-attention-backend triton "
        "--sglang-disable-radix-cache "
        "--sglang-enable-metrics "
        "--sglang-log-level warning "
        "--sglang-disable-custom-all-reduce"
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars=WORKER_ENV_VARS,
    )


if __name__ == "__main__":
    prepare()
    execute()
