import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-7B-Instruct"
MODEL_TYPE = "qwen2.5-7B"
# MODEL_NAME = "Qwen3-4B"
# MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 4
TP = 4
PP = 1
DP = NUM_GPUS // (TP * PP)

WORKER_ENV_VARS = {
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NET_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    # "PYTORCH_ALLOC_CONF": "expandable_segments:False",
    "PYTHONUNBUFFERED": "1",
    "NCCL_DEBUG": "WARN",
    "NCCL_TIMEOUT": "60",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"ln -sfn $(HF_HUB_OFFLINE=1 hf download Qwen/{MODEL_NAME}) /root/models/{MODEL_NAME}")
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
        "--rm-type random "
        "--num-rollout 2 "
        "--rollout-batch-size 2 "
        "--n-samples-per-prompt 4 "
        "--global-batch-size 4 "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        # don't drop zero-std samples
        # "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
    )

    perf_args = (
        f"--tensor-model-parallel-size {TP} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {PP} "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {512} "
        "--recompute-granularity full"
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
        f"--sglang-data-parallel-size {DP} "
        f"{'--sglang-enable-dp-attention ' if DP > 1 else ''}"
        f"--sglang-mem-fraction-static 0.8 "
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
