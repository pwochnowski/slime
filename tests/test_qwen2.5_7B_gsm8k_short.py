import os
import slime.utils.external_utils.command_utils as U

# MODEL_NAME = "Qwen3-4B"
# MODEL_TYPE = "qwen3-4B"
MODEL_NAME = "Qwen2.5-7B-Instruct"
MODEL_TYPE = "qwen2.5-7B"

NUM_GPUS = 4
TP = 4
PP = 1
DP = NUM_GPUS // (TP * PP)  # megatron training data-parallel degree

# sglang per-engine dp-attention degree. Must divide gpus-per-engine (=TP).
# 1 => pure TP inference, dp-attention off. Engines = NUM_GPUS // TP.
SGLANG_DP = 1

WORKER_ENV_VARS = {
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NET_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    # "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True",
    # "TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC": "0",
    "PYTHONUNBUFFERED": "1",
    "GCR_KEEP_RESIDENT": "1",
    "GCR_HUGETLB_DIR": "/mnt/huge",
}


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"ln -sfn $(HF_HUB_OFFLINE=1 hf download Qwen/{MODEL_NAME}) /root/models/{MODEL_NAME}")
    # U.hf_download_dataset("zhuzilin/gsm8k")
    os.environ["RAY_SILENT_MODE"] = "1"
    os.environ["GCR_HOME"] = "/root/GCR"
    os.environ["GCR_PRELOAD_PATH"] = "/root/GCR/GCR/libpreload.so:/root/GCR/GCR/libcuda.so"
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
        f"--sglang-mem-fraction-static 0.9 "
        f"--sglang-data-parallel-size {SGLANG_DP} "
        f"{'--sglang-enable-dp-attention ' if SGLANG_DP > 1 else ''}"
        f"--sglang-cuda-graph-max-bs 16 "
        # "--sglang-attention-backend triton "
        "--sglang-disable-radix-cache "
        "--sglang-enable-metrics "
        "--sglang-enable-gcr "
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
