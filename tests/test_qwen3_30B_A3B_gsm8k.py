import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8

# Megatron (training) parallelism
MEGATRON_TP_SIZE = 2
MEGATRON_PP_SIZE = 2
MEGATRON_CP_SIZE = 2
MEGATRON_EP_SIZE = 2
MEGATRON_ETP_SIZE = 2

# SGLang (rollout) parallelism
SGLANG_PP_SIZE = 2
SGLANG_EP_SIZE = 2

WORKER_ENV_VARS = {
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NET_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "PYTHONUNBUFFERED": "1",
    "GCR_KEEP_RESIDENT": "1",
    "GCR_PREFAULT_MB": "81920",
}


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    os.environ["RAY_SILENT_MODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["GCR_HOME"] = "/root/GCR"
    os.environ["GCR_PRELOAD_PATH"] = "/root/GCR/GCR/libpreload.so:/root/GCR/GCR/libcuda.so"
    os.environ["GCR_PREFAULT_MB"] = "81920"



def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/models/{MODEL_NAME} "
    )

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 5 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--global-batch-size 16 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {MEGATRON_TP_SIZE} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {MEGATRON_PP_SIZE} "
        f"--context-parallel-size {MEGATRON_CP_SIZE} "
        f"--expert-model-parallel-size {MEGATRON_EP_SIZE} "
        f"--expert-tensor-parallel-size {MEGATRON_ETP_SIZE} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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
        # "--optimizer-cpu-offload "
        # "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {NUM_GPUS} "
        f"--sglang-pipeline-parallel-size {SGLANG_PP_SIZE} "
        f"--sglang-expert-parallel-size {SGLANG_EP_SIZE} "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-cuda-graph-max-bs 32 "
        "--sglang-max-running-requests 512 "
        "--sglang-enable-metrics "
        "--sglang-log-level warning "
        "--sglang-disable-custom-all-reduce "
        "--sglang-enable-gcr "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--moe-token-dispatcher-type alltoall "
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
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
