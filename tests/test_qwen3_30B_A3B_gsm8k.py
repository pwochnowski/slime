import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8

TP_SIZE = 2
PP_SIZE = 2
CP_SIZE = 2
EP_SIZE = 2
ETP_SIZE = 1

WORKER_ENV_VARS = {
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NET_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "PYTHONUNBUFFERED": "1",
}


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    os.environ["RAY_SILENT_MODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # don't need without --ref
    # U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=TP_SIZE * PP_SIZE, tensor_model_parallel_size=TP_SIZE, extra_args=f"--expert-tensor-parallel-size {ETP_SIZE}")


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        # f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--global-batch-size 16 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {TP_SIZE} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {PP_SIZE} "
        f"--context-parallel-size {CP_SIZE} "
        f"--expert-model-parallel-size {EP_SIZE} "
        f"--expert-tensor-parallel-size {ETP_SIZE} "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {NUM_GPUS} "
        f"--sglang-pipeline-parallel-size {PP_SIZE} "
        f"--sglang-expert-parallel-size {EP_SIZE} "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-cuda-graph-max-bs 32 "
        "--sglang-max-running-requests 512 "
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
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--moe-token-dispatcher-type alltoall "
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
