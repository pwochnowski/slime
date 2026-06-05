#!/bin/bash
#
# Run a command inside the slime container on a cluster node using Singularity.
#
# Usage:
#   bash docker/run_cluster.sh <command>
#   bash docker/run_cluster.sh python tests/test_qwen3_30B_A3B_gsm8k.py
#
# Expects WORK_DIR env var pointing to the parent of your repo checkouts.
# Defaults to ~/WORK/paulw.

set -ex

WORK_DIR="${WORK_DIR:-$HOME/WORK/paulw}"
CACHE_DIR="${WORK_DIR}/.cache/huggingface"
SIF_PATH="${WORK_DIR}/slime.sif"
DOCKER_IMAGE="${SLIME_DOCKER_IMAGE:-slimerl/slime@sha256:453121f6e6acb3342077ed6a891a6be4a407728584307e1f0ed9e98973f4f4c4}"

mkdir -p "$CACHE_DIR"

if [ ! -f "$SIF_PATH" ]; then
    echo "Building SIF image (one-time)..."
    singularity pull "$SIF_PATH" "docker://${DOCKER_IMAGE}"
fi

GCR_OFFLOAD="${WORK_DIR}/.gcr_offload"
SING_TMP="${WORK_DIR}/.singularity/tmp"
SING_ROOT="${WORK_DIR}/.singularity/root"
SING_HUGE="/dev/shm/${USER}/huge"
mkdir -p "$GCR_OFFLOAD" "$SING_TMP" "$SING_ROOT" "$SING_HUGE"

singularity exec --nv --contain --writable-tmpfs \
    --bind "$SING_TMP":/tmp \
    --bind "$SING_ROOT":/root \
    --bind "$CACHE_DIR":/root/.cache/huggingface \
    --bind "$WORK_DIR/models":/root/models \
    --bind "$WORK_DIR/GCR":/root/GCR \
    --bind "$WORK_DIR/slime":/root/slime \
    --bind "$WORK_DIR/Megatron-LM":/root/Megatron-LM \
    --bind "$WORK_DIR/sglang":/sgl-workspace/sglang \
    --bind "$GCR_OFFLOAD":/root/gcr_offload \
    --bind "$SING_HUGE":/mnt/huge \
    --bind /lib/x86_64-linux-gnu/libcuda.so:/usr/lib/x86_64-linux-gnu/libcuda.so \
    --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    "$SIF_PATH" bash -c "
        set -ex
        export HOME=/root
        export PIP_BREAK_SYSTEM_PACKAGES=1
        cd /root/slime

        git config --global --add safe.directory /root/slime
        git config --global --add safe.directory /root/Megatron-LM
        git config --global --add safe.directory /sgl-workspace/sglang
        export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
        export PIP_TRUSTED_HOST=mirrors.tuna.tsinghua.edu.cn
        PIP_CACHE=/root/models/.pip_cache
        mkdir -p \$PIP_CACHE
        pip install --cache-dir \$PIP_CACHE -e /root/slime
        pip install --cache-dir \$PIP_CACHE -e /root/Megatron-LM
        pip install --cache-dir \$PIP_CACHE -e /root/GCR
        pip install --cache-dir \$PIP_CACHE -e /sgl-workspace/sglang/python

        export PYTHONUNBUFFERED=1
        export CUDA_ROOT='/usr/local/cuda' 
        export PATH="$CUDA_ROOT/bin:/root/GCR/GCR:$PATH"
        export GCR_HOME="/root/GCR"
        export GCR_OFFLOAD_DIR="/root/gcr_offload"
        export FLASHINFER_DISABLE_VERSION_CHECK=1
        export CXX=g++
        make CXX=g++ -C /root/GCR/GCR clean all

        $*
    "
