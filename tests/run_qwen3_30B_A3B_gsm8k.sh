#!/bin/bash
#SBATCH --job-name=qwen3-30B-A3B-gsm8k
#SBATCH --partition=j03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -ex

echo "Job $SLURM_JOB_ID on node $(hostname), partition $SLURM_JOB_PARTITION"
nvidia-smi || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR/.."

bash docker/run_cluster.sh python tests/test_qwen3_30B_A3B_gsm8k.py
echo "Test exited with code $?"

echo "Holding node — SSH in to iterate. Will release when job time expires."
sleep infinity
