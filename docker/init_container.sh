#!/bin/bash
set -e

# Mark repos as safe for git
git config --global --add safe.directory /root/slime
git config --global --add safe.directory /root/Megatron-LM
git config --global --add safe.directory /sgl-workspace/sglang

# Install local packages in editable mode
pip install -e /root/slime
pip install -e /root/Megatron-LM
pip install -e /root/GCR
pip install -e /sgl-workspace/sglang/python

# Build GCR
CUDA_ROOT='/usr/local/cuda' make -C /root/GCR/GCR clean all

# Setup zsh prompt with docker indicator
cat >> /root/.zshrc << 'ZSHEOF'
function docker_prompt_info() {
  if [ -f /.dockerenv ]; then
    echo "🐳 "
  fi
}

setopt prompt_subst
PROMPT='$(docker_prompt_info)'$PROMPT

export CUDA_ROOT='/usr/local/cuda'
export PATH="/root/GCR/GCR:$PATH"
export GCR_HOME="/root/GCR"
ZSHEOF
