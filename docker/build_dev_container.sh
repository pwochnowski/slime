#!/bin/bash

set -e

# Pull the latest image
# docker pull slimerl/slime:latest

# Start the container
docker run --rm --gpus all --privileged --ipc=host --name slime_og \
    -v /mnt/data/huggingface-cache:/root/.cache/huggingface \
    -v /home/paul/repos/slime_og:/root/slime \
    -v /home/paul/repos/Megatron-LM:/root/Megatron-LM \
    -v /home/paul/repos/sglang_og:/sgl-workspace/sglang \
    -v /mnt/huge:/mnt/huge \
    --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -td slimerl/slime:working /bin/zsh

# Install packages, build GCR, configure shell
docker exec slime_og bash /root/slime/docker/init_container.sh

echo "Finished setup"
