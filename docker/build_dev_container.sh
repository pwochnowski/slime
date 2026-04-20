#!/bin/bash

set -e

# Pull the latest image
docker pull slimerl/slime:latest

# Start the container
docker run --rm --gpus all --privileged --ipc=host --name slime \
    -v /data/:/root/.cache/huggingface \
    -v /home/paul/repos/GCR:/root/GCR \
    -v /home/paul/repos/slime:/root/slime \
    -v /home/paul/repos/Megatron-LM:/root/Megatron-LM \
    -v /home/paul/repos/sglang:/sgl-workspace/sglang \
    -v /mnt/huge:/mnt/huge \
    --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -td slimerl/slime:latest /bin/zsh

# Install packages, build GCR, configure shell
docker exec slime bash /root/slime/docker/init_container.sh

echo "Finished setup"