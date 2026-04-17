# Pull the latest image
docker pull slimerl/slime:latest

# Start the container
docker run --rm --gpus all --privileged --ipc=host --name slime \
    -v /data/:/root/.cache/huggingface \
    -v /home/paul/repos/slime:/root/slime \
    -v /home/paul/repos/Megatron-LM:/root/Megatron-LM \
    -v /home/paul/repos/sglang:/sgl-workspace/sglang \
    --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -td slimerl/slime:latest /bin/zsh

# Setup zsh prompt inside container
docker exec slime bash -c 'cat >> /root/.zshrc << '\''ZSHEOF'\''
function docker_prompt_info() {
  if [ -f /.dockerenv ]; then
    echo "🐳 "
  fi
}

setopt prompt_subst
PROMPT='\''$(docker_prompt_info)'\''$PROMPT
ZSHEOF'
