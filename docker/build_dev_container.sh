# Pull the latest image
docker pull slimerl/slime:latest

# Start the container
docker run --rm --gpus all --privileged --ipc=host --name slime \
    -v /data/:/root/.cache/huggingface \
    -v /home/paul/repos/GCR:/root/GCR \
    -v /home/paul/repos/slime:/root/slime \
    -v /home/paul/repos/Megatron-LM:/root/Megatron-LM \
    -v /home/paul/repos/sglang:/sgl-workspace/sglang \
    --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -td slimerl/slime:latest /bin/zsh

docker exec slime bash -c 'cd /root/slime && git config --global --add safe.directory /root/slime && pip install -e .; cd /root/Megatron-LM && git config --global --add safe.directory /root/Megatron-LM  && pip install -e .; cd /sgl-workspace/sglang/python && git config --global --add safe.directory /sgl-workspace/sglang && pip install -e .'

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
