#!/usr/bin/env bash
# Run the container with the repository mounted at /workspace.
# GPUs: pass e.g. GPUS=2 to expose only GPU 2 (shared server: never more than 2 GPUs in total).
cd "$(dirname "$0")/.."
[ -f "$HOME/.netrc" ] || touch "$HOME/.netrc"
chmod 600 "$HOME/.netrc"

if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    GPU_FLAG="--gpus \"device=${GPUS:-0}\""
    echo "Exposing GPU(s): ${GPUS:-0}"
else
    GPU_FLAG=""
    echo "No NVIDIA GPU detected. Running without GPU support."
fi

eval docker run -it --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd)":/workspace \
    -v "$HOME/.config/wandb":/home/"$(whoami)"/.config/wandb \
    -v "$HOME/.netrc:/home/$(whoami)/.netrc:rw" \
    -e PYTHONPATH=/workspace \
    $GPU_FLAG \
    --rm cgnn
