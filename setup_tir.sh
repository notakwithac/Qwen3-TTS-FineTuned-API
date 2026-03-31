#!/bin/bash
set -e

echo "=== [1/6] Installing uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed, skipping."
fi
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/6] Creating Python 3.11 venv ==="
if [ ! -d ".venv" ]; then
    uv venv --python 3.11 .venv
else
    echo ".venv already exists, skipping."
fi
. .venv/bin/activate

echo "=== [3/6] Syncing stable CUDA 12.1 deps ==="
uv sync --extra cu121

echo "=== [4/6] Limiting build to A100 arch ==="
export TORCH_CUDA_ARCH_LIST="8.0"
export GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-300}"
export GPU_MAX_MODELS="${GPU_MAX_MODELS:-4}"

echo "=== [5/6] Installing flash-attn ==="
if ! python -c "import flash_attn" &> /dev/null; then
    uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
else
    echo "flash-attn is already installed, skipping."
fi

echo "=== [6/6] Verifying GPU ==="
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"