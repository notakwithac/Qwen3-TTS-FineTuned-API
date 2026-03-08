#!/bin/bash
set -e

echo "=== [1/6] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/6] Creating Python 3.11 venv ==="
uv venv --python 3.11 .venv
source .venv/bin/activate

echo "=== [3/6] Syncing stable CUDA 12.1 deps ==="
uv sync --extra cu121

echo "=== [4/6] Limiting build to A100 arch ==="
export TORCH_CUDA_ARCH_LIST="8.0"
export GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-300}"
export GPU_MAX_MODELS="${GPU_MAX_MODELS:-4}"

echo "=== [5/6] Installing flash-attn ==="
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "=== [6/6] Verifying GPU ==="
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"