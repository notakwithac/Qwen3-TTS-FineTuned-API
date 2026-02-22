#!/bin/bash
set -e

echo "=== [1/6] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/6] Creating Python 3.10 venv ==="
uv venv --python 3.10 .venv
source .venv/bin/activate

echo "=== [3/6] Syncing stable CUDA 12.1 deps ==="
uv sync --extra cu121

echo "=== [4/6] Limiting build to A100 arch ==="
export TORCH_CUDA_ARCH_LIST="8.0"

echo "=== [5/6] Installing flash-attn skipped ==="
#uv pip install flash-attn --no-build-isolation

echo "=== [6/6] Verifying GPU ==="
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"