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

echo "=== [5/6] Installing flash-attn ==="
# Check if there is a linux wheel for flash-attn in the current directory
FLASH_WHEEL=$(ls flash_attn-*.whl 2>/dev/null | grep "manylinux" || true)
if [ -n "$FLASH_WHEEL" ]; then
    echo "Found local flash-attn wheel: $FLASH_WHEEL. Installing..."
    uv pip install "$FLASH_WHEEL"
elif [ -f "flash_attn_linux.whl" ]; then
    echo "Found flash_attn_linux.whl. Installing..."
    uv pip install flash_attn_linux.whl
else
    echo "No local Linux wheel found. Attempting to install from PyPI (this may take a long time to compile)..."
    # uv pip install flash-attn --no-build-isolation
    echo "Skipping compilation to avoid OOM. Please provide a pre-built wheel if possible."
fi

echo "=== [6/6] Verifying GPU ==="
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"