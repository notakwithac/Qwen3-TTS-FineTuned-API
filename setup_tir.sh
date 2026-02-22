#!/bin/bash
# =============================================================
# E2E Networks TIR — One-time environment setup
# Run this ONCE after cloning the repo inside the TIR notebook.
# =============================================================
set -e

echo "=== [1/5] Installing uv package manager ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/5] Creating Python 3.11 virtual environment ==="
uv venv --python 3.11 .venv
source .venv/bin/activate

echo "=== [3/5] Installing project dependencies ==="
# On Linux we install flash-attn from PyPI (builds from source),
# so we override the Windows-only .whl source.
uv sync --extra cu128 --override pyproject-linux.toml

echo "=== [4/5] Verifying GPU access ==="
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=== [5/5] Downloading Qwen3-TTS models from HuggingFace ==="
python download_models.py

echo ""
echo "✅  Setup complete!  Next steps:"
echo "   1. Set E2E storage credentials (optional):"
echo "      export E2E_ACCESS_KEY=your_key"
echo "      export E2E_SECRET_KEY=your_secret"
echo "   2. Start the API server:"
echo "      bash start_api.sh"
echo "   3. API docs at: http://0.0.0.0:8000/docs"
