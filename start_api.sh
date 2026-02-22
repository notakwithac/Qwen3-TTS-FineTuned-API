#!/bin/bash
# Start the fine-tuning API server on a TIR notebook.
# Run this after setup_tir.sh has been executed.
set -e

source .venv/bin/activate 2>/dev/null || true

# Optional: set env vars
export DEVICE="${DEVICE:-cuda:0}"
export USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
export GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-300}"    # 5 min idle unload
export GPU_MAX_CONCURRENCY="${GPU_MAX_CONCURRENCY:-4}" # Concurrent tasks
export GPU_MAX_MODELS="${GPU_MAX_MODELS:-4}"           # LRU Cache size (4 characters)

echo "🚀 Starting Qwen3-TTS Fine-Tuning API on port 8000..."
echo "   Device: $DEVICE"
echo "   Flash Attention: $USE_FLASH_ATTN"
echo ""
echo "   API docs:  http://0.0.0.0:8000/docs"
echo ""

uvicorn api_server:app --host 0.0.0.0 --port 8000
