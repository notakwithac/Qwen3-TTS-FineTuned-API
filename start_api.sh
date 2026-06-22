#!/bin/bash
# Start the fine-tuning API server on a TIR notebook.
# Run this after setup_tir.sh has been executed.

. .venv/bin/activate 2>/dev/null || true

# Optional: set env vars
export DEVICE="${DEVICE:-cuda:0}"
export USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
export GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-300}"    # 5 min idle unload
export GPU_MAX_CONCURRENCY="${GPU_MAX_CONCURRENCY:-4}" # Concurrent tasks
export GPU_MAX_MODELS="${GPU_MAX_MODELS:-4}"           # LRU Cache size (4 characters)

# Ensure PyTorch native libs are discoverable for flash-attn / other CUDA extensions.
TORCH_LIB_DIR="$(python - <<'PY'
import os
try:
    import torch
    print(os.path.join(os.path.dirname(torch.__file__), "lib"))
except Exception:
    print("")
PY
)"
if [ -n "${TORCH_LIB_DIR}" ] && [ -d "${TORCH_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:/opt/conda/lib:${LD_LIBRARY_PATH}"
    echo "   Torch native libs: ${TORCH_LIB_DIR}"
fi

# Clean up any stale signal file from a previous run
if [ -f terminate_signal.tmp ]; then
    echo "⚠️  Removing stale terminate_signal.tmp from previous run"
    rm -f terminate_signal.tmp
fi

echo "🚀 Starting GPU Idle Watchdog..."
mkdir -p logs

# Launch watchdog: tee mirrors output to both logs/watchdog.log AND stdout
# so watchdog logs appear in the same stream as the API logs.
nohup python gpu_idle_watchdog.py 2>&1 | tee -a logs/watchdog.log &
WATCHDOG_PID=$!
sleep 3

if kill -0 $WATCHDOG_PID 2>/dev/null; then
    echo "✅ Watchdog started OK (PID: $WATCHDOG_PID)"
else
    echo "❌ ERROR: Watchdog process died immediately after launch!"
    echo "   Check logs/watchdog.log for the error."
    tail -20 logs/watchdog.log
fi

echo "🚀 Starting Qwen3-TTS Fine-Tuning API on port 8000..."
echo "   Device: $DEVICE"
echo "   Flash Attention: $USE_FLASH_ATTN"
echo ""
echo "   API docs:  http://0.0.0.0:8000/docs"
echo ""

python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
