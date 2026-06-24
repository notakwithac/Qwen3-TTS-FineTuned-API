#!/bin/bash
# robust startup script for Qwen3-TTS API on E2E Networks TIR
# This script is intended to be called by systemd or as a Startup Script.

# --- 1. SETTINGS ---
# Adjust this to your repository's absolute path
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Ensure logs directory exists
mkdir -p "${LOG_DIR}"

echo "[${TIMESTAMP}] 🚀 Initializing Qwen3-TTS Startup..." | tee -a "${LOG_DIR}/startup.log"

# --- 2. NAVIGATION ---
cd "${PROJECT_DIR}" || {
    echo "[ERROR] Could not change directory to ${PROJECT_DIR}" | tee -a "${LOG_DIR}/startup.log"
    exit 1
}

# --- 3. ENVIRONMENT ---
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment..." | tee -a "${LOG_DIR}/startup.log"
    source .venv/bin/activate
else
    echo "[INFO] .venv not found. Using the container's default Python environment." | tee -a "${LOG_DIR}/startup.log"
fi

# Load .env if it exists
if [ -f ".env" ]; then
    echo "[INFO] Loading environment variables from .env..." | tee -a "${LOG_DIR}/startup.log"
    export $(grep -v '^#' .env | xargs)
fi

# Set defaults if not provided in .env
export DEVICE="${DEVICE:-cuda:0}"
export USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
export GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-300}"
export GPU_MAX_MODELS="${GPU_MAX_MODELS:-4}"

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
    echo "[INFO] Added Torch native lib path to LD_LIBRARY_PATH: ${TORCH_LIB_DIR}" | tee -a "${LOG_DIR}/startup.log"
fi

# Pre-download model weights into the Hugging Face cache. This does not load
# models into GPU; it only avoids first-request network downloads.
export PREFETCH_MODELS_ON_START="${PREFETCH_MODELS_ON_START:-1}"
export PREFETCH_MODEL_SET="${PREFETCH_MODEL_SET:-base voice_design tokenizer gemma sarvam_translate}"
if [ "${PREFETCH_MODELS_ON_START}" = "1" ]; then
    echo "[INFO] Pre-fetching model files: ${PREFETCH_MODEL_SET}" | tee -a "${LOG_DIR}/startup.log"
    if python download_models.py --models ${PREFETCH_MODEL_SET} >> "${LOG_DIR}/model_prefetch.log" 2>&1; then
        echo "[INFO] Model prefetch complete." | tee -a "${LOG_DIR}/startup.log"
    else
        echo "[WARN] Model prefetch failed; continuing API startup. See ${LOG_DIR}/model_prefetch.log" | tee -a "${LOG_DIR}/startup.log"
    fi
else
    echo "[INFO] Model prefetch skipped (PREFETCH_MODELS_ON_START=${PREFETCH_MODELS_ON_START})." | tee -a "${LOG_DIR}/startup.log"
fi

# --- 4. EXECUTION ---
echo "[INFO] Starting GPU Idle Watchdog..." | tee -a "${LOG_DIR}/startup.log"
nohup python gpu_idle_watchdog.py >> "${LOG_DIR}/watchdog.log" 2>&1 &

echo "[INFO] Starting Uvicorn API server on 0.0.0.0:8000..." | tee -a "${LOG_DIR}/startup.log"

# Use exec to ensure the process replaces the shell (better for systemd)
exec uvicorn api_server:app --host 0.0.0.0 --port 8000 >> "${LOG_DIR}/api_output.log" 2>&1
