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
    echo "[ERROR] .venv not found. Please run setup_tir.sh first." | tee -a "${LOG_DIR}/startup.log"
    exit 1
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

# --- 4. EXECUTION ---
echo "[INFO] Starting Uvicorn API server on 0.0.0.0:8000..." | tee -a "${LOG_DIR}/startup.log"

# Use exec to ensure the process replaces the shell (better for systemd)
exec uvicorn api_server:app --host 0.0.0.0 --port 8000 >> "${LOG_DIR}/api_output.log" 2>&1
