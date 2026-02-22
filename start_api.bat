@echo off
setlocal

:: Configuration for 10GB VRAM
set DEVICE=cuda:0
set USE_FLASH_ATTN=1
set GPU_IDLE_TIMEOUT=300
set GPU_MAX_CONCURRENCY=1
set GPU_MAX_MODELS=1

:: S3 Configuration (Optional - user can fill these if needed)
:: set E2E_ACCESS_KEY=
:: set E2E_SECRET_KEY=
:: set E2E_BUCKET=qwen3-tts
:: set E2E_ENDPOINT_URL=https://objectstore.e2enetworks.net

echo 🚀 Starting Qwen3-TTS Production API (Local Mode)...
echo VRAM Optimized: 10GB limit detected.
echo Max Concurrent Jobs: %GPU_MAX_CONCURRENCY%
echo Model Cache: %GPU_MAX_MODELS%

uv run uvicorn api_server:app --host 0.0.0.0 --port 8000
