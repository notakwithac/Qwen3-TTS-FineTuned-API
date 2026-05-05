#!/bin/bash
set -euo pipefail

if [ "${DOCKER_PRELOAD_SARVAM_TRANSLATE:-1}" = "1" ]; then
    echo "Preloading Sarvam-Translate into the Docker Hugging Face cache..."
    python download_models.py --models sarvam_translate
else
    echo "Skipping Sarvam-Translate preload because DOCKER_PRELOAD_SARVAM_TRANSLATE=${DOCKER_PRELOAD_SARVAM_TRANSLATE:-0}"
fi

exec python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
