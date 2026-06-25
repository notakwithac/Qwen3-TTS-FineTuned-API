#!/usr/bin/env python3
# coding=utf-8
"""Download all Qwen3-TTS models from HuggingFace.

Run this ONCE on the TIR instance to pre-cache all models before starting the API.
This avoids download delays during fine-tuning or inference.

Usage:
    python download_models.py                    # download all
    python download_models.py --models base      # download only base
    python download_models.py --cache-dir /data/models  # custom cache dir

Models downloaded:
    1. Qwen/Qwen3-TTS-12Hz-1.7B-Base          (~3.5 GB) — for fine-tuning
    2. Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   (~3.5 GB) — for custom voice inference
    3. Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign   (~3.5 GB) — for voice design inference
    4. Qwen/Qwen3-TTS-Tokenizer-12Hz          (~400 MB) — for audio tokenization
    5. Gemma e4b                              — for LLM extraction via vLLM
    6. sarvamai/sarvam-translate              — for /translate via vLLM
"""

import argparse
import logging
import os
import time
from typing import Any

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MODELS = {
    "base": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "desc": "Base model (for fine-tuning)",
        "size": "~3.5 GB",
    },
    "custom_voice": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "desc": "CustomVoice model (inference with fine-tuned speakers)",
        "size": "~3.5 GB",
    },
    "voice_design": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "desc": "VoiceDesign model (generate voices from text descriptions)",
        "size": "~3.5 GB",
    },
    "tokenizer": {
        "repo": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "desc": "Audio tokenizer (data preparation)",
        "size": "~400 MB",
    },
    "gemma": {
        "repo": os.environ.get("GEMMA_VLLM_HF_MODEL", "google/gemma-4-E4B"),
        "desc": "Gemma e4b model (lazy vLLM LLM proxy)",
        "size": "large",
        "skip_if_unauthenticated": True,
    },
    "sarvam_translate": {
        "repo": os.environ.get("SARVAM_VLLM_HF_MODEL", "sarvamai/sarvam-translate"),
        "desc": "Sarvam Translate model (lazy vLLM translation proxy)",
        "size": "large",
    },
}


def _is_access_denied_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in (401, 403):
        return True

    message = str(exc).lower()
    return (
        "gated repo" in message
        or "access to model" in message
        or "restricted" in message
    )


def download_model(
    name: str, info: dict[str, Any], cache_dir: str = None
) -> dict[str, Any]:
    """Download a single model."""
    log.info(f"{'=' * 60}")
    log.info(f"Downloading: {info['repo']}")
    log.info(f"  Description: {info['desc']}")
    log.info(f"  Size: {info['size']}")

    t0 = time.time()
    try:
        path = snapshot_download(
            info["repo"],
            cache_dir=cache_dir,
        )
        elapsed = time.time() - t0
        log.info(f"  ✅  Downloaded to: {path}")
        log.info(f"  ⏱  Time: {elapsed:.1f}s")
        return {"path": path, "status": "downloaded", "reason": None}
    except Exception as e:
        elapsed = time.time() - t0
        if info.get("skip_if_unauthenticated") and _is_access_denied_error(e):
            log.warning(
                f"  ⚠  Skipping optional gated repo after {elapsed:.1f}s: {e}"
            )
            return {
                "path": None,
                "status": "skipped",
                "reason": "optional gated repo unavailable",
            }
        log.error(f"  ❌  Failed after {elapsed:.1f}s: {e}")
        return {"path": None, "status": "failed", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-TTS models from HuggingFace"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to download (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    models_to_download = list(MODELS.keys()) if "all" in args.models else args.models

    log.info(f"Downloading {len(models_to_download)} model(s)...")
    total_t0 = time.time()

    results = {}
    for name in models_to_download:
        info = MODELS[name]
        results[name] = download_model(name, info, cache_dir=args.cache_dir)

    total_elapsed = time.time() - total_t0
    log.info(f"{'=' * 60}")
    log.info(f"Download complete in {total_elapsed:.1f}s")
    log.info("")

    for name, result in results.items():
        status = {"downloaded": "✅", "skipped": "⏭️", "failed": "❌"}[result["status"]]
        log.info(f"  {status}  {MODELS[name]['repo']}")
        if result["path"]:
            log.info(f"       → {result['path']}")

    failed = [n for n, r in results.items() if r["status"] == "failed"]
    if failed:
        log.warning(
            f"\n⚠  {len(failed)} model(s) failed to download: {', '.join(failed)}"
        )
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
