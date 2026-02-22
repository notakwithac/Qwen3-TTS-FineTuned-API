#!/bin/bash
# Linux equivalent of train17.bat — main fine-tuning entry point
set -e
source .venv/bin/activate 2>/dev/null || true

uv run finetuning/sft_12hz.py \
  --lr_scheduler cosine \
  --warmup_ratio 0.05 \
  --save_interval 10 \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path runs/run1 \
  --train_jsonl gared_voice_qwen3_tts_dataset/train_with_codes.jsonl \
  --batch_size 1 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_gared
