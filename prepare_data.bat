set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run finetuning/prepare_data.py ^
--device cuda:0 ^
--tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz ^
--input_jsonl gared_voice_qwen3_tts_dataset/train.jsonl ^
--output_jsonl gared_voice_qwen3_tts_dataset/train_with_codes.jsonl