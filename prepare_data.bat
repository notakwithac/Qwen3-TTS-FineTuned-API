uv run finetuning/prepare_data.py ^
--device cuda:0 ^
--tokenizer_model_path models/Qwen3-TTS-Tokenizer-12Hz ^
--input_jsonl qwen3_test_qwen3_tts_dataset/train.jsonl ^
--output_jsonl qwen3_test_qwen3_tts_dataset/train_with_codes.jsonl