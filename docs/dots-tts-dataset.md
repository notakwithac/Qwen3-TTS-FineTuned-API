# dots.tts-mf dataset generation

`dots.tts-mf` is installed locally as a reference-conditioned dataset generator. It is useful for expanding one authorized reference clip into scripted dialogue or other training utterances. It is not a replacement for Qwen voice design: it cannot create a new speaker from a prose voice description.

## Local layout

- `dots.tts-mf/` contains the checkpoint pinned at Hugging Face revision `25c53fb462e57087e52237daa5ea30df1c5cc328`.
- `dots.tts/` contains the runtime source pinned at Git commit `b995bdb7de14dad008b98cfa085ecc83600e5b7a`.
- Both directories are ignored by the parent repository because the checkpoint is about 5.17 GB and the runtime is an independent Git repository.

There is no upstream vLLM implementation for this model. The supported runtime is `DotsTtsRuntime`; run it as an offline, GPU-exclusive dataset job. Stop the Qwen API and managed Gemma/Sarvam vLLM processes first so the models do not compete for VRAM.

## Isolated environment

Do not install dots.tts into the Qwen API environment. Qwen currently pins `numpy<2`, while dots.tts requires `numpy>=2.2.6`.

On Linux:

```bash
python3.11 -m venv .venv-dots-tts
source .venv-dots-tts/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./dots.tts -c ./dots.tts/constraints/recommended.txt
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv-dots-tts
.\.venv-dots-tts\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\dots.tts -c .\dots.tts\constraints\recommended.txt
```

## Generate dialogue clips

Create a JSONL manifest with one utterance per row. `filename` is optional; `text` must exactly match the desired audio.

```json
{"filename":"line_0001.wav","text":"I thought you said the road would be empty tonight."}
```

Then run:

```bash
python scripts/generate_dots_dataset.py \
  --manifest examples/dots_dataset_manifest.jsonl \
  --prompt-audio /path/to/authorized_reference.wav \
  --prompt-text "The exact transcript spoken in the reference audio." \
  --output-dir datasets/my_voice
```

The generator loads the model once, uses the upstream-recommended four MeanFlow steps, and writes:

```text
datasets/my_voice/
  train_raw.jsonl
  data/
    ref_audio.wav
    line_0001.wav
    line_0002.wav
```

`train_raw.jsonl` already matches `finetuning/prepare_data.py`. Continue with:

```bash
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl datasets/my_voice/train_raw.jsonl \
  --output_jsonl datasets/my_voice/train_with_codes.jsonl
```

Review generated clips before training. Only use reference voices with documented consent, keep the reference transcript exact, and remove mispronounced or speaker-inconsistent samples.

Run the generator separately for each character/reference voice. A single output dataset is intentionally single-speaker, matching Qwen3-TTS's current fine-tuning contract.
