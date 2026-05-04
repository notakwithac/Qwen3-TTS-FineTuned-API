# Qwen3-TTS Pronunciation SFT and LoRA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve recurring pronunciation problems by training on correctly pronounced examples, using the current SFT pipeline first and adding a clean LoRA extension point only if PEFT support is introduced.

**Architecture:** Treat pronunciation as a supervised adaptation problem. Build a pronunciation-focused dataset with varied contexts and verified reference audio, train from the base or selected finetuned checkpoint, then compare checkpoints with a fixed pronunciation eval set before promoting one.

**Tech Stack:** Python, Qwen3-TTS 12Hz finetuning, Accelerate, PyTorch, safetensors, pytest

---

### Task 1: Add pronunciation dataset guidance and validation

**Files:**
- Create: `docs/pronunciation_finetuning.md`
- Modify: `finetuning/prepare_data.py`
- Test: `tests/test_pronunciation_dataset_validation.py`

**Step 1: Write failing validation tests**

Add tests for pronunciation dataset metadata:
- each target term has at least 5 context sentences
- each item has `text`, `audio`, `ref_audio`, `language`, and `target_terms`
- terms appear in the text unless the item is marked as a preservation sample
- preservation samples are allowed and counted separately

Run:

```bash
python -m pytest tests/test_pronunciation_dataset_validation.py -q
```

Expected: FAIL because validation does not exist.

**Step 2: Implement validation helper**

Add a helper that can run during dataset preparation and as a standalone check:

```bash
python finetuning/prepare_data.py --validate-pronunciation-manifest path/to/train.jsonl
```

Keep validation warning-oriented by default; use `--strict-pronunciation-validation` to fail.

**Step 3: Document dataset requirements**

In `docs/pronunciation_finetuning.md`, specify:
- 5-10 contexts per target word
- at least 20-50 preservation samples for unrelated common text
- native-speaker or trusted reference pronunciations for target terms
- stable transcript/audio alignment before training

### Task 2: Add a pronunciation adaptation training profile

**Files:**
- Modify: `finetuning/sft_12hz.py`
- Modify: `finetuning/README.md`
- Test: `tests/test_pronunciation_training_args.py`

**Step 1: Write failing CLI tests**

Test that the training script accepts:
- `--adaptation_type pronunciation`
- `--preservation_loss_weight`
- `--target_term_metadata`

Do not change default behavior when these args are absent.

**Step 2: Implement SFT profile**

For `--adaptation_type pronunciation`:
- default to a conservative learning rate if not explicitly set
- log target term coverage at startup
- keep current training loss path unchanged
- include preservation metadata in `trainer_state.json`

Do not implement LoRA in this task. The repo does not currently depend on PEFT, and current `TRAINABLE_SCOPE_CHOICES` only supports full or selected Qwen3-TTS weights.

**Step 3: Document commands**

Add examples:

```bash
python finetuning/sft_12hz.py --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --output_model_path runs/pronunciation_sft --train_jsonl data/pronunciation/train.jsonl --speaker_name pronunciation_speaker --adaptation_type pronunciation --num_epochs 3 --lr 1e-5
```

### Task 3: Add checkpoint comparison workflow

**Files:**
- Create: `tools/compare_pronunciation_checkpoints.py`
- Test: `tests/test_compare_pronunciation_checkpoints.py`
- Modify: `API_DOCS.md`

**Step 1: Implement dry-run comparison**

The script must accept:
- `--eval-jsonl`
- `--model-path` multiple times
- `--output-dir`
- `--dry-run`

Dry-run writes the planned model/eval matrix without loading model weights.

**Step 2: Implement generation mode**

For each model path:
- generate one WAV per eval sentence
- write `results.jsonl`
- record generation config, checkpoint path, language, and target terms

Scoring can be manual in v1, but the output schema must reserve fields for ASR text, target correctness, PER, WER, and reviewer notes.

**Step 3: Add docs**

Document comparing:
- base model
- current production finetune
- pronunciation-adapted finetune

### Task 4: Optional PEFT/LoRA follow-up gate

**Files:**
- Modify: `pyproject.toml`
- Modify: `finetuning/sft_12hz.py`
- Test: `tests/test_pronunciation_lora_args.py`

**Step 1: Only implement if PEFT is accepted as a dependency**

Do not add this in the first implementation unless the team explicitly wants adapter weights instead of full checkpoints.

**Step 2: If enabled, target Qwen talker projections**

LoRA targets should be limited to:
- `talker.model.layers.*.self_attn.q_proj`
- `talker.model.layers.*.self_attn.k_proj`
- `talker.model.layers.*.self_attn.v_proj`
- `talker.model.layers.*.self_attn.o_proj`
- optionally `talker.model.layers.*.mlp.*_proj`

Do not attach LoRA to the speech tokenizer or speaker encoder.

### Task 5: Verify

**Files:**
- Test: `tests/test_pronunciation_dataset_validation.py`
- Test: `tests/test_pronunciation_training_args.py`
- Test: `tests/test_compare_pronunciation_checkpoints.py`

**Step 1: Run focused tests**

```bash
python -m pytest tests/test_pronunciation_dataset_validation.py tests/test_pronunciation_training_args.py tests/test_compare_pronunciation_checkpoints.py -q
```

Expected: PASS.

**Step 2: Compile scripts**

```bash
python -m py_compile finetuning/sft_12hz.py finetuning/prepare_data.py tools/compare_pronunciation_checkpoints.py
```

Expected: no output.

### How to use this plan

**Base model**

Start from `Qwen/Qwen3-TTS-12Hz-1.7B-Base` when the pronunciation issue is general, such as names, places, brands, or a language/domain pattern. This creates a clean pronunciation-adapted checkpoint that can be evaluated directly against the original base model.

**Existing finetuned models**

Continue training from the existing finetuned checkpoint only when the pronunciation issue depends on that voice, domain, or dataset. For general vocabulary fixes, prefer training from base and comparing against the finetuned model before deciding whether to merge the adaptation into production.

**Upcoming finetune models**

Add pronunciation-balanced examples to the training dataset before the first finetune. Include preservation samples in the same manifest so the model learns the target pronunciations without degrading ordinary text.

**Acceptance criteria**

- Current finetuning behavior is unchanged unless `--adaptation_type pronunciation` is passed.
- Pronunciation datasets can be validated before training.
- Checkpoint comparison can run in dry-run mode without model weights.
- LoRA is treated as an explicit follow-up, not silently introduced.
