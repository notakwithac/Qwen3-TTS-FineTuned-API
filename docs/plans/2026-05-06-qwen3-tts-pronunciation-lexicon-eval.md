# Qwen3-TTS Pronunciation Lexicon and Eval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a low-risk pronunciation correction layer before Qwen3-TTS tokenization and an evaluation harness that measures whether target words are spoken correctly.

**Architecture:** Keep model weights unchanged. Add an opt-in text preprocessing layer that rewrites known problematic words or phrases before `Qwen3TTSModel` builds assistant text, then add a pronunciation eval runner that generates samples and records ASR/PER-style review data for target terms.

**Tech Stack:** Python, JSON/YAML-style config, FastAPI plumbing, Qwen3-TTS inference wrapper, pytest

---

### Task 1: Define the pronunciation lexicon format

**Files:**
- Create: `qwen_tts/pronunciation.py`
- Create: `tests/test_pronunciation_lexicon.py`
- Modify: `.env.example`

**Step 1: Write failing parser tests**

Create tests for:
- exact word replacement
- phrase replacement
- case-insensitive matching that preserves surrounding punctuation
- disabled lexicon returning the original text
- per-model overlay entries overriding global entries

Run:

```bash
python -m pytest tests/test_pronunciation_lexicon.py -q
```

Expected: FAIL because the module does not exist.

**Step 2: Implement minimal lexicon loading**

Implement:
- `PronunciationRule` with `pattern`, `replacement`, `match_type`, `case_sensitive`
- `PronunciationLexicon.from_file(path)`
- `PronunciationLexicon.apply(text, model_id=None)`

Use JSON as the first supported format:

```json
{
  "global": [
    {"pattern": "Qwen", "replacement": "Quen", "match_type": "word"}
  ],
  "models": {
    "custom_voice_gared": [
      {"pattern": "Gared", "replacement": "Gah-red", "match_type": "word"}
    ]
  }
}
```

Do not add IPA parsing or SSML yet; Qwen3-TTS currently accepts plain text through a Qwen2 tokenizer, not a phoneme-aware frontend.

**Step 3: Add env docs**

Add env examples:
- `QWEN_TTS_PRONUNCIATION_LEXICON_PATH=`
- `QWEN_TTS_PRONUNCIATION_LEXICON_ENABLED=false`

### Task 2: Wire lexicon into inference without changing default behavior

**Files:**
- Modify: `qwen_tts/inference/qwen3_tts_model.py`
- Modify: `api_server.py`
- Test: `tests/test_qwen3_tts_model_optimizations.py`

**Step 1: Write failing wrapper tests**

Test that `generate_voice_clone`, `generate_custom_voice`, and `generate_voice_design` pass text through the lexicon when enabled and leave text unchanged when disabled.

Use mocks around `_tokenize_texts()` so tests do not load model weights.

**Step 2: Add optional lexicon dependency to `Qwen3TTSModel`**

Add constructor fields:
- `pronunciation_lexicon=None`
- `pronunciation_model_id=None`

Add helper:
- `_apply_pronunciation_lexicon(text: str) -> str`

Call this helper before `_build_assistant_text()` in all generation paths.

**Step 3: Wire API config**

In `api_server.py`, load the lexicon once at startup if enabled and pass it to all model/session construction paths that instantiate `Qwen3TTSModel`.

Use the loaded model id or checkpoint/session id as the `pronunciation_model_id` where available.

### Task 3: Add pronunciation eval runner

**Files:**
- Create: `tools/pronunciation_eval.py`
- Create: `tests/test_pronunciation_eval.py`
- Modify: `API_DOCS.md`

**Step 1: Define eval input format**

Use JSONL:

```json
{"id":"name_001","text":"Gared will join the call.","target":"Gared","expected":"Gah-red","language":"English"}
```

**Step 2: Implement dry-run and generation modes**

The runner must support:
- `--input-jsonl`
- `--output-dir`
- `--model-path`
- `--lexicon-path`
- `--dry-run`

Dry-run writes rewritten text and expected output paths without loading model weights.

Generation mode produces WAVs and a `results.jsonl` manifest. ASR/PER scoring can be added later; for v1, include fields for manual review and optional external scorer output.

**Step 3: Add docs**

Document how to run:

```bash
python tools/pronunciation_eval.py --input-jsonl examples/pronunciation_eval.jsonl --output-dir runs/pronunciation_eval --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --lexicon-path configs/pronunciation_lexicon.json
```

### Task 4: Verify

**Files:**
- Test: `tests/test_pronunciation_lexicon.py`
- Test: `tests/test_pronunciation_eval.py`
- Test: `tests/test_qwen3_tts_model_optimizations.py`

**Step 1: Run focused tests**

```bash
python -m pytest tests/test_pronunciation_lexicon.py tests/test_pronunciation_eval.py tests/test_qwen3_tts_model_optimizations.py -q
```

Expected: PASS.

**Step 2: Compile changed modules**

```bash
python -m py_compile qwen_tts/pronunciation.py tools/pronunciation_eval.py qwen_tts/inference/qwen3_tts_model.py api_server.py
```

Expected: no output.

### How to use this plan

**Base model**

Use the global lexicon with `Qwen/Qwen3-TTS-12Hz-1.7B-Base` or `Qwen/Qwen3-TTS-12Hz-0.6B-Base` for known bad terms. This is the safest production path because it does not alter model weights and can be disabled instantly.

**Existing finetuned models**

Create per-model overlays keyed by the local checkpoint name, model id, or session id. Keep global entries only for terms that should be corrected for every voice; put voice/domain-specific pronunciation rewrites in the model overlay.

**Upcoming finetune models**

Use the same lexicon during dataset preparation and inference. Store a copy of the lexicon alongside the finetune artifact so generated training text and production inference text stay consistent.

**Acceptance criteria**

- Pronunciation rewrites are opt-in and disabled by default.
- Existing API inputs and outputs remain compatible.
- The eval runner can show before/after rewritten text without loading model weights.
- Model-specific lexicon entries override global entries.
