from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_tts.inference.qwen3_tts_model import InferenceOptimizationConfig, Qwen3TTSModel


class _FakeProcessor:
    def __init__(self, padding_side="right"):
        self.calls = []
        self.padding_side = padding_side

    def __call__(self, text, return_tensors="pt", padding=True):
        texts = text if isinstance(text, list) else [text]
        self.calls.append(list(texts))
        max_len = max(len(t) for t in texts)
        input_ids = []
        attention_mask = []
        for item in texts:
            row = [ord(ch) % 17 + 1 for ch in item]
            pad = max_len - len(row)
            if self.padding_side == "left":
                input_ids.append(([0] * pad) + row)
                attention_mask.append(([0] * pad) + ([1] * len(row)))
            else:
                input_ids.append(row + ([0] * pad))
                attention_mask.append(([1] * len(row)) + ([0] * pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
        }


class _FakeSpeechTokenizer:
    def decode(self, payload):
        wavs = [np.zeros((entry["audio_codes"].shape[0],), dtype=np.float32) for entry in payload]
        return wavs, 24000


class _FakeModel:
    def __init__(self, tts_model_type="custom_voice", tts_model_size="1b7"):
        self.tts_model_type = tts_model_type
        self.tts_model_size = tts_model_size
        self.tokenizer_type = "fake"
        self.generate_config = {}
        self.speech_tokenizer = _FakeSpeechTokenizer()
        self.device = torch.device("cpu")
        self.generate_calls = []

    def parameters(self):
        yield torch.zeros(1)

    def get_supported_languages(self):
        return ["English", "Auto"]

    def get_supported_speakers(self):
        return ["narrator"]

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        batch_size = len(kwargs["input_ids"])
        return [torch.ones((6, 2), dtype=torch.int64) for _ in range(batch_size)], None


def test_tokenize_texts_batches_unique_inputs():
    processor = _FakeProcessor()
    wrapper = Qwen3TTSModel(
        model=_FakeModel(),
        processor=processor,
        optimization_config=InferenceOptimizationConfig(enable_batched_tokenization=True),
    )

    tokenized = wrapper._tokenize_texts(["hello", "world", "hello"])

    assert len(tokenized) == 3
    assert processor.calls == [["hello", "world"]]
    assert torch.equal(tokenized[0], tokenized[2])


def test_tokenize_texts_removes_left_padding():
    processor = _FakeProcessor(padding_side="left")
    wrapper = Qwen3TTSModel(
        model=_FakeModel(),
        processor=processor,
        optimization_config=InferenceOptimizationConfig(enable_batched_tokenization=True),
    )

    tokenized = wrapper._tokenize_texts(["short", "much longer", "short"])

    expected_short = torch.tensor(
        [[ord(ch) % 17 + 1 for ch in "short"]],
        dtype=torch.int64,
    )
    assert torch.equal(tokenized[0].cpu(), expected_short)
    assert torch.equal(tokenized[0], tokenized[2])


def test_generate_custom_voice_reuses_batched_instruction_tokenization():
    processor = _FakeProcessor()
    wrapper = Qwen3TTSModel(
        model=_FakeModel(),
        processor=processor,
        optimization_config=InferenceOptimizationConfig(enable_batched_tokenization=True),
    )

    wavs, sr = wrapper.generate_custom_voice(
        text=["line a", "line b"],
        speaker="narrator",
        language="English",
        instruct=["calm", "calm"],
    )

    assert len(wavs) == 2
    assert sr == 24000
    assert processor.calls == [
        ["<|im_start|>assistant\nline a<|im_end|>\n<|im_start|>assistant\n",
         "<|im_start|>assistant\nline b<|im_end|>\n<|im_start|>assistant\n"],
        ["<|im_start|>user\ncalm<|im_end|>\n"],
    ]


def test_compile_failure_falls_back_to_eager(monkeypatch):
    processor = _FakeProcessor()
    monkeypatch.setattr(
        "qwen_tts.inference.qwen3_tts_model.torch.compile",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )
    wrapper = Qwen3TTSModel(
        model=_FakeModel(),
        processor=processor,
        optimization_config=InferenceOptimizationConfig(
            compile_enabled=True,
            enable_batched_tokenization=False,
        ),
    )

    wavs, sr = wrapper.generate_custom_voice(
        text="hello",
        speaker="narrator",
        language="English",
    )

    assert len(wavs) == 1
    assert sr == 24000
    assert wrapper._compile_failure == "compile failed"
    assert len(wrapper.model.generate_calls) == 1


def test_warmup_for_inference_custom_voice_uses_existing_api():
    wrapper = Qwen3TTSModel(
        model=_FakeModel(),
        processor=_FakeProcessor(),
        optimization_config=InferenceOptimizationConfig(enable_batched_tokenization=False),
    )

    called = {}

    def _generate_custom_voice(**kwargs):
        called.update(kwargs)
        return [np.zeros((1,), dtype=np.float32)], 24000

    wrapper.generate_custom_voice = _generate_custom_voice

    wrapper.warmup_for_inference(
        mode="custom_voice",
        text="warm",
        speaker="narrator",
        language="English",
        max_new_tokens=8,
    )

    assert called["text"] == "warm"
    assert called["speaker"] == "narrator"
    assert called["max_new_tokens"] == 8
