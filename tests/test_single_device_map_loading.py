from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_tts.inference import qwen3_tts_model as qwen3_tts_model_module
from qwen_tts.inference import qwen3_tts_tokenizer as qwen3_tts_tokenizer_module


class _FakeLoadedModel:
    def __init__(self):
        self.config = object()
        self.generate_config = {"temperature": 0.7}
        self.to_calls = []
        self.device = "cpu"

    def to(self, device):
        self.to_calls.append(device)
        self.device = device
        return self


def _stub_hf_registration(monkeypatch, module):
    monkeypatch.setattr(module.AutoConfig, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.AutoModel, "register", lambda *args, **kwargs: None)
    if hasattr(module, "AutoProcessor"):
        monkeypatch.setattr(module.AutoProcessor, "register", lambda *args, **kwargs: None)


def test_qwen3_tts_model_loads_single_device_map_without_dispatch(monkeypatch):
    _stub_hf_registration(monkeypatch, qwen3_tts_model_module)
    captured = {}
    fake_model = _FakeLoadedModel()

    class _ExpectedModel(_FakeLoadedModel):
        pass

    fake_model.__class__ = _ExpectedModel

    monkeypatch.setattr(qwen3_tts_model_module, "Qwen3TTSForConditionalGeneration", _ExpectedModel)

    def _load(path, **kwargs):
        captured["call"] = (path, kwargs)
        return fake_model

    monkeypatch.setattr(qwen3_tts_model_module.AutoModel, "from_pretrained", _load)
    monkeypatch.setattr(
        qwen3_tts_model_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    wrapper = qwen3_tts_model_module.Qwen3TTSModel.from_pretrained(
        "checkpoint",
        device_map="cuda:0",
        dtype="bf16",
    )

    _, forwarded_kwargs = captured["call"]
    assert forwarded_kwargs["device_map"] == {"": "cuda:0"}
    assert forwarded_kwargs["dtype"] == "bf16"
    assert fake_model.to_calls == []
    assert wrapper.model is fake_model


def test_qwen3_tts_model_preserves_real_device_map(monkeypatch):
    _stub_hf_registration(monkeypatch, qwen3_tts_model_module)
    captured = {}
    fake_model = _FakeLoadedModel()

    class _ExpectedModel(_FakeLoadedModel):
        pass

    fake_model.__class__ = _ExpectedModel

    monkeypatch.setattr(qwen3_tts_model_module, "Qwen3TTSForConditionalGeneration", _ExpectedModel)

    def _load(path, **kwargs):
        captured["call"] = (path, kwargs)
        return fake_model

    monkeypatch.setattr(qwen3_tts_model_module.AutoModel, "from_pretrained", _load)
    monkeypatch.setattr(
        qwen3_tts_model_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    qwen3_tts_model_module.Qwen3TTSModel.from_pretrained(
        "checkpoint",
        device_map={"": "cuda:0"},
    )

    _, forwarded_kwargs = captured["call"]
    assert forwarded_kwargs["device_map"] == {"": "cuda:0"}
    assert fake_model.to_calls == []


def test_qwen3_tts_tokenizer_loads_single_device_map_without_dispatch(monkeypatch):
    _stub_hf_registration(monkeypatch, qwen3_tts_tokenizer_module)
    captured = {}
    fake_model = _FakeLoadedModel()

    def _load(path, **kwargs):
        captured["call"] = (path, kwargs)
        return fake_model

    monkeypatch.setattr(qwen3_tts_tokenizer_module.AutoModel, "from_pretrained", _load)
    monkeypatch.setattr(
        qwen3_tts_tokenizer_module.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    tokenizer = qwen3_tts_tokenizer_module.Qwen3TTSTokenizer.from_pretrained(
        "tokenizer",
        device_map="cuda:0",
        dtype="bf16",
    )

    _, forwarded_kwargs = captured["call"]
    assert forwarded_kwargs["device_map"] == {"": "cuda:0"}
    assert forwarded_kwargs["dtype"] == "bf16"
    assert fake_model.to_calls == []
    assert tokenizer.model is fake_model


def test_skip_dispatch_for_single_device_restores_original_dispatch(monkeypatch):
    original_dispatch = qwen3_tts_model_module.hf_modeling_utils.dispatch_model
    seen = []

    def _replacement(model, **kwargs):
        seen.append((model, kwargs))
        return "replacement"

    monkeypatch.setattr(qwen3_tts_model_module.hf_modeling_utils, "dispatch_model", _replacement)

    with qwen3_tts_model_module._skip_dispatch_for_single_device({"": "cuda:0"}):
        assert qwen3_tts_model_module.hf_modeling_utils.dispatch_model("model") == "model"

    assert qwen3_tts_model_module.hf_modeling_utils.dispatch_model is _replacement
    assert seen == []
