import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _DummyBroadcast:
    def __init__(self, *_args, **_kwargs):
        pass

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def publish(self, *_args, **_kwargs):
        return None


sse_module = types.ModuleType("sse_starlette.sse")
sse_module.EventSourceResponse = object
sys.modules.setdefault("sse_starlette", types.ModuleType("sse_starlette"))
sys.modules["sse_starlette.sse"] = sse_module

broadcast_module = types.ModuleType("broadcaster")
broadcast_module.Broadcast = _DummyBroadcast
sys.modules["broadcaster"] = broadcast_module

import api_server


class _StubInference:
    def __init__(self):
        self.calls = []
        self.unloads = 0

    @property
    def stats(self):
        return {
            "inference_limiter": {
                "capacity": 1,
                "active": 0,
                "available": 1,
                "holders": {},
                "waiting": {},
            }
        }

    def run_external_gpu_call(self, label, fn):
        self.calls.append(label)
        return fn()

    def unload(self):
        self.unloads += 1
        return {}


class _StubPipeline:
    def __init__(self):
        self.inference = _StubInference()


class _FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    content = b'{"choices":[{"message":{"content":"ok"}}]}'

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}

    def raise_for_status(self):
        return None


class _StubVllmRuntime:
    def __init__(self):
        self.ensure_calls = []
        self.mark_calls = []
        self.stop_all_calls = []
        self.log_path = "/tmp/pathnam-vllm-sarvam.log"

    def base_url(self, name):
        return {
            "gemma": "http://127.0.0.1:8101/v1",
            "sarvam": "http://127.0.0.1:8102/v1",
        }[name]

    def ensure_running(self, name):
        self.ensure_calls.append(name)

    def mark_used(self, name):
        self.mark_calls.append(name)

    def stop_all(self):
        self.stop_all_calls.append(True)

    def status(self):
        return {}

    def get(self, name):
        return types.SimpleNamespace(log_path=self.log_path)

    def tail_log(self, name, *, max_chars=2000):
        return "sarvam log tail"


def test_chat_completions_proxy_uses_gemma_model_and_gpu_limiter(monkeypatch):
    pipeline = _StubPipeline()
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _FakeResponse()

    monkeypatch.setattr(api_server, "pipeline", pipeline)
    runtime = _StubVllmRuntime()
    monkeypatch.setattr(api_server, "vllm_runtime", runtime)
    monkeypatch.setattr(api_server, "MANAGED_VLLM_ENABLED", True)
    monkeypatch.setattr(api_server, "GEMMA_VLLM_MODEL", "e4b")
    monkeypatch.setattr(api_server, "GEMMA_VLLM_FORCE_MODEL", True)
    monkeypatch.setattr(api_server.requests, "post", fake_post)

    client = TestClient(api_server.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gemma4:e4b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert pipeline.inference.calls == ["inference_gemma_vllm"]
    assert pipeline.inference.unloads == 1
    assert runtime.ensure_calls == ["gemma"]
    assert runtime.mark_calls == ["gemma"]
    assert captured["url"] == "http://127.0.0.1:8101/v1/chat/completions"
    assert captured["json"]["model"] == "e4b"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_translate_proxy_uses_sarvam_and_same_gpu_limiter(monkeypatch):
    pipeline = _StubPipeline()
    runtime = _StubVllmRuntime()
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr(api_server, "pipeline", pipeline)
    monkeypatch.setattr(api_server, "vllm_runtime", runtime)
    monkeypatch.setattr(api_server, "MANAGED_VLLM_ENABLED", True)
    monkeypatch.setattr(api_server, "SARVAM_VLLM_MODEL", "sarvam-translate")
    monkeypatch.setattr(api_server.requests, "post", fake_post)

    client = TestClient(api_server.app)
    response = client.post(
        "/translate",
        json={
            "text": "hello",
            "source_language": "English",
            "target_language": "Hindi",
        },
    )

    assert response.status_code == 200
    assert response.json()["text"] == "ok"
    assert pipeline.inference.calls == ["inference_sarvam_vllm"]
    assert pipeline.inference.unloads == 1
    assert runtime.ensure_calls == ["sarvam"]
    assert runtime.mark_calls == ["sarvam"]
    assert captured["url"] == "http://127.0.0.1:8102/v1/chat/completions"
    assert captured["json"]["model"] == "sarvam-translate"


def test_translate_proxy_returns_503_when_managed_sarvam_fails(monkeypatch):
    pipeline = _StubPipeline()
    runtime = _StubVllmRuntime()

    def fail_ensure_running(name):
        raise TimeoutError("vLLM service sarvam did not become ready: boot log")

    runtime.ensure_running = fail_ensure_running
    monkeypatch.setattr(api_server, "pipeline", pipeline)
    monkeypatch.setattr(api_server, "vllm_runtime", runtime)
    monkeypatch.setattr(api_server, "MANAGED_VLLM_ENABLED", True)

    client = TestClient(api_server.app)
    response = client.post(
        "/translate",
        json={
            "text": "hello",
            "source_language": "English",
            "target_language": "Hindi",
        },
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "sarvam_unavailable"
    assert "Sarvam vLLM backend unavailable" in detail["message"]
    assert "boot log" in detail["message"]
    assert detail["log_path"] == "/tmp/pathnam-vllm-sarvam.log"
    assert detail["log_tail"] == "sarvam log tail"
    assert pipeline.inference.calls == ["inference_sarvam_vllm"]


def test_translate_proxy_returns_structured_503_when_sarvam_read_times_out(monkeypatch):
    pipeline = _StubPipeline()
    runtime = _StubVllmRuntime()

    def fake_post(url, json, headers, timeout):
        raise api_server.requests.ReadTimeout("read timeout")

    monkeypatch.setattr(api_server, "pipeline", pipeline)
    monkeypatch.setattr(api_server, "vllm_runtime", runtime)
    monkeypatch.setattr(api_server, "MANAGED_VLLM_ENABLED", True)
    monkeypatch.setattr(api_server, "SARVAM_VLLM_TIMEOUT_SECONDS", 600.0)
    monkeypatch.setattr(api_server.requests, "post", fake_post)

    client = TestClient(api_server.app)
    response = client.post(
        "/translate",
        json={
            "text": "hello",
            "source_language": "English",
            "target_language": "Hindi",
        },
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "sarvam_timeout"
    assert detail["timeout_seconds"] == 600.0
    assert detail["log_path"] == "/tmp/pathnam-vllm-sarvam.log"
    assert detail["log_tail"] == "sarvam log tail"
    assert pipeline.inference.calls == ["inference_sarvam_vllm"]
    assert runtime.ensure_calls == ["sarvam"]
    assert runtime.mark_calls == []


def test_sarvam_vllm_uses_eager_mode_by_default():
    assert "--enforce-eager" in api_server.vllm_runtime.get("sarvam").extra_args


def test_sarvam_vllm_disables_image_prompt_profiling_by_default():
    extra_args = api_server.vllm_runtime.get("sarvam").extra_args

    limit_index = extra_args.index("--limit-mm-per-prompt")
    assert extra_args[limit_index + 1] == '{"image":0}'


def test_sarvam_vllm_limits_batch_profile_by_default():
    extra_args = api_server.vllm_runtime.get("sarvam").extra_args

    batch_index = extra_args.index("--max-num-batched-tokens")
    assert extra_args[batch_index + 1] == "1024"


def test_sarvam_vllm_uses_bitsandbytes_by_default():
    extra_args = api_server.vllm_runtime.get("sarvam").extra_args

    load_index = extra_args.index("--load-format")
    quantization_index = extra_args.index("--quantization")
    assert extra_args[load_index + 1] == "bitsandbytes"
    assert extra_args[quantization_index + 1] == "bitsandbytes"
    assert "--cpu-offload-gb" not in extra_args


def test_models_lists_configured_gemma_model(monkeypatch):
    monkeypatch.setattr(api_server, "GEMMA_VLLM_MODEL", "e4b")

    client = TestClient(api_server.app)
    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "e4b"
