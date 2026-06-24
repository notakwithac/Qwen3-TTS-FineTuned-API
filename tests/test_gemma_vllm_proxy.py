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
    monkeypatch.setattr(api_server, "GEMMA_VLLM_MODEL", "gemma12b")
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
    assert captured["json"]["model"] == "gemma12b"
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


def test_models_lists_configured_gemma_model(monkeypatch):
    monkeypatch.setattr(api_server, "GEMMA_VLLM_MODEL", "gemma12b")

    client = TestClient(api_server.app)
    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "gemma12b"
