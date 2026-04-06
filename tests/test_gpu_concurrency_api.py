import sys
import types

from fastapi.testclient import TestClient


class _DummyBroadcast:
    def __init__(self, *_args, **_kwargs):
        pass

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def publish(self, *_args, **_kwargs):
        return None

    def subscribe(self, *_args, **_kwargs):
        class _Subscriber:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _Subscriber()


sse_module = types.ModuleType("sse_starlette.sse")
sse_module.EventSourceResponse = object
sys.modules.setdefault("sse_starlette", types.ModuleType("sse_starlette"))
sys.modules["sse_starlette.sse"] = sse_module

broadcast_module = types.ModuleType("broadcaster")
broadcast_module.Broadcast = _DummyBroadcast
sys.modules["broadcaster"] = broadcast_module

import api_server


class StubInference:
    def __init__(self):
        self._max_models = 5
        self._shared_model_replicas = {
            "voice_design": 1,
            "voice_clone": 1,
        }
        self._shared_model_min_headroom_gb = 3.0

    @property
    def stats(self):
        return {
            "max_models": self._max_models,
            "shared_model_replicas": dict(self._shared_model_replicas),
            "shared_model_min_headroom_gb": self._shared_model_min_headroom_gb,
        }

    def update_runtime_config(
        self,
        *,
        max_models=None,
        shared_model_replicas=None,
        shared_model_min_headroom_gb=None,
    ):
        if max_models is not None:
            self._max_models = max_models
        if shared_model_replicas is not None:
            self._shared_model_replicas.update(shared_model_replicas)
        if shared_model_min_headroom_gb is not None:
            self._shared_model_min_headroom_gb = shared_model_min_headroom_gb


class StubPipeline:
    def __init__(self, inference):
        self.inference = inference

    def shutdown(self):
        return None


def test_get_gpu_concurrency_returns_effective_runtime_config(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))

    with TestClient(api_server.app) as client:
        response = client.get("/gpu/concurrency")

    assert response.status_code == 200
    assert response.json() == {
        "gpu_max_models": 5,
        "voice_design_replicas": 1,
        "voice_clone_replicas": 1,
        "shared_model_min_headroom_gb": 3.0,
    }


def test_post_gpu_concurrency_updates_runtime_config(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))

    with TestClient(api_server.app) as client:
        response = client.post(
            "/gpu/concurrency",
            json={
                "gpu_max_models": 7,
                "voice_design_replicas": 2,
                "voice_clone_replicas": 2,
                "shared_model_min_headroom_gb": 4,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "gpu_max_models": 7,
        "voice_design_replicas": 2,
        "voice_clone_replicas": 2,
        "shared_model_min_headroom_gb": 4.0,
    }
    assert inference.stats["max_models"] == 7
    assert inference.stats["shared_model_replicas"] == {
        "voice_design": 2,
        "voice_clone": 2,
    }
    assert inference.stats["shared_model_min_headroom_gb"] == 4.0


def test_post_gpu_concurrency_rejects_invalid_payload(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))

    with TestClient(api_server.app) as client:
        response = client.post(
            "/gpu/concurrency",
            json={
                "gpu_max_models": 0,
                "voice_design_replicas": 1,
                "voice_clone_replicas": 1,
                "shared_model_min_headroom_gb": -1,
            },
        )

    assert response.status_code == 422
