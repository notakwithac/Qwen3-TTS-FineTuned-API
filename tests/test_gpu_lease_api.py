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


sse_module = types.ModuleType("sse_starlette.sse")
sse_module.EventSourceResponse = object
sys.modules.setdefault("sse_starlette", types.ModuleType("sse_starlette"))
sys.modules["sse_starlette.sse"] = sse_module

broadcast_module = types.ModuleType("broadcaster")
broadcast_module.Broadcast = _DummyBroadcast
sys.modules["broadcaster"] = broadcast_module

import api_server
from gpu_lease_manager import LeaseGrant


class StubLeaseManager:
    def __init__(self):
        self.calls = []
        self.release_results = [True, False]

    def acquire(self, owner, ttl_seconds):
        self.calls.append(("acquire", owner, ttl_seconds))
        return LeaseGrant(
            owner=owner,
            label=f"external:{owner}",
            permit_count=2,
            acquired_at=100.0,
            expires_at=100.0 + ttl_seconds,
            last_heartbeat_at=100.0,
            token="opaque-lease-token",
        )

    def heartbeat(self, token, ttl_seconds=None):
        self.calls.append(("heartbeat", token, ttl_seconds))
        return self.acquire("dataset-lab", ttl_seconds or 30)

    def release(self, token):
        self.calls.append(("release", token))
        return self.release_results.pop(0)

    def status(self):
        return {
            "active": True,
            "pending": False,
            "owner": "dataset-lab",
            "expires_at": 130.0,
        }


def _client_with_key(monkeypatch, manager=None):
    monkeypatch.setenv("GPU_LEASE_API_KEY", "service-secret")
    if manager is not None:
        monkeypatch.setattr(api_server, "gpu_lease_manager", manager)
    return TestClient(api_server.app)


def test_lease_endpoints_are_disabled_without_service_key(monkeypatch):
    monkeypatch.delenv("GPU_LEASE_API_KEY", raising=False)
    response = TestClient(api_server.app).get("/gpu/leases/status")
    assert response.status_code == 503


def test_lease_endpoints_reject_wrong_service_key(monkeypatch):
    monkeypatch.setenv("GPU_LEASE_API_KEY", "service-secret")
    response = TestClient(api_server.app).get(
        "/gpu/leases/status", headers={"X-GPU-Lease-Key": "wrong"}
    )
    assert response.status_code == 401


def test_acquire_runs_manager_off_event_loop_and_returns_token(monkeypatch):
    manager = StubLeaseManager()
    thread_calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        thread_calls.append(fn.__name__)
        return fn(*args, **kwargs)

    monkeypatch.setattr(api_server.asyncio, "to_thread", fake_to_thread)
    client = _client_with_key(monkeypatch, manager)

    response = client.post(
        "/gpu/leases",
        headers={"X-GPU-Lease-Key": "service-secret"},
        json={"owner": "dataset-lab", "ttl_seconds": 60},
    )

    assert response.status_code == 200
    assert response.json()["token"] == "opaque-lease-token"
    assert manager.calls == [("acquire", "dataset-lab", 60)]
    assert thread_calls == ["acquire"]


def test_heartbeat_requires_service_key_and_uses_path_token(monkeypatch):
    manager = StubLeaseManager()
    client = _client_with_key(monkeypatch, manager)

    unauthorized = client.post(
        "/gpu/leases/opaque-lease-token/heartbeat", json={"ttl_seconds": 90}
    )
    response = client.post(
        "/gpu/leases/opaque-lease-token/heartbeat",
        headers={"X-GPU-Lease-Key": "service-secret"},
        json={"ttl_seconds": 90},
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert ("heartbeat", "opaque-lease-token", 90) in manager.calls


def test_status_never_returns_lease_token(monkeypatch):
    manager = StubLeaseManager()
    client = _client_with_key(monkeypatch, manager)

    response = client.get(
        "/gpu/leases/status", headers={"X-GPU-Lease-Key": "service-secret"}
    )

    assert response.status_code == 200
    assert "token" not in response.json()
    assert "opaque-lease-token" not in response.text


def test_release_endpoint_is_idempotent(monkeypatch):
    manager = StubLeaseManager()
    client = _client_with_key(monkeypatch, manager)
    headers = {"X-GPU-Lease-Key": "service-secret"}

    first = client.delete("/gpu/leases/opaque-lease-token", headers=headers)
    second = client.delete("/gpu/leases/opaque-lease-token", headers=headers)

    assert first.json() == {"released": True}
    assert second.json() == {"released": False}


def test_lease_preparation_stops_vllm_before_unloading_qwen(monkeypatch):
    events = []
    runtime = types.SimpleNamespace(stop_all=lambda: events.append("stop-vllm"))
    inference = types.SimpleNamespace(unload=lambda: events.append("unload-qwen"))
    monkeypatch.setattr(api_server, "vllm_runtime", runtime)
    monkeypatch.setattr(
        api_server, "pipeline", types.SimpleNamespace(inference=inference)
    )

    api_server._prepare_gpu_for_external_lease()

    assert events == ["stop-vllm", "unload-qwen"]
