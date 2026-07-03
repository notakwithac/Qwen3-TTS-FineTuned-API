import subprocess

import pytest

from vllm_runtime_manager import ManagedVllmService, VllmRuntimeManager


class _ExitedProcess:
    pid = 123
    returncode = 42

    def poll(self):
        return self.returncode


def test_startup_exit_error_includes_recent_vllm_log(monkeypatch, tmp_path):
    log_path = tmp_path / "sarvam.log"

    def fake_popen(*_args, **kwargs):
        kwargs["stdout"].write("CUDA out of memory while loading weights\n")
        kwargs["stdout"].flush()
        return _ExitedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    service = ManagedVllmService(
        name="sarvam",
        model="sarvamai/sarvam-translate",
        served_model_name="sarvam-translate",
        port=8102,
        startup_timeout_seconds=1,
        log_path=str(log_path),
    )
    manager = VllmRuntimeManager({"sarvam": service})

    with pytest.raises(RuntimeError) as exc_info:
        manager.ensure_running("sarvam")

    message = str(exc_info.value)
    assert "exited during startup with code 42" in message
    assert "CUDA out of memory while loading weights" in message


def test_startup_disables_implicit_hf_token_without_explicit_token(monkeypatch, tmp_path):
    captured = {}

    class _RunningProcess:
        pid = 456

        def poll(self):
            return None

    def fake_popen(*_args, **kwargs):
        captured["env"] = kwargs["env"]
        return _RunningProcess()

    class _ReadyResponse:
        ok = True

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("vllm_runtime_manager.requests.get", lambda *_args, **_kwargs: _ReadyResponse())

    service = ManagedVllmService(
        name="sarvam",
        model="sarvamai/sarvam-translate",
        served_model_name="sarvam-translate",
        port=8102,
        startup_timeout_seconds=1,
        log_path=str(tmp_path / "sarvam.log"),
    )
    manager = VllmRuntimeManager({"sarvam": service})

    manager.ensure_running("sarvam")

    assert captured["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_startup_strips_hf_token_when_service_does_not_use_token(monkeypatch, tmp_path):
    captured = {}

    class _RunningProcess:
        pid = 789

        def poll(self):
            return None

    def fake_popen(*_args, **kwargs):
        captured["env"] = kwargs["env"]
        return _RunningProcess()

    class _ReadyResponse:
        ok = True

    monkeypatch.setenv("HF_TOKEN", "expired-token")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("vllm_runtime_manager.requests.get", lambda *_args, **_kwargs: _ReadyResponse())

    service = ManagedVllmService(
        name="sarvam",
        model="sarvamai/sarvam-translate",
        served_model_name="sarvam-translate",
        port=8102,
        startup_timeout_seconds=1,
        log_path=str(tmp_path / "sarvam.log"),
        use_hf_token=False,
    )
    manager = VllmRuntimeManager({"sarvam": service})

    manager.ensure_running("sarvam")

    assert "HF_TOKEN" not in captured["env"]
    assert captured["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
