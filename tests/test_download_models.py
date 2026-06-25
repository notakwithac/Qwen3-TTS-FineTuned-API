import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import download_models


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _AccessDeniedError(Exception):
    def __init__(self, message: str = "Access denied"):
        super().__init__(message)
        self.response = _FakeResponse(401)


def test_download_model_skips_gated_repo_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        raise _AccessDeniedError(
            "Cannot access gated repo for url https://huggingface.co/google/gemma-3-12b-it"
        )

    monkeypatch.setattr(download_models, "snapshot_download", fake_snapshot_download)

    result = download_models.download_model(
        "gemma",
        dict(download_models.MODELS["gemma"]),
        cache_dir=None,
    )

    assert calls
    assert result["status"] == "skipped"
    assert result["path"] is None
    assert result["reason"] == "optional gated repo unavailable"


def test_download_model_skips_gated_repo_when_token_is_present(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    def fake_snapshot_download(*_args, **_kwargs):
        raise _AccessDeniedError(
            "Cannot access gated repo for url https://huggingface.co/google/gemma-3-12b-it"
        )

    monkeypatch.setattr(download_models, "snapshot_download", fake_snapshot_download)

    result = download_models.download_model(
        "gemma",
        dict(download_models.MODELS["gemma"]),
        cache_dir=None,
    )

    assert result["status"] == "skipped"
    assert result["path"] is None
    assert result["reason"] == "optional gated repo unavailable"
