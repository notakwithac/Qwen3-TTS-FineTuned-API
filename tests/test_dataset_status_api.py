import asyncio
import io
import json
import sys
import time
import types
import zipfile

from fastapi.testclient import TestClient

import dataset_jobs


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


def _patch_startup_dependencies(monkeypatch):
    monkeypatch.setattr(api_server.session_mgr, "start_cleanup_loop", lambda: None)
    monkeypatch.setattr(api_server.metrics_collector, "start", lambda: None)
    monkeypatch.setattr(api_server.metrics_collector, "stop", lambda: None)
    async def no_preload():
        return None
    monkeypatch.setattr(api_server, "_startup_preload_shared_models", no_preload)


def _configure_storage(monkeypatch):
    monkeypatch.setattr(api_server.storage, "access_key", "key")
    monkeypatch.setattr(api_server.storage, "secret_key", "secret")


def _prepare_payload(job_id: str):
    return {
        "job_id": job_id,
        "book_id": "book-1",
        "chapter_id": "chapter-1",
        "character_id": "char-1",
        "character_name": "Elena",
        "ref_audio_url": "s3://bucket/audio/ref.wav",
        "ref_text": "reference text",
        "items": [
            {
                "filename": "clone_0001.wav",
                "prompt_id": "p1",
                "text": "hello there",
                "s3_url": "s3://bucket/audio/clone_0001.wav",
            }
        ],
    }


def _package_payload(job_id: str):
    return {
        "job_id": job_id,
        "book_id": "book-1",
        "chapter_id": "chapter-1",
        "character_id": "char-1",
        "dataset_items": [
            {
                "id": "clone_0001.wav",
                "s3_url": "s3://bucket/datasets/book-1/items/char-1_clone_0001.wav",
                "text": "hello there",
                "is_reference": True,
                "included": True,
            }
        ],
    }


def test_post_dataset_prepare_creates_job_and_status(monkeypatch):
    _patch_startup_dependencies(monkeypatch)
    _configure_storage(monkeypatch)
    api_server.dataset_jobs.clear()

    async def fake_prepare(job_id, req):
        await asyncio.sleep(0)
        job = api_server.dataset_jobs[job_id]
        job["status"] = "completed"
        job["phase"] = "awaiting_approval"
        job["completed"] = job["total"]
        job["dataset_items"] = [
            {
                "id": "clone_0001.wav",
                "url": "https://signed/item.wav",
                "s3_url": "s3://bucket/datasets/book-1/items/char-1_clone_0001.wav",
                "text": "hello there",
                "is_reference": False,
                "included": True,
            }
        ]

    monkeypatch.setattr(api_server, "_run_dataset_prepare_job", fake_prepare)
    async def no_existing_prepare(*_args, **_kwargs):
        return None

    monkeypatch.setattr(api_server, "load_existing_prepare_result", no_existing_prepare)

    with TestClient(api_server.app) as client:
        response = client.post("/dataset/prepare", json=_prepare_payload("job-prepare-1"))
        assert response.status_code == 200
        assert response.json()["job_id"] == "job-prepare-1"

        for _ in range(20):
            status_response = client.get("/dataset/status/job-prepare-1")
            if status_response.json()["status"] == "completed":
                break
            time.sleep(0.01)

    body = status_response.json()
    assert body["kind"] == "prepare"
    assert body["status"] == "completed"
    assert body["dataset_items"][0]["id"] == "clone_0001.wav"
    assert body["dataset_items"][0]["is_reference"] is False


def test_post_dataset_prepare_reuses_existing_job(monkeypatch):
    _patch_startup_dependencies(monkeypatch)
    _configure_storage(monkeypatch)
    api_server.dataset_jobs.clear()
    api_server.dataset_jobs["job-prepare-existing"] = {
        "job_id": "job-prepare-existing",
        "kind": "prepare",
        "status": "processing",
        "phase": "preparing_dataset_items",
        "total": 1,
        "completed": 0,
        "failed": 0,
        "dataset_items": [],
        "error": None,
        "created_at": time.time(),
    }

    async def fail_prepare(*_args, **_kwargs):
        raise AssertionError("existing dataset job should be reused")

    monkeypatch.setattr(api_server, "_run_dataset_prepare_job", fail_prepare)

    with TestClient(api_server.app) as client:
        response = client.post("/dataset/prepare", json=_prepare_payload("job-prepare-existing"))

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-prepare-existing",
        "finetune_job_id": "job-prepare-existing",
        "status": "processing",
        "kind": "prepare",
        "total": 1,
    }


def test_post_dataset_package_creates_job_and_status(monkeypatch):
    _patch_startup_dependencies(monkeypatch)
    _configure_storage(monkeypatch)
    api_server.dataset_jobs.clear()

    async def fake_package(job_id, req):
        await asyncio.sleep(0)
        job = api_server.dataset_jobs[job_id]
        job["status"] = "completed"
        job["phase"] = "packaged"
        job["completed"] = job["total"]
        job["dataset_s3_key"] = "datasets/book-1/dataset_char-1_job-package-1.zip"
        job["dataset_s3_url"] = "https://bucket/datasets/book-1/dataset_char-1_job-package-1.zip"

    monkeypatch.setattr(api_server, "_run_dataset_package_job", fake_package)
    async def no_existing_package(*_args, **_kwargs):
        return None

    monkeypatch.setattr(api_server, "load_existing_package_result", no_existing_package)

    with TestClient(api_server.app) as client:
        response = client.post("/dataset/package", json=_package_payload("job-package-1"))
        assert response.status_code == 200
        assert response.json()["job_id"] == "job-package-1"

        for _ in range(20):
            status_response = client.get("/dataset/status/job-package-1")
            if status_response.json()["status"] == "completed":
                break
            time.sleep(0.01)

    body = status_response.json()
    assert body["kind"] == "package"
    assert body["dataset_s3_key"] == "datasets/book-1/dataset_char-1_job-package-1.zip"
    assert body["dataset_s3_url"].endswith("dataset_char-1_job-package-1.zip")


def test_prepare_job_no_longer_auto_packages(monkeypatch):
    _patch_startup_dependencies(monkeypatch)
    _configure_storage(monkeypatch)
    api_server.dataset_jobs.clear()

    async def fake_prepare(*_args, **_kwargs):
        return [
            {
                "id": "clone_0001.wav",
                "url": "https://signed/item.wav",
                "s3_url": "s3://bucket/datasets/book-1/items/char-1_clone_0001.wav",
                "text": "hello there",
                "is_reference": False,
                "included": True,
            }
        ]

    monkeypatch.setattr(api_server, "prepare_dataset_items", fake_prepare)

    payload = _prepare_payload("job-prepare-auto")
    payload["approval_mode"] = "auto"

    with TestClient(api_server.app) as client:
        response = client.post("/dataset/prepare", json=payload)
        assert response.status_code == 200

        for _ in range(20):
            status_response = client.get("/dataset/status/job-prepare-auto")
            if status_response.json()["status"] == "completed":
                break
            time.sleep(0.01)

    body = status_response.json()
    assert body["kind"] == "prepare"
    assert body["phase"] == "awaiting_approval"
    assert body["dataset_s3_key"] is None
    assert body["dataset_s3_url"] is None


def test_package_dataset_uses_reference_item_without_training_on_it(monkeypatch):
    _patch_startup_dependencies(monkeypatch)
    _configure_storage(monkeypatch)

    downloaded = []
    uploaded_payloads = []

    async def fake_download(_storage, ref):
        downloaded.append(ref)
        return f"bytes-for:{ref}".encode("utf-8")

    async def fake_upload(_storage, payload, key, **_kwargs):
        assert payload
        uploaded_payloads.append((key, payload))
        return f"https://bucket/{key}"

    monkeypatch.setattr(dataset_jobs, "_download_bytes", fake_download)
    monkeypatch.setattr(dataset_jobs, "_upload_bytes", fake_upload)

    async def fake_exists(*_args, **_kwargs):
        return False

    monkeypatch.setattr(dataset_jobs, "_object_exists", fake_exists)

    package = asyncio.run(
        dataset_jobs.package_dataset(
            api_server.storage,
            book_id="book-1",
            character_id="char-1",
            job_id="job-package-ref",
            dataset_items=[
                {
                    "id": "ref_audio.wav",
                    "s3_url": "s3://bucket/ref.wav",
                    "text": "reference text",
                    "is_reference": True,
                    "included": True,
                },
                {
                    "id": "clip-1.wav",
                    "s3_url": "s3://bucket/clip-1.wav",
                    "text": "hello",
                    "is_reference": False,
                    "included": True,
                },
                {
                    "id": "clip-2.wav",
                    "s3_url": "s3://bucket/clip-2.wav",
                    "text": "world",
                    "is_reference": False,
                    "included": True,
                },
            ],
        )
    )

    assert package["dataset_s3_key"] == "datasets/book-1/dataset_char-1_job-package-ref.zip"
    assert downloaded == ["s3://bucket/ref.wav", "s3://bucket/clip-1.wav", "s3://bucket/clip-2.wav"]

    assert len(uploaded_payloads) == 1
    _key, payload = uploaded_payloads[0]
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        train_rows = [json.loads(line) for line in zf.read("train.jsonl").decode("utf-8").splitlines()]

    assert [row["audio"] for row in train_rows] == ["./data/clip-1.wav", "./data/clip-2.wav"]


def test_get_dataset_status_404_for_missing_job(monkeypatch):
    _patch_startup_dependencies(monkeypatch)
    api_server.dataset_jobs.clear()

    with TestClient(api_server.app) as client:
        response = client.get("/dataset/status/missing-job")

    assert response.status_code == 404
