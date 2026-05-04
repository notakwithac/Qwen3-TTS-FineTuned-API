from pathlib import Path

import storage as storage_module
from storage import StorageClient


class _FakeS3Client:
    def __init__(self, existing, *, fail_head=False):
        self.existing = set(existing)
        self.fail_head = fail_head
        self.head_calls = []
        self.get_calls = []
        self.download_calls = []

    def head_object(self, *, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        if self.fail_head:
            raise Exception("head forbidden")
        if (Bucket, Key) not in self.existing:
            raise Exception("not found")

    def get_object(self, *, Bucket, Key):
        self.get_calls.append((Bucket, Key))
        if (Bucket, Key) not in self.existing:
            raise Exception("not found")
        return {"Body": _FakeBody(f"{Bucket}/{Key}".encode("utf-8"))}

    def download_file(self, bucket, key, local_path):
        self.download_calls.append((bucket, key, local_path))
        if (bucket, key) not in self.existing:
            raise Exception("not found")
        Path(local_path).write_bytes(f"{bucket}/{key}".encode("utf-8"))


class _FakeBody:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


def test_object_exists_checks_s3_fallback_after_primary_miss(monkeypatch):
    clients = [
        _FakeS3Client(existing=[]),
        _FakeS3Client(existing=[("pathnam-ai", "models/book/voice/checkpoint.zip")]),
    ]

    monkeypatch.setenv("E2E_ACCESS_KEY", "minio-key")
    monkeypatch.setenv("E2E_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("E2E_BUCKET", "qwen3-tts")
    monkeypatch.setenv("E2E_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("E2E_REGION", "us-east-1")
    monkeypatch.setenv("S3_ACCESS_KEY", "aws-key")
    monkeypatch.setenv("S3_SECRET_KEY", "aws-secret")
    monkeypatch.setenv("S3_BUCKET", "pathnam-ai")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.ap-southeast-2.amazonaws.com")
    monkeypatch.setenv("S3_REGION", "ap-southeast-2")
    monkeypatch.setattr(storage_module.boto3, "client", lambda *args, **kwargs: clients.pop(0))

    client = StorageClient()

    assert client.object_exists("models/book/voice/checkpoint.zip") is True


def test_download_file_uses_bucket_from_s3_uri(tmp_path, monkeypatch):
    clients = [_FakeS3Client(existing=[("pathnam-ai", "models/book/voice/checkpoint.zip")])]

    monkeypatch.setenv("E2E_ACCESS_KEY", "minio-key")
    monkeypatch.setenv("E2E_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("E2E_BUCKET", "qwen3-tts")
    monkeypatch.setenv("E2E_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("E2E_REGION", "us-east-1")
    monkeypatch.setenv("S3_ACCESS_KEY", "aws-key")
    monkeypatch.setenv("S3_SECRET_KEY", "aws-secret")
    monkeypatch.setenv("S3_BUCKET", "pathnam-ai")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.ap-southeast-2.amazonaws.com")
    monkeypatch.setenv("S3_REGION", "ap-southeast-2")
    monkeypatch.setattr(storage_module.boto3, "client", lambda *args, **kwargs: clients.pop(0))

    client = StorageClient()
    local_path = tmp_path / "checkpoint.zip"

    client.download_file("s3://pathnam-ai/models/book/voice/checkpoint.zip", str(local_path))

    assert local_path.read_bytes() == b"pathnam-ai/models/book/voice/checkpoint.zip"


def test_download_bytes_does_not_require_head_object_permission(monkeypatch):
    clients = [
        _FakeS3Client(existing=[]),
        _FakeS3Client(
            existing=[("pathnam-ai", "jobs/7e0bd35e9e72/job.json")],
            fail_head=True,
        ),
    ]

    monkeypatch.setenv("E2E_ACCESS_KEY", "minio-key")
    monkeypatch.setenv("E2E_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("E2E_BUCKET", "qwen3-tts")
    monkeypatch.setenv("E2E_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("E2E_REGION", "us-east-1")
    monkeypatch.setenv("S3_ACCESS_KEY", "aws-key")
    monkeypatch.setenv("S3_SECRET_KEY", "aws-secret")
    monkeypatch.setenv("S3_BUCKET", "pathnam-ai")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.ap-southeast-2.amazonaws.com")
    monkeypatch.setenv("S3_REGION", "ap-southeast-2")
    monkeypatch.setattr(storage_module.boto3, "client", lambda *args, **kwargs: clients.pop(0))

    client = StorageClient()

    assert client.download_bytes("jobs/7e0bd35e9e72/job.json") == b"pathnam-ai/jobs/7e0bd35e9e72/job.json"


def test_download_bytes_resolves_local_minio_url_with_embedded_bucket(monkeypatch):
    clients = [
        _FakeS3Client(existing=[("qwen3-tts", "audio/voice_design/ref.wav")]),
    ]

    monkeypatch.setenv("E2E_ACCESS_KEY", "minio-key")
    monkeypatch.setenv("E2E_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("E2E_BUCKET", "qwen3-tts")
    monkeypatch.setenv("E2E_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("E2E_REGION", "us-east-1")
    monkeypatch.setenv("S3_ACCESS_KEY", "aws-key")
    monkeypatch.setenv("S3_SECRET_KEY", "aws-secret")
    monkeypatch.setenv("S3_BUCKET", "pathnam-ai")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.ap-southeast-2.amazonaws.com")
    monkeypatch.setenv("S3_REGION", "ap-southeast-2")
    monkeypatch.setattr(storage_module.boto3, "client", lambda *args, **kwargs: clients.pop(0))

    client = StorageClient()

    assert client.can_resolve_storage_ref(
        "http://minio:9000/pathnam-ai/qwen3-tts/audio/voice_design/ref.wav?X-Amz-Signature=test"
    )
    assert (
        client.download_bytes(
            "http://minio:9000/pathnam-ai/qwen3-tts/audio/voice_design/ref.wav?X-Amz-Signature=test"
        )
        == b"qwen3-tts/audio/voice_design/ref.wav"
    )


def test_unknown_public_http_url_is_not_claimed_by_storage(monkeypatch):
    monkeypatch.setenv("E2E_ACCESS_KEY", "minio-key")
    monkeypatch.setenv("E2E_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("E2E_BUCKET", "qwen3-tts")
    monkeypatch.setenv("E2E_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_SECRET_KEY", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)

    client = StorageClient()

    assert not client.can_resolve_storage_ref("https://example.com/audio/ref.wav")


def test_download_bytes_resolves_virtual_hosted_s3_url(monkeypatch):
    clients = [
        _FakeS3Client(existing=[("pathnam-ai", "audio/voice_design/ref.wav")]),
    ]

    monkeypatch.setenv("E2E_ACCESS_KEY", "minio-key")
    monkeypatch.setenv("E2E_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("E2E_BUCKET", "qwen3-tts")
    monkeypatch.setenv("E2E_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "aws-key")
    monkeypatch.setenv("S3_SECRET_KEY", "aws-secret")
    monkeypatch.setenv("S3_BUCKET", "pathnam-ai")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.ap-southeast-2.amazonaws.com")
    monkeypatch.setenv("S3_REGION", "ap-southeast-2")
    monkeypatch.setattr(storage_module.boto3, "client", lambda *args, **kwargs: clients.pop(0))

    client = StorageClient()

    assert (
        client.download_bytes(
            "https://pathnam-ai.s3.ap-southeast-2.amazonaws.com/audio/voice_design/ref.wav?sig=test"
        )
        == b"pathnam-ai/audio/voice_design/ref.wav"
    )
