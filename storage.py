# coding=utf-8
"""E2E Object Storage client — S3-compatible storage via boto3.

Configure via environment variables:
    E2E_ACCESS_KEY      — S3 access key
    E2E_SECRET_KEY      — S3 secret key
    E2E_BUCKET          — bucket name (default: qwen3-tts)
    E2E_ENDPOINT_URL    — endpoint (default: https://objectstore.e2enetworks.net)
    E2E_PUBLIC_ENDPOINT_URL — endpoint to return in object URLs (default: E2E_ENDPOINT_URL)
    E2E_REGION           — region (default: us-east-1)
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import unquote, urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from ops_logger import ops_log

logger = logging.getLogger(__name__)


def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _first_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return default


def _normalize_s3_key(value: str) -> tuple[Optional[str], str]:
    raw = str(value).strip()
    if not raw.startswith("s3://"):
        return None, raw
    parsed = urlparse(raw)
    return parsed.netloc or None, parsed.path.lstrip("/")


def _split_url_bucket_key(value: str) -> tuple[Optional[str], str]:
    parsed = urlparse(str(value).strip())
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if not parts:
        return None, ""
    return parts[0], "/".join(parts[1:])


class _StorageBackend:
    def __init__(
        self,
        *,
        name: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        endpoint_url: str,
        region: str,
    ):
        self.name = name
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.region = region
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self.access_key and self.secret_key and self.bucket)

    @property
    def client(self):
        if self._client is None:
            if not self.is_configured:
                raise RuntimeError(f"Storage backend {self.name} is not configured.")
            self._client = boto3.client(
                "s3",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                endpoint_url=self.endpoint_url or None,
                region_name=self.region,
                config=BotoConfig(
                    signature_version="s3v4",
                    max_pool_connections=50,
                ),
            )
        return self._client

    def matches_bucket(self, bucket: Optional[str]) -> bool:
        return not bucket or bucket == self.bucket


class StorageClient:
    """S3-compatible storage client for E2E Networks Object Storage."""

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.access_key = access_key or _get_env("E2E_ACCESS_KEY")
        self.secret_key = secret_key or _get_env("E2E_SECRET_KEY")
        self.bucket = bucket or _get_env("E2E_BUCKET", "qwen3-tts")
        self.endpoint_url = endpoint_url or _get_env(
            "E2E_ENDPOINT_URL", "https://objectstore.e2enetworks.net"
        )
        self.public_endpoint_url = _get_env("E2E_PUBLIC_ENDPOINT_URL", self.endpoint_url)
        self.region = region or _get_env("E2E_REGION", "us-east-1")

        self._primary_backend = _StorageBackend(
            name="primary",
            access_key=self.access_key,
            secret_key=self.secret_key,
            bucket=self.bucket,
            endpoint_url=self.endpoint_url,
            region=self.region,
        )
        self._fallback_backends = self._build_fallback_backends()

    @property
    def is_configured(self) -> bool:
        """Check if credentials are set."""
        return self._primary_backend.is_configured

    @property
    def has_read_backend(self) -> bool:
        return any(backend.is_configured for backend in self._read_backends())

    @property
    def client(self):
        """Lazy-init the boto3 S3 client."""
        if not self.is_configured:
            raise RuntimeError(
                "Storage not configured. Set E2E_ACCESS_KEY and E2E_SECRET_KEY "
                "environment variables."
            )
        return self._primary_backend.client

    def _build_fallback_backends(self) -> list[_StorageBackend]:
        s3_access_key = _first_env("S3_ACCESS_KEY", "AWS_ACCESS_KEY_ID")
        s3_secret_key = _first_env("S3_SECRET_KEY", "AWS_SECRET_ACCESS_KEY")
        s3_bucket = _first_env("S3_BUCKET", "AWS_S3_BUCKET")
        s3_endpoint_url = _first_env("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL")
        s3_region = _first_env("S3_REGION", "AWS_DEFAULT_REGION", "AWS_REGION", default="us-east-1")

        if not (s3_access_key and s3_secret_key and s3_bucket):
            logger.warning(
                "No fallback S3 read backend configured. Set S3_ACCESS_KEY, "
                "S3_SECRET_KEY, and S3_BUCKET to check AWS S3 in addition to E2E storage."
            )
            return []

        primary_signature = (
            self.access_key,
            self.secret_key,
            self.bucket,
            self.endpoint_url,
            self.region,
        )
        fallback_signature = (
            s3_access_key,
            s3_secret_key,
            s3_bucket,
            s3_endpoint_url,
            s3_region,
        )
        if fallback_signature == primary_signature:
            return []

        return [
            _StorageBackend(
                name="s3-fallback",
                access_key=s3_access_key,
                secret_key=s3_secret_key,
                bucket=s3_bucket,
                endpoint_url=s3_endpoint_url,
                region=s3_region,
            )
        ]

    def _read_backends(self) -> list[_StorageBackend]:
        return [self._primary_backend, *self._fallback_backends]

    def describe_read_backends(self) -> list[dict[str, str]]:
        return [
            {
                "name": backend.name,
                "bucket": backend.bucket,
                "endpoint_url": backend.endpoint_url or "aws-default",
                "region": backend.region,
                "configured": str(backend.is_configured),
            }
            for backend in self._read_backends()
        ]

    def _candidate_read_backends(self, key: str) -> tuple[str, list[_StorageBackend]]:
        bucket, normalized_key = self._normalize_storage_ref(key)
        return normalized_key, [
            backend
            for backend in self._read_backends()
            if backend.is_configured and backend.matches_bucket(bucket)
        ]

    def _configured_buckets(self) -> set[str]:
        return {
            backend.bucket
            for backend in self._read_backends()
            if backend.is_configured and backend.bucket
        }

    def _normalize_storage_ref(self, value: str) -> tuple[Optional[str], str]:
        raw = str(value).strip()
        parsed = urlparse(raw)

        if parsed.scheme == "s3":
            return parsed.netloc or None, parsed.path.lstrip("/")

        if parsed.scheme in ("http", "https"):
            bucket, key = _split_url_bucket_key(raw)
            buckets = self._configured_buckets()
            host = parsed.netloc.split("@")[-1].split(":")[0]
            path_key = parsed.path.lstrip("/")
            for configured_bucket in buckets:
                if host == configured_bucket or host.startswith(f"{configured_bucket}."):
                    return configured_bucket, unquote(path_key)
            for configured_bucket in buckets:
                prefix = f"{configured_bucket}/"
                if key.startswith(prefix):
                    return configured_bucket, key[len(prefix):]
            if bucket in buckets:
                return bucket, key
            return bucket, key

        return None, raw

    def can_resolve_storage_ref(self, value: str) -> bool:
        raw = str(value).strip()
        parsed = urlparse(raw)
        if parsed.scheme == "s3":
            bucket, _key = self._normalize_storage_ref(raw)
            return bool(self._candidate_backends_for_bucket(bucket))
        if parsed.scheme in ("http", "https"):
            bucket, key = self._normalize_storage_ref(raw)
            buckets = self._configured_buckets()
            return bool(key) and bool(bucket) and bucket in buckets
        return bool(raw) and not parsed.scheme

    def _candidate_backends_for_bucket(self, bucket: Optional[str]) -> list[_StorageBackend]:
        return [
            backend
            for backend in self._read_backends()
            if backend.is_configured and backend.matches_bucket(bucket)
        ]

    def ensure_bucket(self):
        """Create the bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    # -- Upload methods -------------------------------------------------------

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
    ) -> str:
        """Upload bytes to S3 and return the object URL.

        Args:
            data: Raw bytes to upload.
            key: S3 object key (path within bucket).
            content_type: MIME type of the content.
            metadata: Custom metadata (x-amz-meta-).

        Returns:
            Public URL of the uploaded object.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Uploading {len(data)} bytes to S3: {self.bucket}/{key}")
        
        with ops_log.operation("s3_put_object", extra={"key": key, "size": len(data), "bucket": self.bucket}):
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata or {},
            )
        logger.info(f"Upload complete: {self._object_url(key)}")
        return self._object_url(key)

    def upload_file(self, local_path: str, key: str, content_type: Optional[str] = None, metadata: Optional[dict] = None) -> str:
        """Upload a local file to S3.

        Args:
            local_path: Path to the local file.
            key: S3 object key.
            content_type: MIME type (auto-detected if omitted).
            metadata: Custom metadata (x-amz-meta-).

        Returns:
            URL of the uploaded object.
        """
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata
        with ops_log.operation("s3_upload_file", extra={"key": key, "local_path": local_path}):
            self.client.upload_file(local_path, self.bucket, key, ExtraArgs=extra_args or None)
        return self._object_url(key)

    def upload_wav(
        self,
        wav_bytes: bytes,
        job_id: str,
        filename: Optional[str] = None,
        prefix: Optional[str] = None,
        model_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Upload a WAV file with a structured key.

        Key format: {prefix or f'audio/{job_id}'}/{filename}

        Args:
            wav_bytes: WAV audio bytes.
            job_id: Job ID for the directory structure (fallback if prefix not provided).
            filename: Custom filename (default: timestamped).
            prefix: Custom S3 prefix (folder path).
            model_id: Model ID for metadata (x-amz-meta-model-id).
            metadata: Additional custom object metadata.

        Returns:
            URL of the uploaded WAV file.
        """
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{ts}.wav"
        
        base_prefix = prefix or f"audio/{job_id}"
        key = f"{base_prefix}/{filename}"
        
        combined_metadata = dict(metadata or {})
        if model_id:
            combined_metadata.setdefault("model-id", model_id)

        return self.upload_bytes(
            wav_bytes,
            key,
            content_type="audio/wav",
            metadata=combined_metadata or None,
        )

    def upload_text(self, text: str, key: str) -> str:
        """Upload a text file to S3."""
        return self.upload_bytes(text.encode("utf-8"), key, content_type="text/plain")

    # -- Download methods -----------------------------------------------------

    def download_bytes(self, key: str) -> bytes:
        """Download an object as bytes."""
        normalized_key, backends = self._candidate_read_backends(key)
        last_error = None
        for backend in backends:
            logger.info(
                "Trying S3 GET bytes via backend=%s bucket=%s endpoint=%s key=%s",
                backend.name,
                backend.bucket,
                backend.endpoint_url or "aws-default",
                normalized_key,
            )
            try:
                with ops_log.operation(
                    "s3_get_object",
                    extra={
                        "key": normalized_key,
                        "bucket": backend.bucket,
                        "backend": backend.name,
                    },
                ):
                    response = backend.client.get_object(Bucket=backend.bucket, Key=normalized_key)
                    return response["Body"].read()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "S3 GET bytes failed via backend=%s bucket=%s endpoint=%s key=%s: %s",
                    backend.name,
                    backend.bucket,
                    backend.endpoint_url or "aws-default",
                    normalized_key,
                    exc,
                )
                continue
        raise FileNotFoundError(f"S3 object not found: {key}") from last_error

    def download_file(self, key: str, local_path: str):
        """Download an object to a local file."""
        normalized_key, backends = self._candidate_read_backends(key)
        last_error = None
        for backend in backends:
            logger.info(
                "Trying S3 download via backend=%s bucket=%s endpoint=%s key=%s",
                backend.name,
                backend.bucket,
                backend.endpoint_url or "aws-default",
                normalized_key,
            )
            try:
                with ops_log.operation(
                    "s3_download_file",
                    extra={
                        "key": normalized_key,
                        "local_path": local_path,
                        "bucket": backend.bucket,
                        "backend": backend.name,
                    },
                ):
                    backend.client.download_file(backend.bucket, normalized_key, local_path)
                    return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "S3 download failed via backend=%s bucket=%s endpoint=%s key=%s: %s",
                    backend.name,
                    backend.bucket,
                    backend.endpoint_url or "aws-default",
                    normalized_key,
                    exc,
                )
                continue
        raise FileNotFoundError(f"S3 object not found: {key}") from last_error

    # -- List / Delete --------------------------------------------------------

    def list_objects(self, prefix: str = "") -> list:
        """List objects under a prefix."""
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def object_exists(self, key: str) -> bool:
        """Check if an object exists."""
        normalized_key, backends = self._candidate_read_backends(key)
        for backend in backends:
            try:
                backend.client.head_object(Bucket=backend.bucket, Key=normalized_key)
                return True
            except Exception:
                continue
        return False

    def delete_object(self, key: str):
        """Delete an object."""
        with ops_log.operation("s3_delete_object", extra={"key": key}):
            self.client.delete_object(Bucket=self.bucket, Key=key)

    def copy_object(self, source_key: str, dest_key: str):
        """Copy an object within the bucket."""
        copy_source = {'Bucket': self.bucket, 'Key': source_key}
        self.client.copy(copy_source, self.bucket, dest_key)

    def move_object(self, source_key: str, dest_key: str):
        """Move an object by copying it and then deleting the source."""
        self.copy_object(source_key, dest_key)
        self.delete_object(source_key)

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for temporary access.

        Args:
            key: S3 object key.
            expires_in: URL validity in seconds (default: 1 hour).

        Returns:
            Presigned URL string.
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def _resolve_read_backend(self, key: str) -> tuple[str, _StorageBackend]:
        normalized_key, backends = self._candidate_read_backends(key)
        if not backends:
            raise FileNotFoundError(f"No configured storage backend can read {key}")

        last_error = None
        for backend in backends:
            try:
                backend.client.head_object(Bucket=backend.bucket, Key=normalized_key)
                return normalized_key, backend
            except Exception as exc:
                last_error = exc

        if isinstance(last_error, ClientError):
            raise FileNotFoundError(f"S3 object not found: {key}") from last_error
        raise FileNotFoundError(f"S3 object not found: {key}") from last_error

    # -- Internal -------------------------------------------------------------

    def _object_url(self, key: str) -> str:
        """Build the full object URL."""
        return f"{self.public_endpoint_url}/{self.bucket}/{key}"


# Singleton instance — configured from env vars
storage = StorageClient()
