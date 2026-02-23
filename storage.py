# coding=utf-8
"""E2E Object Storage client — S3-compatible storage via boto3.

Configure via environment variables:
    E2E_ACCESS_KEY      — S3 access key
    E2E_SECRET_KEY      — S3 secret key
    E2E_BUCKET          — bucket name (default: qwen3-tts)
    E2E_ENDPOINT_URL    — endpoint (default: https://objectstore.e2enetworks.net)
    E2E_REGION           — region (default: us-east-1)
"""

import io
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin

import boto3
from botocore.config import Config as BotoConfig


def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


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
        self.region = region or _get_env("E2E_REGION", "us-east-1")

        self._client = None

    @property
    def is_configured(self) -> bool:
        """Check if credentials are set."""
        return bool(self.access_key and self.secret_key)

    @property
    def client(self):
        """Lazy-init the boto3 S3 client."""
        if self._client is None:
            if not self.is_configured:
                raise RuntimeError(
                    "Storage not configured. Set E2E_ACCESS_KEY and E2E_SECRET_KEY "
                    "environment variables."
                )
            self._client = boto3.client(
                "s3",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                endpoint_url=self.endpoint_url,
                region_name=self.region,
                config=BotoConfig(
                    signature_version="s3v4",
                    max_pool_connections=50,
                ),
            )
        return self._client

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
    ) -> str:
        """Upload bytes to S3 and return the object URL.

        Args:
            data: Raw bytes to upload.
            key: S3 object key (path within bucket).
            content_type: MIME type of the content.

        Returns:
            Public URL of the uploaded object.
        """
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return self._object_url(key)

    def upload_file(self, local_path: str, key: str, content_type: Optional[str] = None) -> str:
        """Upload a local file to S3.

        Args:
            local_path: Path to the local file.
            key: S3 object key.
            content_type: MIME type (auto-detected if omitted).

        Returns:
            URL of the uploaded object.
        """
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        self.client.upload_file(local_path, self.bucket, key, ExtraArgs=extra_args or None)
        return self._object_url(key)

    def upload_wav(self, wav_bytes: bytes, job_id: str, filename: Optional[str] = None, prefix: Optional[str] = None) -> str:
        """Upload a WAV file with a structured key.

        Key format: {prefix or f'audio/{job_id}'}/{filename}

        Args:
            wav_bytes: WAV audio bytes.
            job_id: Job ID for the directory structure (fallback if prefix not provided).
            filename: Custom filename (default: timestamped).
            prefix: Custom S3 prefix (folder path).

        Returns:
            URL of the uploaded WAV file.
        """
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{ts}.wav"
        
        base_prefix = prefix or f"audio/{job_id}"
        key = f"{base_prefix}/{filename}"
        return self.upload_bytes(wav_bytes, key, content_type="audio/wav")

    def upload_text(self, text: str, key: str) -> str:
        """Upload a text file to S3."""
        return self.upload_bytes(text.encode("utf-8"), key, content_type="text/plain")

    # -- Download methods -----------------------------------------------------

    def download_bytes(self, key: str) -> bytes:
        """Download an object as bytes."""
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def download_file(self, key: str, local_path: str):
        """Download an object to a local file."""
        self.client.download_file(self.bucket, key, local_path)

    # -- List / Delete --------------------------------------------------------

    def list_objects(self, prefix: str = "") -> list:
        """List objects under a prefix."""
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def object_exists(self, key: str) -> bool:
        """Check if an object exists."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete_object(self, key: str):
        """Delete an object."""
        self.client.delete_object(Bucket=self.bucket, Key=key)

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

    # -- Internal -------------------------------------------------------------

    def _object_url(self, key: str) -> str:
        """Build the full object URL."""
        return f"{self.endpoint_url}/{self.bucket}/{key}"


# Singleton instance — configured from env vars
storage = StorageClient()
