"""Hugging Face Hub storage for fine-tuned Qwen3-TTS checkpoints."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pipeline import Job

logger = logging.getLogger(__name__)

REPO_TYPE = "model"
DEFAULT_NAMESPACE = "vaaniqo-finetuned-models"
DEFAULT_REPO_PREFIX = "Qwen3"


@dataclass(frozen=True)
class HfModelStorageConfig:
    token: str
    namespace: str
    repo_prefix: str

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.namespace)


def slugify(value: str, *, title_case: bool = True) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value or "")
    if not parts:
        return "Unknown"
    if title_case:
        return "-".join(part[:1].upper() + part[1:] for part in parts)
    return "-".join(part.lower() for part in parts)


def get_hf_model_storage_config() -> Optional[HfModelStorageConfig]:
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        return None
    namespace = (os.environ.get("HF_MODEL_NAMESPACE") or DEFAULT_NAMESPACE).strip()
    repo_prefix = (os.environ.get("HF_MODEL_REPO_PREFIX") or DEFAULT_REPO_PREFIX).strip()
    if not namespace:
        return None
    return HfModelStorageConfig(token=token, namespace=namespace, repo_prefix=repo_prefix)


def build_repo_id(job: Job, config: HfModelStorageConfig) -> str:
    book_label = slugify(job.book_id or "unsorted")
    speaker_label = slugify(job.speaker_name)
    job_label = slugify(job.job_id, title_case=False)
    repo_name = f"{config.repo_prefix}-{book_label}-{speaker_label}-{job_label}"
    return f"{config.namespace}/{repo_name}"


def hf_model_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def checkpoint_path_in_repo(checkpoint_dir_name: str) -> str:
    return f"checkpoints/{checkpoint_dir_name}"


def _model_card(job: Job, repo_id: str, checkpoint_dir_name: str, checkpoint_epoch: int) -> str:
    return f"""---
license: other
tags:
- qwen3-tts
- text-to-speech
- voice-clone
- pathnam
pipeline_tag: text-to-speech
---

# {repo_id.split("/", 1)[-1]}

Fine-tuned Qwen3-TTS voice checkpoint exported from the Qwen3-TTS pipeline.

## Job

- Job ID: `{job.job_id}`
- Speaker: {job.speaker_name}
- Book ID: `{job.book_id or "N/A"}`
- Character ID: `{job.character_id or "N/A"}`
- Checkpoint epoch: `{checkpoint_epoch}`
- Checkpoint path: `{checkpoint_path_in_repo(checkpoint_dir_name)}`
"""


def upload_checkpoint_to_hf(
    job: Job,
    checkpoint_dir: Path,
    checkpoint_epoch: int,
    *,
    checkpoint_dir_name: str,
) -> tuple[str, str]:
    """Upload a checkpoint directory to a public HF model repo.

    Returns (repo_id, model_url).
    """
    config = get_hf_model_storage_config()
    if config is None:
        raise ValueError("HF model storage is not configured (HF_TOKEN required)")

    from huggingface_hub import HfApi

    repo_id = build_repo_id(job, config)
    api = HfApi(token=config.token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        private=False,
        exist_ok=True,
        token=config.token,
    )

    path_in_repo = checkpoint_path_in_repo(checkpoint_dir_name)
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        token=config.token,
        commit_message=f"Upload {checkpoint_dir_name}",
    )

    metadata = {
        "job_id": job.job_id,
        "speaker_name": job.speaker_name,
        "book_id": job.book_id,
        "character_id": job.character_id,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_dir_name": checkpoint_dir_name,
        "checkpoint_path_in_repo": path_in_repo,
    }
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as handle:
        metadata_path = Path(handle.name)
        handle.write(json.dumps(metadata, indent=2))

    readme_path: Optional[Path] = None
    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            token=config.token,
            path_or_fileobj=str(metadata_path),
            path_in_repo="job_metadata.json",
            commit_message="Add job metadata",
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".md", encoding="utf-8") as handle:
            readme_path = Path(handle.name)
            handle.write(_model_card(job, repo_id, checkpoint_dir_name, checkpoint_epoch))
        api.upload_file(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            token=config.token,
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            commit_message="Add model card",
        )
    finally:
        metadata_path.unlink(missing_ok=True)
        if readme_path is not None:
            readme_path.unlink(missing_ok=True)

    model_url = hf_model_url(repo_id)
    logger.info("Uploaded checkpoint for job %s to HF repo %s", job.job_id, repo_id)
    return repo_id, model_url


def hf_checkpoint_repo_exists(repo_id: Optional[str]) -> bool:
    if not repo_id:
        return False
    config = get_hf_model_storage_config()
    if config is None:
        return bool(repo_id)

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=config.token)
    try:
        api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE, token=config.token)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as exc:
        logger.warning("HF repo existence check failed for %s: %s", repo_id, exc)
        return False


def restore_checkpoint_from_hf(
    job: Job,
    target_dir: Path,
    *,
    checkpoint_dir_name: str,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Download a checkpoint snapshot from HF into target_dir."""
    resolved_repo = repo_id or job.hf_model_repo
    if not resolved_repo:
        raise ValueError(f"Job {job.job_id} has no HF model repo")

    config = get_hf_model_storage_config()
    token = config.token if config is not None else (os.environ.get("HF_TOKEN") or None)

    from huggingface_hub import snapshot_download

    if filename:
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("Hugging Face filename must be a relative basename")
        if not filename.lower().endswith(".zip"):
            raise ValueError("Legacy Hugging Face checkpoint files must be ZIP archives")

        from huggingface_hub import hf_hub_download

        logger.info("Downloading HF checkpoint archive %s/%s", resolved_repo, filename)
        archive_path = Path(
            hf_hub_download(
                repo_id=resolved_repo,
                repo_type=REPO_TYPE,
                filename=filename,
                token=token,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_dir = Path(temp_dir) / "checkpoint"
            extracted_dir.mkdir()
            root = extracted_dir.resolve()
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    destination = (extracted_dir / member.filename).resolve()
                    if destination != root and root not in destination.parents:
                        raise ValueError(f"Unsafe path in Hugging Face checkpoint ZIP: {member.filename}")
                archive.extractall(extracted_dir)
            if not any(extracted_dir.iterdir()):
                raise ValueError(f"HF repo {resolved_repo} contains an empty checkpoint ZIP")
            target_dir.mkdir(parents=True, exist_ok=True)
            for item in extracted_dir.iterdir():
                dest = target_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                fd = os.open(archive_path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError as exc:
                logger.debug("Could not evict HF archive from page cache: %s", exc)

        checkpoint_path = str(target_dir.resolve())
        logger.info("Restored checkpoint for job %s from HF repo %s to %s", job.job_id, resolved_repo, checkpoint_path)
        return checkpoint_path

    path_in_repo = checkpoint_path_in_repo(checkpoint_dir_name)
    cache_dir = snapshot_download(
        repo_id=resolved_repo,
        repo_type=REPO_TYPE,
        token=token,
        allow_patterns=[f"{path_in_repo}/**"],
    )

    source_dir = Path(cache_dir) / path_in_repo
    if not source_dir.is_dir() or not any(source_dir.iterdir()):
        raise ValueError(
            f"HF repo {resolved_repo} does not contain checkpoint files at {path_in_repo}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    checkpoint_path = str(target_dir.resolve())
    logger.info("Restored checkpoint for job %s from HF repo %s to %s", job.job_id, resolved_repo, checkpoint_path)
    return checkpoint_path
