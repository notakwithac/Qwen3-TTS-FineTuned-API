"""Schemas for experimental SonoEdit-style Qwen3-TTS edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


METADATA_FILENAME = "sonoedit_metadata.json"


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return value


def _span(value: Any, key: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{key} must be a two-item integer span")
    start, end = value
    if start < 0 or end <= start:
        raise ValueError(f"{key} must satisfy 0 <= start < end")
    return start, end


@dataclass(frozen=True)
class TargetPronunciationExample:
    audio_path: str
    transcript: str | None = None
    codec0_frame_span: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetPronunciationExample":
        return cls(
            audio_path=_require_str(data, "audio_path"),
            transcript=data.get("transcript"),
            codec0_frame_span=_span(data.get("codec0_frame_span"), "codec0_frame_span"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.codec0_frame_span is not None:
            data["codec0_frame_span"] = list(self.codec0_frame_span)
        return data


@dataclass(frozen=True)
class PreservationExample:
    sentence: str
    audio_path: str | None = None
    speaker_id: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreservationExample":
        return cls(
            sentence=_require_str(data, "sentence"),
            audio_path=data.get("audio_path"),
            speaker_id=data.get("speaker_id"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class SonoEditRequest:
    target_term: str
    source_sentence: str
    desired_pronunciation: TargetPronunciationExample
    preservation_manifest: list[PreservationExample]
    model_checkpoint_path: str
    output_checkpoint_path: str
    selected_edit_layers: list[str] = field(default_factory=list)
    target_frame_span: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SonoEditRequest":
        desired = data.get("desired_pronunciation")
        if not isinstance(desired, dict):
            raise ValueError("desired_pronunciation must be an object")
        preservation = data.get("preservation_manifest")
        if not isinstance(preservation, list):
            raise ValueError("preservation_manifest must be a list")
        return cls(
            target_term=_require_str(data, "target_term"),
            source_sentence=_require_str(data, "source_sentence"),
            desired_pronunciation=TargetPronunciationExample.from_dict(desired),
            preservation_manifest=[PreservationExample.from_dict(item) for item in preservation],
            model_checkpoint_path=_require_str(data, "model_checkpoint_path"),
            output_checkpoint_path=_require_str(data, "output_checkpoint_path"),
            selected_edit_layers=_string_list(data, "selected_edit_layers"),
            target_frame_span=_span(data.get("target_frame_span"), "target_frame_span"),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SonoEditRequest":
        import json

        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["desired_pronunciation"] = self.desired_pronunciation.to_dict()
        data["target_frame_span"] = list(self.target_frame_span) if self.target_frame_span else None
        return data

    @property
    def metadata_path(self) -> Path:
        return Path(self.output_checkpoint_path) / METADATA_FILENAME


@dataclass(frozen=True)
class SonoEditResult:
    request: SonoEditRequest
    edited_weights: list[str]
    selected_layers: list[str]
    causal_trace_scores: dict[str, float] = field(default_factory=dict)
    metadata_path: str | None = None
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["request"] = self.request.to_dict()
        return data

