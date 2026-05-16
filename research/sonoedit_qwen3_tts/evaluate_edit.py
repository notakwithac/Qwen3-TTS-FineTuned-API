"""Regression-evaluation artifact writer for experimental SonoEdit edits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .schema import SonoEditRequest


@dataclass(frozen=True)
class EditEvalRecord:
    model_path: str
    model_role: str
    example_role: str
    text: str
    target_term: str | None = None
    target_correctness: bool | None = None
    asr_text: str | None = None
    per: float | None = None
    wer: float | None = None
    speaker_similarity: float | None = None
    reviewer_notes: str | None = None


def build_eval_matrix(
    request: SonoEditRequest,
    source_model_path: str,
    edited_model_path: str,
    scorer: Callable[[EditEvalRecord], dict[str, Any]] | None = None,
) -> list[EditEvalRecord]:
    rows: list[EditEvalRecord] = []
    for model_role, model_path in [("source", source_model_path), ("edited", edited_model_path)]:
        target = EditEvalRecord(
            model_path=model_path,
            model_role=model_role,
            example_role="target",
            text=request.source_sentence,
            target_term=request.target_term,
        )
        rows.append(_score(target, scorer))
        for example in request.preservation_manifest:
            rows.append(
                _score(
                    EditEvalRecord(
                        model_path=model_path,
                        model_role=model_role,
                        example_role="preservation",
                        text=example.sentence,
                        target_term=None,
                    ),
                    scorer,
                )
            )
    return rows


def _score(record: EditEvalRecord, scorer: Callable[[EditEvalRecord], dict[str, Any]] | None) -> EditEvalRecord:
    if scorer is None:
        return record
    updates = scorer(record)
    return EditEvalRecord(**{**asdict(record), **updates})


def write_results_jsonl(records: list[EditEvalRecord], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def run_eval_artifact(
    request: SonoEditRequest,
    source_model_path: str,
    edited_model_path: str,
    output_path: str | Path,
    scorer: Callable[[EditEvalRecord], dict[str, Any]] | None = None,
) -> list[EditEvalRecord]:
    records = build_eval_matrix(request, source_model_path, edited_model_path, scorer)
    write_results_jsonl(records, output_path)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--source-model-path", required=True)
    parser.add_argument("--edited-model-path", required=True)
    parser.add_argument("--results-jsonl", required=True)
    args = parser.parse_args(argv)

    request = SonoEditRequest.from_json_file(args.request_json)
    records = run_eval_artifact(request, args.source_model_path, args.edited_model_path, args.results_jsonl)
    print(json.dumps({"results_jsonl": args.results_jsonl, "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

