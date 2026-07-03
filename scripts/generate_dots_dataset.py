"""Generate a Qwen3-TTS fine-tuning dataset with dots.tts.

The dots runtime is imported lazily so this module can be tested from the
Qwen3-TTS environment. Install and run it from a separate virtual environment;
dots.tts currently requires NumPy 2 while this project pins NumPy below 2.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class DatasetPrompt:
    filename: str
    text: str


class DotsRuntime(Protocol):
    def generate(self, **kwargs: Any) -> dict[str, Any]: ...


def _looks_like_hf_repo_id(value: str) -> bool:
    if "\\" in value or value.startswith((".", "/")) or ":" in value:
        return False
    return value.count("/") == 1


def resolve_model_spec(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("--model must not be empty")
    candidate = Path(normalized)
    if candidate.is_dir():
        return str(candidate.resolve())
    if candidate.exists():
        raise FileNotFoundError(candidate)
    if _looks_like_hf_repo_id(normalized):
        return normalized
    raise FileNotFoundError(candidate)


def _safe_filename(value: str, line_number: int) -> str:
    filename = Path(value).name
    if filename != value or filename in {"", ".", ".."}:
        raise ValueError(f"Line {line_number}: filename must be a plain file name")
    if Path(filename).suffix.lower() != ".wav":
        filename = f"{filename}.wav"
    return filename


def load_prompts(path: Path) -> list[DatasetPrompt]:
    prompts: list[DatasetPrompt] = []
    filenames: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number}: each entry must be a JSON object")
        text = str(item.get("text") or "").strip()
        if not text:
            raise ValueError(f"Line {line_number}: text cannot be empty")
        filename = _safe_filename(
            str(item.get("filename") or item.get("id") or f"sample_{line_number:04d}.wav"),
            line_number,
        )
        if filename in filenames:
            raise ValueError(f"Line {line_number}: duplicate filename {filename!r}")
        filenames.add(filename)
        prompts.append(DatasetPrompt(filename=filename, text=text))
    if not prompts:
        raise ValueError("The prompt manifest contains no entries")
    return prompts


def _write_train_jsonl(output_dir: Path, prompts: Iterable[DatasetPrompt]) -> Path:
    manifest_path = output_dir / "train_raw.jsonl"
    lines = [
        json.dumps(
            {
                "audio": f"./data/{prompt.filename}",
                "text": prompt.text,
                "ref_audio": "./data/ref_audio.wav",
            },
            ensure_ascii=False,
        )
        for prompt in prompts
    ]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def generate_dataset(
    runtime: DotsRuntime,
    *,
    prompts: list[DatasetPrompt],
    prompt_audio: Path,
    prompt_text: str,
    output_dir: Path,
    language: str | None,
    num_steps: int,
    speaker_scale: float,
    normalize_text: bool,
    overwrite: bool,
) -> Path:
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    ref_audio = data_dir / "ref_audio.wav"
    if ref_audio.exists() and not overwrite:
        raise FileExistsError(f"{ref_audio} already exists; pass --overwrite to replace it")
    shutil.copyfile(prompt_audio, ref_audio)

    completed: list[DatasetPrompt] = []
    for index, prompt in enumerate(prompts, 1):
        output_path = data_dir / prompt.filename
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        print(f"[{index}/{len(prompts)}] Generating {prompt.filename}")
        result = runtime.generate(
            text=prompt.text,
            prompt_audio_path=str(prompt_audio),
            prompt_text=prompt_text,
            language=language,
            template_name="tts",
            num_steps=num_steps,
            speaker_scale=speaker_scale,
            normalize_text=normalize_text,
        )
        sf.write(
            output_path,
            result["audio"].float().cpu().squeeze().numpy(),
            int(result["sample_rate"]),
        )
        completed.append(prompt)
        _write_train_jsonl(output_dir, completed)
    return output_dir / "train_raw.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand one authorized reference voice into a Qwen3-TTS training dataset."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL rows with text and optional filename")
    parser.add_argument("--prompt-audio", type=Path, required=True, help="Authorized reference WAV")
    parser.add_argument("--prompt-text", required=True, help="Exact transcript of the reference WAV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="rednote-hilab/dots.tts-soar")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--speaker-scale", type=float, default=1.5)
    parser.add_argument("--max-generate-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-text", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_steps < 1:
        raise ValueError("--num-steps must be positive")
    if not args.prompt_text.strip():
        raise ValueError("--prompt-text must contain the exact reference transcript")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.prompt_audio.is_file():
        raise FileNotFoundError(args.prompt_audio)
    model_spec = resolve_model_spec(args.model)

    from dots_tts.runtime import DotsTtsRuntime
    from dots_tts.utils.util import seed_everything

    seed_everything(args.seed)
    runtime = DotsTtsRuntime.from_pretrained(
        model_spec,
        precision=args.precision,
        max_generate_length=args.max_generate_length,
    )
    output_manifest = generate_dataset(
        runtime,
        prompts=load_prompts(args.manifest),
        prompt_audio=args.prompt_audio.resolve(),
        prompt_text=args.prompt_text.strip(),
        output_dir=args.output_dir,
        language=args.language,
        num_steps=args.num_steps,
        speaker_scale=args.speaker_scale,
        normalize_text=args.normalize_text,
        overwrite=args.overwrite,
    )
    print(f"Dataset ready: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
