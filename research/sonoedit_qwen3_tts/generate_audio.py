"""Generate a WAV from a custom-voice checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch


def normalize_speaker_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace(".", "_").strip("_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args(argv)

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint_path,
        device_map=args.device,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    )
    kwargs = {}
    if args.max_new_tokens is not None:
        kwargs["max_new_tokens"] = args.max_new_tokens
    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=normalize_speaker_name(args.speaker),
        instruct=args.instruct or None,
        **kwargs,
    )
    output = Path(args.output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wavs[0], sample_rate, format="WAV")
    print({"output_wav": str(output), "sample_rate": sample_rate})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
