import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _time_call(fn, repeats: int = 1):
    durations = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - started)
    return {
        "last_result": result,
        "runs": durations,
        "mean_s": statistics.mean(durations),
        "min_s": min(durations),
        "max_s": max(durations),
    }


def _write_temp_ref_audio() -> str:
    wav = np.zeros((24000,), dtype=np.float32)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    handle.close()
    sf.write(handle.name, wav, 24000)
    return handle.name


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3TTSModel wrapper inference paths.")
    parser.add_argument("--model", required=True, help="Model path or HF repo id")
    parser.add_argument("--mode", choices=["custom_voice", "voice_design", "voice_clone"], default="custom_voice")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--language", default="English")
    parser.add_argument("--text", default="The lighthouse watched the sea in silence.")
    parser.add_argument("--instruct", default="Speak calmly and clearly.")
    parser.add_argument("--ref-text", default="This is a reusable reference prompt.")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--compile-enabled", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    ref_audio = args.ref_audio
    temp_ref = None
    if args.mode == "voice_clone" and not ref_audio:
        temp_ref = _write_temp_ref_audio()
        ref_audio = temp_ref

    load_stats = _time_call(
        lambda: Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            compile_enabled=args.compile_enabled,
            compile_mode=args.compile_mode,
            matmul_precision=args.matmul_precision,
        )
    )
    model = load_stats["last_result"]

    if args.warmup:
        warmup_kwargs = {}
        if args.mode == "custom_voice":
            warmup_kwargs["speaker"] = args.speaker
            warmup_kwargs["instruct"] = args.instruct
        elif args.mode == "voice_design":
            warmup_kwargs["instruct"] = args.instruct
        else:
            warmup_kwargs["ref_audio"] = ref_audio
            warmup_kwargs["ref_text"] = args.ref_text
        model.warmup_for_inference(
            mode=args.mode,
            text=args.text,
            language=args.language,
            max_new_tokens=64,
            **warmup_kwargs,
        )

    def _single():
        if args.mode == "custom_voice":
            return model.generate_custom_voice(
                text=args.text,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                max_new_tokens=256,
            )
        if args.mode == "voice_design":
            return model.generate_voice_design(
                text=args.text,
                instruct=args.instruct,
                language=args.language,
                max_new_tokens=256,
            )
        return model.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=ref_audio,
            ref_text=args.ref_text,
            max_new_tokens=256,
        )

    def _batch():
        batch_texts = [args.text, args.text + " Again."]
        if args.mode == "custom_voice":
            return model.generate_custom_voice(
                text=batch_texts,
                speaker=[args.speaker, args.speaker],
                language=[args.language, args.language],
                instruct=[args.instruct, args.instruct],
                max_new_tokens=256,
            )
        if args.mode == "voice_design":
            return model.generate_voice_design(
                text=batch_texts,
                instruct=[args.instruct, args.instruct],
                language=[args.language, args.language],
                max_new_tokens=256,
            )
        return model.generate_voice_clone(
            text=batch_texts,
            language=[args.language, args.language],
            ref_audio=[ref_audio, ref_audio],
            ref_text=[args.ref_text, args.ref_text],
            max_new_tokens=256,
        )

    single_stats = _time_call(_single, repeats=args.repeats)
    batch_stats = _time_call(_batch, repeats=args.repeats)

    prompt_reuse_stats = None
    if args.mode == "voice_clone":
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=[ref_audio],
            ref_text=[args.ref_text],
        )
        prompt_reuse_stats = _time_call(
            lambda: model.generate_voice_clone(
                text=[args.text, args.text + " Again."],
                language=[args.language, args.language],
                voice_clone_prompt=prompt_items * 2,
                max_new_tokens=256,
            ),
            repeats=args.repeats,
        )

    print(f"load_mean_s={load_stats['mean_s']:.4f}")
    print(f"single_mean_s={single_stats['mean_s']:.4f}")
    print(f"batch_mean_s={batch_stats['mean_s']:.4f}")
    if prompt_reuse_stats:
        print(f"voice_clone_prompt_reuse_mean_s={prompt_reuse_stats['mean_s']:.4f}")

    if temp_ref:
        Path(temp_ref).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
