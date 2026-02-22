# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for audio tokenization")
    args = parser.parse_args()

    input_jsonl_path = Path(args.input_jsonl)
    input_base_dir = input_jsonl_path.parent

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    total_lines = open(args.input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]

    final_lines = []
    batch_lines = []
    batch_audios = []
    for line in total_lines:
        audio_path = Path(line["audio"])
        if not audio_path.is_absolute():
            audio_path = input_base_dir / audio_path
        line["audio"] = str(audio_path)

        ref_audio = line.get("ref_audio")
        if ref_audio:
            ref_audio_path = Path(ref_audio)
            if not ref_audio_path.is_absolute():
                ref_audio_path = input_base_dir / ref_audio_path
            line["ref_audio"] = str(ref_audio_path)

        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= args.batch_size:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line['audio_codes'] = code.cpu().tolist()
                final_lines.append(line)
            batch_lines.clear()
            batch_audios.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, line in zip(enc_res.audio_codes, batch_lines):
            line['audio_codes'] = code.cpu().tolist()
            final_lines.append(line)
        batch_lines.clear()
        batch_audios.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

    with open(args.output_jsonl, 'w') as f:
        for line in final_lines:
            f.writelines(line + '\n')

def prepare_programmatic(
    input_jsonl: str,
    output_jsonl: str,
    device: str = "cuda:0",
    tokenizer_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    batch_size: int = 1,
    on_progress=None,
):
    """Run data preparation programmatically (no argparse).

    Args:
        input_jsonl: Path to input JSONL file.
        output_jsonl: Path to write output JSONL with audio codes.
        device: CUDA device string.
        tokenizer_model_path: HuggingFace model path for tokenizer.
        batch_size: Batch size for audio tokenization.
        on_progress: Optional callback(current, total) for progress.
    """
    input_jsonl_path = Path(input_jsonl)
    input_base_dir = input_jsonl_path.parent

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_path,
        device_map=device,
    )

    total_lines = open(input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]
    total_count = len(total_lines)

    final_lines = []
    batch_lines = []
    batch_audios = []
    processed = 0
    for line in total_lines:
        audio_path = Path(line["audio"])
        if not audio_path.is_absolute():
            audio_path = input_base_dir / audio_path
        line["audio"] = str(audio_path)

        ref_audio = line.get("ref_audio")
        if ref_audio:
            ref_audio_path = Path(ref_audio)
            if not ref_audio_path.is_absolute():
                ref_audio_path = input_base_dir / ref_audio_path
            line["ref_audio"] = str(ref_audio_path)

        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= batch_size:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, bl in zip(enc_res.audio_codes, batch_lines):
                bl['audio_codes'] = code.cpu().tolist()
                final_lines.append(bl)
                processed += 1
            batch_lines.clear()
            batch_audios.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if on_progress:
                on_progress(processed, total_count)

    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, bl in zip(enc_res.audio_codes, batch_lines):
            bl['audio_codes'] = code.cpu().tolist()
            final_lines.append(bl)
            processed += 1
        batch_lines.clear()
        batch_audios.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(processed, total_count)

    final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

    with open(output_jsonl, 'w') as f:
        for line in final_lines:
            f.writelines(line + '\n')


if __name__ == "__main__":
    main()
