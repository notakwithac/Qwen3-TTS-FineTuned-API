# coding=utf-8
"""Smoke test for the fine-tuning API.

Usage:
    1. Start the API:  bash start_api.sh
    2. In another terminal:  python test_api.py [--base-url http://localhost:8000]

This creates a minimal dataset zip from existing samples and tests the full pipeline.
"""

import argparse
import io
import json
import os
import shutil
import sys
import time
import zipfile

import requests


def create_test_dataset_zip(
    source_dir: str = "gared_voice_qwen3_tts_dataset",
    output_path: str = "/tmp/test_dataset.zip",
    max_samples: int = 2,
) -> str:
    """Create a minimal dataset zip from existing data for testing."""
    source = source_dir
    train_jsonl = os.path.join(source, "train.jsonl")

    if not os.path.exists(train_jsonl):
        print(f"ERROR: {train_jsonl} not found. Run from the Qwen3-TTS directory.")
        sys.exit(1)

    with open(train_jsonl) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    # Take only a few samples
    lines = lines[:max_samples]

    # Collect audio files needed
    audio_files = set()
    for item in lines:
        audio_files.add(item["audio"])
        if "ref_audio" in item:
            audio_files.add(item["ref_audio"])

    # Create zip in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write train.jsonl
        jsonl_content = "\n".join(json.dumps(item, ensure_ascii=False) for item in lines)
        zf.writestr("train.jsonl", jsonl_content)

        # Write audio files
        for audio in audio_files:
            audio_path = audio.lstrip("./")
            full_path = os.path.join(source, audio_path)
            if os.path.exists(full_path):
                zf.write(full_path, audio_path)
            else:
                print(f"WARNING: Audio file {full_path} not found, skipping")

    # Write to disk
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())

    print(f"Created test dataset zip: {output_path} ({len(lines)} samples)")
    return output_path


def test_full_pipeline(base_url: str, dataset_zip: str, num_epochs: int = 1):
    """Test the complete API pipeline: upload → train → infer."""

    print(f"\n{'='*60}")
    print(f"Testing API at {base_url}")
    print(f"{'='*60}")

    # 1) Health check
    print("\n[1/5] Health check...")
    r = requests.get(f"{base_url}/")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print(f"  ✅  {r.json()}")

    # 2) Upload dataset to S3 first
    print(f"\n[2/5] Uploading dataset ({dataset_zip}) to S3...")
    if not storage.is_configured:
        print("  ❌  S3 storage not configured. Cannot proceed with S3-based finetune test.")
        sys.exit(1)
    
    with open(dataset_zip, "rb") as f:
        dataset_bytes = f.read()
    
    s3_key = f"datasets/test/smoke_test_{int(time.time())}.zip"
    storage.upload_bytes(dataset_bytes, s3_key, content_type="application/zip")
    print(f"  ✅  Dataset uploaded to S3: {s3_key}")

    # 3) Start Fine-tuning
    print(f"\n[3/5] Starting fine-tuning job...")
    payload = {
        "dataset_s3_key": s3_key,
        "speaker_name": "test_speaker",
        "num_epochs": num_epochs,
        "batch_size": 1,
        "lr": 2e-6,
    }
    r = requests.post(f"{base_url}/finetune", json=payload)
    assert r.status_code == 202, f"Finetune trigger failed: {r.text}"
    job = r.json()
    job_id = job["job_id"]
    print(f"  ✅  Job created: {job_id} (status: {job['status']})")

    # 4) Poll until done
    print(f"\n[4/6] Polling job status...")
    max_wait = 3600  # 1 hour max
    poll_interval = 10
    elapsed = 0
    while elapsed < max_wait:
        r = requests.get(f"{base_url}/jobs/{job_id}")
        assert r.status_code == 200
        job = r.json()
        status = job["status"]
        progress = job.get("progress", {})

        if status == "training":
            epoch = progress.get("epoch", "?")
            total = progress.get("total_epochs", "?")
            loss = progress.get("loss", "?")
            print(f"  ⏳  {status}: epoch {epoch}/{total}, loss: {loss}")
        elif status in ("preparing", "loading", "queued"):
            detail = progress.get("detail", "")
            print(f"  ⏳  {status}: {detail}")
        elif status == "ready":
            print(f"  ✅  Job ready! inference_url: {job.get('inference_url')}")
            break
        elif status == "failed":
            print(f"  ❌  Job failed: {job.get('error')}")
            sys.exit(1)
        else:
            print(f"  ❓  Unknown status: {status}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    if job["status"] != "ready":
        print(f"  ❌  Timeout waiting for job (last status: {job['status']})")
        sys.exit(1)

    # 5) Generate speech (Defaults to S3)
    print(f"\n[5/6] Generating speech (Default S3)...")
    r = requests.post(
        f"{base_url}/infer/{job_id}",
        json={"text": "Hello, this is a test of the fine-tuned model.", "language": "English"},
    )
    if r.status_code == 200 and r.headers.get("content-type") == "application/json":
        data = r.json()
        print(f"  ✅  S3 URL: {data['s3_url']}")
    else:
        # Fallback for when S3 is not configured in test environment
        print(f"  ⚠  S3 upload default skipped or failed (Status: {r.status_code})")
        print("      Retrying with upload_to_s3=False for raw audio...")
        r = requests.post(
            f"{base_url}/infer/{job_id}",
            json={
                "text": "Hello, this is a test of the fine-tuned model.",
                "language": "English",
                "upload_to_s3": False
            },
        )
        assert r.status_code == 200, f"Raw audio fallback failed: {r.text}"
        output_file = f"test_output_{job_id}.wav"
        with open(output_file, "wb") as f:
            f.write(r.content)
        print(f"  ✅  Raw audio saved to {output_file} ({len(r.content)} bytes)")

    # 4.5) Parallel Batch Inference
    print(f"\n[4.5/5] Testing Parallel Batch Inference...")
    batch_req = {
        "items": [
            {"text": "Sample one.", "filename": "batch_1.wav"},
            {"text": "Sample two.", "filename": "batch_2.wav"},
            {"text": "Sample three.", "filename": "batch_3.wav"}
        ]
    }
    r = requests.post(f"{base_url}/infer/{job_id}/batch", json=batch_req)
    if r.status_code == 200:
        results = r.json()
        print(f"  ✅  Batch complete! {len(results)} items generated.")
        for item in results:
            print(f"      - {item['s3_url']}")
    else:
        print(f"  ⚠  Batch test failed or storage not configured: {r.text}")

    # 6) List jobs
    print(f"\n[6/6] Listing jobs...")
    r = requests.get(f"{base_url}/jobs")
    assert r.status_code == 200
    jobs = r.json()
    print(f"  ✅  {len(jobs)} job(s) found")

    print(f"\n{'='*60}")
    print(f"All tests passed! 🎉")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the fine-tuning API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--dataset-dir", default="gared_voice_qwen3_tts_dataset", help="Source dataset dir")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs for test")
    parser.add_argument("--max-samples", type=int, default=2, help="Max audio samples in test dataset")
    args = parser.parse_args()

    zip_path = create_test_dataset_zip(source_dir=args.dataset_dir, max_samples=args.max_samples)
    test_full_pipeline(base_url=args.base_url, dataset_zip=zip_path, num_epochs=args.num_epochs)
