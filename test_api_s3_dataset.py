# coding=utf-8
"""Test for S3-based fine-tuning dataset submission.

Usage:
    1. Start the API with S3 configured:  bash start_api.sh
    2. Run this test:  python test_api_s3_dataset.py
"""

import os
import json
import time
import requests
import zipfile
import io
import sys
from storage import storage

def create_mock_dataset_zip():
    """Create a minimal mock dataset for testing."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # Minimal train.jsonl
        train_data = [
            {"audio": "./data/test.wav", "text": "Test audio.", "ref_audio": "./data/test.wav"}
        ]
        jsonl_content = "\n".join(json.dumps(d) for d in train_data)
        zf.writestr("train.jsonl", jsonl_content)
        
        # Create a tiny dummy wav (not valid but enough for parsing)
        zf.writestr("data/test.wav", b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    
    return buf.getvalue()

def test_s3_finetune(base_url="http://localhost:8000"):
    print(f"Testing S3-based finetune at {base_url}...")
    
    if not storage.is_configured:
        print("ERROR: S3 storage (E2E_ACCESS_KEY/SECRET) not configured in environment.")
        return

    # 1. Create and Upload Dataset
    print("\n[1/3] Creating and uploading mock dataset to S3...")
    dataset_bytes = create_mock_dataset_zip()
    s3_key = f"datasets/test/mock_dataset_{int(time.time())}.zip"
    
    try:
        url = storage.upload_bytes(dataset_bytes, s3_key, content_type="application/zip")
        print(f"  ✅ Uploaded to: {url}")
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        return

    # 2. Trigger Finetune
    print("\n[2/3] Triggering /finetune via JSON...")
    payload = {
        "dataset_s3_key": s3_key,
        "speaker_name": "TestVoice",
        "num_epochs": 1,
        "batch_size": 1,
        "lr": 1e-6
    }
    
    r = requests.post(f"{base_url}/finetune", json=payload)
    if r.status_code != 202:
        print(f"  ❌ Failed: {r.status_code} - {r.text}")
        return
    
    job = r.json()
    job_id = job["job_id"]
    print(f"  ✅ Job created: {job_id}")

    # 3. Poll Status
    print(f"\n[3/3] Polling status for {job_id}...")
    for _ in range(10):
        r = requests.get(f"{base_url}/jobs/{job_id}")
        if r.status_code == 200:
            status = r.json()["status"]
            print(f"  Status: {status}")
            if status in ["preparing", "training", "ready"]:
                print("  ✅ Flow verified (job moved beyond 'queued' or 'prepared').")
                return
            if status == "failed":
                print(f"  ❌ Job failed: {r.json().get('error')}")
                return
        else:
            print(f"  ❌ Job poll failed: {r.status_code}")
            return
        time.sleep(2)
    
    print("  ⚠️ Polling timed out, but job was successfully created.")

if __name__ == "__main__":
    test_s3_finetune()
