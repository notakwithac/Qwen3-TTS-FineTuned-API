"""Trigger fine-tuning from S3 dataset and monitor progress."""
import requests
import time
import sys

BASE = "http://localhost:8000"

print("=== Triggering Fine-Tune Job ===")
r = requests.post(f"{BASE}/finetune", json={
    "dataset_s3_key": "datasets/a1b2c3d4e5/trial.zip",
    "speaker_name": "TrialVoice",
    "num_epochs": 10,
    "batch_size": 1,
}, timeout=30)

print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

if r.status_code != 200:
    print("FAILED to start fine-tuning job.")
    sys.exit(1)

job_id = r.json().get("job_id")
print(f"\nJob ID: {job_id}")
print("Polling for progress...")

while True:
    time.sleep(10)
    status_r = requests.get(f"{BASE}/jobs/{job_id}")
    data = status_r.json()
    status = data.get("status")
    progress = data.get("progress", {})
    error = data.get("error")
    
    print(f"  [{status}] {progress}")
    
    if status in ("ready", "failed", "cancelled"):
        if error:
            print(f"  ERROR: {error}")
        break

print(f"\nFinal status: {data.get('status')}")
if data.get("checkpoint_path"):
    print(f"Checkpoint: {data['checkpoint_path']}")
