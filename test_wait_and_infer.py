"""Wait for training to finish, then generate an excited voice sample."""
import requests
import time
import sys

BASE = "http://localhost:8000"
JOB_ID = "c507d8b82ec3"
MAX_WAIT = 30 * 60  # 30 minutes

print(f"=== Monitoring Job {JOB_ID} (max {MAX_WAIT//60} min) ===")
start = time.time()

while True:
    elapsed = time.time() - start
    if elapsed > MAX_WAIT:
        print(f"\nTIMEOUT: Training did not finish in {MAX_WAIT//60} minutes.")
        sys.exit(1)

    r = requests.get(f"{BASE}/jobs/{JOB_ID}")
    data = r.json()
    status = data.get("status")
    progress = data.get("progress", {})
    error = data.get("error")
    epoch = progress.get("epoch", "?")
    total = progress.get("total_epochs", "?")
    step = progress.get("step", "?")
    loss = progress.get("loss")

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"  [{mins:02d}:{secs:02d}] Status={status} | Epoch {epoch}/{total} | Step {step} | Loss={loss}")

    if status == "ready":
        print(f"\n✅ TRAINING COMPLETE!")
        print(f"   Checkpoint: {data.get('checkpoint_path')}")
        break
    elif status in ("failed", "cancelled"):
        print(f"\n❌ Training {status}: {error}")
        sys.exit(1)

    time.sleep(30)

# --- Generate excited voice sample ---
print("\n=== Generating Excited Voice Sample ===")
checkpoint = data.get("checkpoint_path")
infer_url = f"{BASE}/infer/{JOB_ID}"

r = requests.post(infer_url, json={
    "text": "Oh my god, training is COMPLETE! This is absolutely INCREDIBLE! I can't believe how amazing this sounds! YES! We did it! This is the best day EVER!",
    "language": "English",
    "instruct": "Speak with extreme excitement, screaming with joy, high energy and enthusiasm. Voice should be bursting with emotion.",
    "upload_to_s3": False,
}, timeout=300)

print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type', 'N/A')}")
print(f"Size: {len(r.content)} bytes")

if r.status_code == 200 and "audio" in r.headers.get("content-type", ""):
    with open("voice_sample_excited.wav", "wb") as f:
        f.write(r.content)
    print("\n🎉 SUCCESS: Saved to voice_sample_excited.wav")
else:
    print(f"\nFAILED: {r.text}")
    sys.exit(1)
