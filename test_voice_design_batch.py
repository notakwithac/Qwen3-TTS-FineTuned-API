"""Integration test for POST /voice-design/batch — concurrent voice design.

Usage:
    1. Start the API:  bash start_api.sh  (or  python -m uvicorn api_server:app)
    2. Run tests:      python test_voice_design_batch.py [--base-url http://localhost:8000]
"""

import argparse
import sys
import time

import requests


CHARACTERS = [
    {
        "text": "The storm raged outside the castle walls, but inside all was calm.",
        "instruct": "A deep, commanding male voice. Middle-aged, authoritative, with measured pacing.",
        "character_name": "King",
    },
    {
        "text": "The storm raged outside the castle walls, but inside all was calm.",
        "instruct": "A young female voice, bright and energetic, with a hint of mischief.",
        "character_name": "Princess",
    },
    {
        "text": "The storm raged outside the castle walls, but inside all was calm.",
        "instruct": "A gravelly old man's voice, slow and wise, warm like a fireside storyteller.",
        "character_name": "Wizard",
    },
]


def test_batch_endpoint(base_url: str, upload_to_s3: bool = False):
    """Test the /voice-design/batch endpoint."""

    print(f"\n{'='*60}")
    print(f"Voice Design Batch Test — {base_url}")
    print(f"{'='*60}")

    # --- 1. Health check ---
    print("\n[1/4] Health check...")
    r = requests.get(f"{base_url}/", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print(f"  ✅  Server is up")

    # --- 2. Batch voice design ---
    print(f"\n[2/4] Submitting batch of {len(CHARACTERS)} voice designs...")
    payload = {
        "items": CHARACTERS,
        "upload_to_s3": upload_to_s3,
        "overwrite": True,
    }

    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/voice-design/batch", json=payload, timeout=600)
    batch_elapsed = time.perf_counter() - t0

    assert r.status_code == 200, f"Batch request failed ({r.status_code}): {r.text}"
    data = r.json()

    print(f"  ✅  Batch completed in {batch_elapsed:.1f}s")
    print(f"      Total: {data['total']}, Succeeded: {data['succeeded']}, "
          f"Skipped: {data['skipped']}, Failed: {data['failed']}")

    for result in data["results"]:
        status_icon = "✅" if result["status"] == "success" else "⏭️" if result["status"] == "skipped" else "❌"
        name = result.get("character_name", f"item-{result['index']}")
        if result["status"] == "failed":
            print(f"      {status_icon}  [{name}] FAILED: {result.get('error')}")
        elif upload_to_s3:
            print(f"      {status_icon}  [{name}] → {result.get('s3_key', 'n/a')}")
        else:
            audio_len = len(result.get("audio_base64", ""))
            print(f"      {status_icon}  [{name}] → {audio_len} base64 chars")

    assert data["failed"] == 0, f"{data['failed']} items failed!"

    # --- 3. Compare with sequential timing ---
    print(f"\n[3/4] Sequential comparison (single calls)...")
    t0 = time.perf_counter()
    for char in CHARACTERS:
        r = requests.post(f"{base_url}/voice-design", json={
            "text": char["text"],
            "instruct": char["instruct"],
            "language": "English",
            "character_name": char["character_name"],
            "upload_to_s3": upload_to_s3,
            "overwrite": True,
        }, timeout=300)
        assert r.status_code == 200, f"Single request failed: {r.text}"
    seq_elapsed = time.perf_counter() - t0

    speedup = seq_elapsed / batch_elapsed if batch_elapsed > 0 else float("inf")
    print(f"  ⏱  Sequential: {seq_elapsed:.1f}s  vs  Batch: {batch_elapsed:.1f}s")
    print(f"  ⚡  Speedup: {speedup:.2f}x")

    # --- 4. Validate limit enforcement ---
    print(f"\n[4/4] Validating max items limit...")
    oversized = {"items": CHARACTERS * 10, "upload_to_s3": False}  # 30 items
    r = requests.post(f"{base_url}/voice-design/batch", json=oversized, timeout=10)
    if r.status_code == 400:
        print(f"  ✅  Correctly rejected oversized batch: {r.json()['detail']}")
    else:
        print(f"  ⚠  Unexpected status {r.status_code} for oversized batch")

    print(f"\n{'='*60}")
    print(f"All tests passed! 🎉")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test voice design batch endpoint")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--s3", action="store_true", help="Enable S3 upload in tests")
    args = parser.parse_args()

    test_batch_endpoint(base_url=args.base_url, upload_to_s3=args.s3)
