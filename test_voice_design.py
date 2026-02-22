"""Quick verification: Voice Design -> WAV output."""
import requests
import sys

BASE = "http://localhost:8000"

print("=== Voice Design Test (English) ===")
r = requests.post(f"{BASE}/voice-design", json={
    "text": "This is a test of the voice design system. The quick brown fox jumps over the lazy dog.",
    "instruct": "A professional male voice, clear and confident, mid-range pitch.",
    "language": "English",
    "upload_to_s3": False,
}, timeout=300)

print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type', 'N/A')}")
print(f"Size: {len(r.content)} bytes")

if r.status_code == 200 and "audio" in r.headers.get("content-type", ""):
    with open("test_design.wav", "wb") as f:
        f.write(r.content)
    print("SUCCESS: Saved to test_design.wav")
else:
    print(f"FAILED: {r.text}")
    sys.exit(1)
