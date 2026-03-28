import requests
import json
import time

base_url = "http://127.0.0.1:8000"
ref_audio_url = 'https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'
ref_text = "Beavers are second only to humans in their ability to manipulate their environment."

def test_clone_single():
    print("\n--- Testing Single /voice-clone ---")
    payload = {
        "text": "This is a single voice clone request using the new endpoint.",
        "ref_audio_url": ref_audio_url,
        "ref_text": ref_text,
        "upload_to_s3": False
    }
    r = requests.post(f"{base_url}/voice-clone", json=payload, timeout=300)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        content_type = r.headers.get("Content-Type", "")
        if "audio/wav" in content_type:
            print(f"Success! Received raw WAV audio ({len(r.content)} bytes).")
        else:
            data = r.json()
            print(f"Success! Received JSON response.")
            print(json.dumps(data, indent=2))
    else:
        print(f"Error: {r.text}")

def test_clone_batch():
    print("\n--- Testing Batch /voice-clone/batch ---")
    payload = {
        "ref_audio_url": ref_audio_url,
        "ref_text": ref_text,
        "items": [
            {"text": "Sentence one for the batch clone test."},
            {"text": "Sentence two for the batch clone test."}
        ],
        "upload_to_s3": False
    }
    r = requests.post(f"{base_url}/voice-clone/batch", json=payload, timeout=300)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Success! Generated {len(data)} items.")
        for i, item in enumerate(data):
            print(f"  Item {i}: {item.get('text')} -> {len(item.get('presigned_url', ''))} chars in base64")
    else:
        print(f"Error: {r.text}")

if __name__ == "__main__":
    test_clone_single()
    test_clone_batch()
