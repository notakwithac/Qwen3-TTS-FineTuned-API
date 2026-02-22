import requests
import time
import zipfile
import io
import os
import json

BASE_URL = "http://localhost:8000"

def verify_flow():
    print("--- 1. VOICE DESIGN ---")
    design_payload = {
        "text": "Hello, I am a custom voice designed specifically for testing purposes. I sound professional and clear.",
        "instruct": "Professional male voice with a clear, calm tone.",
        "language": "English"
    }
    
    resp = requests.post(f"{BASE_URL}/gpu/design", json=design_payload)
    if resp.status_code != 200:
        print(f"FAILED: Voice Design - {resp.text}")
        return
    
    audio_data = resp.content
    print(f"SUCCESS: Designed voice audio received ({len(audio_data)} bytes)")

    print("\n--- 2. PREPARE DATASET ZIP ---")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("data/sample.wav", audio_data)
        zf.writestr("train.jsonl", json.dumps({
            "audio": "data/sample.wav",
            "text": "Hello, I am a custom voice designed specifically for testing purposes. I sound professional and clear.",
            "ref_audio": "data/sample.wav"
        }) + "\n")
    
    zip_bytes = zip_buffer.getvalue()
    print(f"SUCCESS: Dataset zip created locally.")

    print("\n--- 3. NOTE ON FINE-TUNING ---")
    print("To trigger fine-tuning, the API currently requires the zip to be in S3.")
    print("Please ensure your .env is filled with S3 credentials.")
    print("\nIf S3 is configured, you would run:")
    print("1. Upload zip_bytes to S3.")
    print("2. POST /finetune with the S3 key.")
    
    # Check if storage is configured in the API
    storage_resp = requests.get(f"{BASE_URL}/storage/config")
    if storage_resp.status_code == 200:
        data = storage_resp.json()
        if not data.get("configured"):
            print("\n!!! WARNING: S3 Storage is NOT configured in the API yet. !!!")
            print("Please edit your .env file and restart start_api.bat.")
        else:
            print(f"\nS3 is READY: Using bucket '{data.get('bucket')}'")

if __name__ == "__main__":
    verify_flow()
