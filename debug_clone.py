import requests

base_url = "http://127.0.0.1:8000"
ref_audio_url = 'https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'
ref_text = "Beavers are second only to humans in their ability to manipulate their environment."

payload = {
    "text": "This is a single voice clone request using the new endpoint.",
    "ref_audio_url": ref_audio_url,
    "ref_text": ref_text,
    "upload_to_s3": False
}

try:
    print(f"Sending request to {base_url}/voice-clone...")
    response = requests.post(f"{base_url}/voice-clone", json=payload, timeout=300)
    print(f"Status: {response.status_code}")
    print(f"Response Body: {response.text[:1000]}")
except Exception as e:
    print(f"Error: {e}")
