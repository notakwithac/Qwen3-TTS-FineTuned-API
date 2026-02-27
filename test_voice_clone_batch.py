import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def test_voice_clone_batch():
    # Public gradio sample audio used in testing
    ref_audio_url = 'https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'
    ref_text = "Beavers are second only to humans in their ability to manipulate their environment."

    payload = {
        "ref_audio_url": ref_audio_url,
        "ref_text": ref_text,
        "items": [
            {
                "text": "This is the first sentence generated in parallel using the zero-shot voice cloning model.",
                "filename": "vc_test_1.wav"
            },
            {
                "text": "And here is the second sentence, also generated concurrently with the first one.",
                "filename": "vc_test_2.wav"
            }
        ],
        "language": "English",
        "use_xvec": False,
        "upload_to_s3": False, # Test locally without S3 first, or you can test with S3 if configured
        "overwrite": True
    }

    print(f"Sending batch request with {len(payload['items'])} items...")
    start_time = time.time()
    
    response = requests.post(f"{BASE_URL}/voice-clone/batch", json=payload)
    
    if response.status_code == 200:
        results = response.json()
        print(f"Success! Generated {len(results)} files in {time.time() - start_time:.2f} seconds.")
        for i, res in enumerate(results):
            # If upload_to_s3 is False, presigned_url contains the base64 audio
            url = res.get('presigned_url', 'No URL')
            is_base64 = url.startswith('data:audio/wav;base64,')
            print(f"Item {i+1}:")
            print(f"  Filename: {res.get('s3_key')}")
            print(f"  URL: {'[Base64 Audio]' if is_base64 else url}")
            print(f"  Sample Rate: {res.get('sample_rate')}")
            
            if is_base64:
                import base64
                audio_data = base64.b64decode(url.split(',')[1])
                with open(res.get('s3_key'), 'wb') as f:
                    f.write(audio_data)
                print(f"  -> Saved to {res.get('s3_key')} locally.")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_voice_clone_batch()
