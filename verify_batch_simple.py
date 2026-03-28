import requests
import json

url = "http://127.0.0.1:8000/voice-design/batch"
payload = {
    "items": [
        {
            "text": "Hello world, testing the new batching system.",
            "instruct": "A warm male voice, middle-aged, calm.",
            "language": "English",
            "character_name": "TestBatcher"
        }
    ],
    "upload_to_s3": False,
    "overwrite": True
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, timeout=300)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
