import asyncio
import aiohttp
import time

API_URL = "http://localhost:8000/voice-design"

async def fetch_voice(session, req_id, text, instruct):
    print(f"[{req_id}] Sending request...")
    start_time = time.time()
    
    payload = {
        "text": text,
        "instruct": instruct,
        "upload_to_s3": False  # Just get raw audio to test generation speed
    }
    
    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                audio_data = await response.read()
                elapsed = time.time() - start_time
                print(f"[{req_id}] Success! Received {len(audio_data)} bytes in {elapsed:.2f} seconds.")
                return req_id, audio_data, elapsed
            else:
                err = await response.text()
                print(f"[{req_id}] Failed with {response.status}: {err}")
                return req_id, None, 0
    except Exception as e:
        print(f"[{req_id}] Error: {str(e)}")
        return req_id, None, 0

async def main():
    print("Starting parallel Voice Design test...")
    
    requests = [
        {"id": 1, "text": "Hello, this is the first concurrent request.", "instruct": "A deep male voice."},
        {"id": 2, "text": "And this is the second simultaneous request.", "instruct": "A high-pitched female voice."},
        {"id": 3, "text": "Here goes the third one trying to batch.", "instruct": "An old man speaking slowly."},
        {"id": 4, "text": "Finally, the fourth request in the batch.", "instruct": "An energetic young boy."},
    ]
    
    overall_start = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for req in requests:
            tasks.append(fetch_voice(session, req["id"], req["text"], req["instruct"]))
            
        results = await asyncio.gather(*tasks)
        
    overall_elapsed = time.time() - overall_start
    print(f"\nAll 4 requests completed in {overall_elapsed:.2f} seconds total.")
    
    # Save the files to verify they are distinct
    for req_id, audio_data, _ in results:
        if audio_data:
            filename = f"batch_test_output_{req_id}.wav"
            with open(filename, "wb") as f:
                f.write(audio_data)
            print(f"Saved {filename}")

if __name__ == "__main__":
    asyncio.run(main())
