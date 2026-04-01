import requests
import time
import threading
import logging

def stream_logs():
    url = "http://localhost:8000/logs/stream"
    print(f"Connecting to {url}...")
    try:
        # Use stream=True to process the response line by line
        with requests.get(url, stream=True, timeout=30) as response:
            if response.status_code != 200:
                print(f"Failed to connect: {response.status_code}")
                return
            
            print("Connected! Listening for logs...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        print(f"RECVD: {decoded_line[6:]}")
    except Exception as e:
        print(f"Error streaming logs: {e}")

if __name__ == "__main__":
    # This script assumes the server is already running on localhost:8000.
    # Since I can't easily run a persistent server and client in this environment 
    # without complex backgrounding, I'll provide this as a tool for the user.
    stream_logs()
