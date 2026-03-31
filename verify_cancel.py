import requests
import time
import os

BASE_URL = "http://localhost:8000"

def test_cancellation_flow():
    print("1. Triggering termination...")
    resp = requests.post(f"{BASE_URL}/gpu/terminate")
    print(f"   Response: {resp.status_code} - {resp.json()}")
    
    print("\n2. Verifying restrictive state (POST /gpu/status should still work, but others might stay blocked)...")
    # Middleware blocks POST requests when draining.
    # Let's try a dummy POST if possible, or just check /gpu/unload (POST)
    resp = requests.post(f"{BASE_URL}/gpu/unload")
    if resp.status_code == 503:
        print("   ✅ SUCCESS: POST /gpu/unload rejected with 503 (Draining)")
    else:
        print(f"   ❌ FAILURE: POST /gpu/unload returned {resp.status_code}")

    print("\n3. Verifying signal file existence...")
    # This script assumes it's running in the same dir as the server for this check
    if os.path.exists("terminate_signal.tmp"):
        print("   ✅ SUCCESS: terminate_signal.tmp exists")
    else:
        print("   ❌ FAILURE: terminate_signal.tmp NOT found")

    print("\n4. Triggering cancellation...")
    resp = requests.post(f"{BASE_URL}/gpu/cancel-terminate")
    print(f"   Response: {resp.status_code} - {resp.json()}")

    print("\n5. Verifying resumed state...")
    resp = requests.post(f"{BASE_URL}/gpu/unload")
    if resp.status_code != 503:
        print(f"   ✅ SUCCESS: POST /gpu/unload accepted (returned {resp.status_code})")
    else:
        print("   ❌ FAILURE: POST /gpu/unload still rejected with 503")

    print("\n6. Verifying signal file deletion...")
    if not os.path.exists("terminate_signal.tmp"):
        print("   ✅ SUCCESS: terminate_signal.tmp deleted")
    else:
        print("   ❌ FAILURE: terminate_signal.tmp still exists")

if __name__ == "__main__":
    try:
        test_cancellation_flow()
    except Exception as e:
        print(f"Test failed: {e}")
