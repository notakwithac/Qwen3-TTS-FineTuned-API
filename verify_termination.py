import os
import time
import subprocess
import signal
from massed_compute_client import MassedComputeClient
from dotenv import load_dotenv

load_dotenv(override=True)

# Mock signal file and environment
SIGNAL_FILE = "terminate_signal.tmp"
ENV_VAR = "GPU_INSTANCE_UUID"
DUMMY_UUID = "test-uuid-12345"

# Monkeypatch authentication for testing
from massed_compute_client import MassedComputeClient
MassedComputeClient.authenticate = lambda self: True

def test_watchdog_signal_detection():
    print(f"Setting up test environment with {ENV_VAR}={DUMMY_UUID}")
    os.environ[ENV_VAR] = DUMMY_UUID
    
    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)
        
    print("Creating termination signal...")
    with open(SIGNAL_FILE, "w") as f:
        f.write(str(time.time()))
        
    print("Running watchdog in dry-run mode for a single poll...")
    # We run the watchdog and look for the specific log message
    try:
        # Note: We need to pass the environment variable to the subprocess
        env = os.environ.copy()
        env[ENV_VAR] = DUMMY_UUID
        
        # Run for a few seconds and check output
        process = subprocess.Popen(
            ["python", "gpu_idle_watchdog.py", "--dry-run", "--skip-auth"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        
        # Wait a bit for it to poll
        time.sleep(5)
        process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
        
        print("\n--- Watchdog Output ---")
        print(output)
        print("--- End Output ---\n")
        
        if "Immediate termination requested via signal file" in output:
            print("✅ SUCCESS: Watchdog detected the signal file.")
        else:
            print("❌ FAILURE: Watchdog did NOT detect the signal file.")
            
        if f"DRY RUN: Would terminate instance {DUMMY_UUID}" in output:
            print(f"✅ SUCCESS: Watchdog used the UUID from environment ({DUMMY_UUID}).")
        else:
            print("❌ FAILURE: Watchdog did NOT use the correct UUID.")

    finally:
        if os.path.exists(SIGNAL_FILE):
            os.remove(SIGNAL_FILE)

if __name__ == "__main__":
    test_watchdog_signal_detection()
