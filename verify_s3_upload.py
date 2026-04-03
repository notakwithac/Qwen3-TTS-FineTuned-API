import os
import glob
from datetime import datetime, timezone
import urllib.request
from gpu_idle_watchdog import upload_final_artifacts
from storage import storage

def verify():
    print("Testing upload_final_artifacts...")
    instance_uuid = "test-uuid-1234"
    
    # Ensure logs/ and metrics/ exist for testing
    os.makedirs("logs", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    with open("logs/test.log", "w") as f:
        f.write("test log")
    with open("metrics/resource_metrics.jsonl", "w") as f:
        f.write("test metrics")
        
    print(f"Current docs: {glob.glob('*DOCS.md')}")
    print(f"Current logs: {glob.glob('logs/*.log')}")
    
    # We won't actually call the real upload unless we want to test S3 connectivity.
    # If storage.is_configured is True, it will try to upload.
    if storage.is_configured:
        print("Storage is configured. Attempting upload...")
        upload_final_artifacts(instance_uuid)
        print("Upload complete (check S3 console for results).")
    else:
        print("Storage NOT configured. Test aborted.")

if __name__ == "__main__":
    verify()
