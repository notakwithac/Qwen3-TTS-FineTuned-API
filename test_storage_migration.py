import os
import sys
from dotenv import load_dotenv
load_dotenv()
from storage import StorageClient

def test_storage():
    client = StorageClient()
    if not client.is_configured:
        print("Storage not configured Skip test.")
        return

    test_data = b"hello migration"
    source_key = "test/migration_source.txt"
    dest_key = "test/migration_dest.txt"

    print(f"Uploading to {source_key}...")
    client.upload_bytes(test_data, source_key)
    
    if client.object_exists(source_key):
        print("Source exists.")
    else:
        print("Failed to upload source.")
        return

    print(f"Moving {source_key} to {dest_key}...")
    client.move_object(source_key, dest_key)

    if client.object_exists(dest_key):
        print("Dest exists.")
    else:
        print("Failed to move to dest.")

    if not client.object_exists(source_key):
        print("Source no longer exists (Correct).")
    else:
        print("Source still exists (Error).")

    # Cleanup
    print("Cleaning up...")
    client.delete_object(dest_key)

if __name__ == "__main__":
    test_storage()
