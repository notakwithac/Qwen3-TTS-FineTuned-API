import os
import asyncio
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv
load_dotenv()

# We need to mock the pipeline and storage to test the logic in api_server.py
# But since we want to test if it actually MOVES files on S3, we'll use the real storage
# but mocked job/req.

from storage import storage

async def test_migration_logic():
    print("Testing migration logic...")
    
    # Setup test data
    book_id = "test_book"
    chapter_id = "test_chapter"
    char_id = "test_char"
    job_id = "test_job"
    filename = "test_migration_api.wav"
    test_content = b"migration test content"
    
    # 1. Test migration from job path to proper path
    job_key = f"audio/{job_id}/{filename}"
    proper_key = f"audio/segments/{book_id}/{chapter_id}/{char_id}/{filename}"
    
    print(f"Uploading to legacy job path: {job_key}")
    storage.upload_bytes(test_content, job_key)
    
    # Mocking req and job as they appear in api_server.py
    class MockReq:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.upload_to_s3 = True
            self.overwrite = False
            self.s3_filename = filename
            self.book_id = book_id
            self.chapter_id = chapter_id
            self.character_id = None # Should come from job

    class MockJob:
        def __init__(self):
            self.character_id = char_id
            self.job_id = job_id

    req = MockReq()
    job = MockJob()

    # Re-implementing the logic from infer endpoint to verify it works
    s3_key_found = None
    if req.upload_to_s3 and not req.overwrite and req.s3_filename and storage.is_configured:
        current_char_id = req.character_id or job.character_id
        
        # 1. Proper path (character-specific)
        if req.book_id and req.chapter_id and current_char_id:
            pk = f"audio/segments/{req.book_id}/{req.chapter_id}/{current_char_id}/{req.s3_filename}"
            if storage.object_exists(pk):
                print(f"Found at proper path: {pk}")
                s3_key_found = pk
        
        # 2. Legacy segment path (no character_id)
        if not s3_key_found and req.book_id and req.chapter_id:
            lsk = f"audio/segments/{req.book_id}/{req.chapter_id}/{req.s3_filename}"
            if storage.object_exists(lsk):
                if current_char_id:
                    pk = f"audio/segments/{req.book_id}/{req.chapter_id}/{current_char_id}/{req.s3_filename}"
                    print(f"Migrating {lsk} -> {pk}")
                    storage.move_object(lsk, pk)
                    s3_key_found = pk
                else:
                    s3_key_found = lsk
        
        # 3. Fallback path (job-level)
        if not s3_key_found:
            jk = f"audio/{job_id}/{req.s3_filename}"
            if storage.object_exists(jk):
                if req.book_id and req.chapter_id and current_char_id:
                    pk = f"audio/segments/{req.book_id}/{req.chapter_id}/{current_char_id}/{req.s3_filename}"
                    print(f"Migrating {jk} -> {pk}")
                    storage.move_object(jk, pk)
                    s3_key_found = pk
                else:
                    s3_key_found = jk

    if s3_key_found == proper_key:
        print("SUCCESS: File migrated from job path to proper path.")
    else:
        print(f"FAILED: Expected {proper_key}, got {s3_key_found}")

    # Cleanup
    if storage.object_exists(proper_key):
        storage.delete_object(proper_key)
    if storage.object_exists(job_key):
        storage.delete_object(job_key)

if __name__ == "__main__":
    asyncio.run(test_migration_logic())
