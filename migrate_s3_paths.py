import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from storage import storage

def migrate_paths():
    if not storage.is_configured:
        print("Error: Storage is not configured. Please check your .env variables.")
        sys.exit(1)

    print(f"Connected to bucket: {storage.bucket}")
    print("Fetching objects under 'audio/segments/'...")
    
    # storage.list_objects might return up to 1000 items if not paginated properly,
    # but let's assume it works for the current scale or we can use boto3 directly for pagination
    # storage.client.get_paginator('list_objects_v2') is better for large buckets.
    paginator = storage.client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=storage.bucket, Prefix='audio/segments/')

    migration_count = 0
    error_count = 0

    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            parts = key.split('/')
            
            # audio/segments/{book_id}/{chapter_id}/{character_id}/{filename} 
            # Parts length would be 6: ['audio', 'segments', 'book_id', 'chapter_id', 'character_id', 'filename']
            if len(parts) == 6:
                book_id = parts[2]
                chapter_id = parts[3]
                character_id = parts[4]
                filename = parts[5]
                
                # New path: audio/segments/{book_id}/{chapter_id}/{filename}
                new_key = f"audio/segments/{book_id}/{chapter_id}/{filename}"
                
                try:
                    storage.move_object(key, new_key)
                    migration_count += 1
                except Exception as e:
                    print(f"Failed to migrate {key}: {e}")
                    error_count += 1

    print("-" * 40)
    print(f"Migration complete. Moved {migration_count} files.")
    if error_count > 0:
        print(f"Encountered {error_count} errors.")

    print("\nNote: Empty directories in S3 are just prefixes and don't need explicit deletion once all their objects are moved.")

if __name__ == "__main__":
    migrate_paths()
