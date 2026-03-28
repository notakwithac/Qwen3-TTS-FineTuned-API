import csv
import os
from dotenv import load_dotenv
load_dotenv()
from storage import storage
import boto3

def update_metadata():
    book_id_to_process = "ddbf58d7-469e-49f8-8bfc-37574f587f2d"
    csv_path = r"e:\Projects\Qwen-Finetune\pathnam.dialogues.csv"
    
    # 1. Read CSV mapping
    dialogue_to_job = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['book_id'] == book_id_to_process:
                job_id = row['finetune_job_id'].strip()
                if job_id:
                    dialogue_to_job[row['dialogue_id']] = job_id

    print(f"Loaded {len(dialogue_to_job)} mappings from CSV for book {book_id_to_process}")

    # 2. List S3 objects
    client = storage.client
    prefix = f"audio/segments/{book_id_to_process}/"
    
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=storage.bucket, Prefix=prefix)
    
    objects_to_update = []
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Find which dialogue_id this key belongs to
            matched_dialogue = None
            for d_id in dialogue_to_job:
                if d_id in key:
                    matched_dialogue = d_id
                    break
            
            if matched_dialogue:
                objects_to_update.append((key, dialogue_to_job[matched_dialogue]))

    print(f"Found {len(objects_to_update)} objects in S3 to update.")

    # 3. Perform updates
    updated_count = 0
    for key, job_id in objects_to_update:
        print(f"Updating {key} -> model-id: {job_id}")
        try:
            # We must use copy_object to update metadata
            copy_source = {'Bucket': storage.bucket, 'Key': key}
            client.copy_object(
                Bucket=storage.bucket,
                Key=key,
                CopySource=copy_source,
                Metadata={'model-id': job_id},
                MetadataDirective='REPLACE',
                # Keep other properties if possible, but usually we just care about ContentType
                ContentType='audio/wav' 
            )
            updated_count += 1
            if updated_count % 10 == 0:
                print(f"Progress: {updated_count}/{len(objects_to_update)}")
        except Exception as e:
            print(f"Failed to update {key}: {e}")

    print(f"Successfully updated {updated_count} objects.")

if __name__ == "__main__":
    update_metadata()
