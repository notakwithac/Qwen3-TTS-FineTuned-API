from dotenv import load_dotenv
load_dotenv()
from storage import storage
import boto3

def list_and_debug():
    print(f"Checking bucket: {storage.bucket}")
    client = storage.client
    
    # List all objects with prefix 'audio/segments/'
    response = client.list_objects_v2(Bucket=storage.bucket, Prefix='audio/segments/')
    contents = response.get('Contents', [])
    print(f"Found {len(contents)} objects under 'audio/segments/'")
    
    for obj in contents[:5]:
        print(f" - {obj['Key']}")

    # Check for the specific book_id
    book_id = "ddbf58d7-469e-49f8-8bfc-37574f587f2d"
    book_prefix = f"audio/segments/{book_id}/"
    response = client.list_objects_v2(Bucket=storage.bucket, Prefix=book_prefix)
    book_contents = response.get('Contents', [])
    print(f"Found {len(book_contents)} objects under {book_prefix}")
    
    for obj in book_contents[:10]:
        print(f" - {obj['Key']}")

if __name__ == "__main__":
    list_and_debug()
