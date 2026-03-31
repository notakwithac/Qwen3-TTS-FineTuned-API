import os
import shutil
import zipfile
import uuid
from pathlib import Path
from pipeline import Pipeline, JobStatus

def setup_dummy_dataset(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('train.jsonl', '{"audio": "data/1.wav", "text": "test"}\n')
        os.makedirs('data', exist_ok=True)
        with open('data/1.wav', 'wb') as f:
            f.write(b'dummy wav')
        zf.write('data/1.wav')
        shutil.rmtree('data')

def test_pipeline_job_id_reuse():
    base_dir = "test_workspace"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    pipeline = Pipeline(base_dir=base_dir, jobs_dir="jobs")
    
    zip_path = os.path.join(base_dir, "test_dataset.zip")
    setup_dummy_dataset(zip_path)
    
    # 1. Create with random ID
    job1 = pipeline.create_job(zip_path, speaker_name="test_speaker")
    print(f"Created job1 with ID: {job1.job_id}")
    assert len(job1.job_id) == 12
    
    # 2. Create with specific ID
    custom_id = "custom_job_id"
    job2 = pipeline.create_job(zip_path, speaker_name="test_speaker", job_id=custom_id)
    print(f"Created job2 with ID: {job2.job_id}")
    assert job2.job_id == custom_id
    assert os.path.exists(os.path.join(base_dir, "jobs", custom_id))
    
    # 3. Reuse same ID (cleanup check)
    # Create some dummy file in the output dir to see if it gets cleaned
    output_file = os.path.join(base_dir, "jobs", custom_id, "output", "should_be_deleted.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("delete me")
    
    print(f"Re-creating job with ID: {custom_id}")
    job3 = pipeline.create_job(zip_path, speaker_name="test_speaker_new", job_id=custom_id)
    assert job3.job_id == custom_id
    assert not os.path.exists(output_file), "Existing directory was not cleaned up!"
    print("Cleanup verified.")
    
    print("Pipeline tests passed!")
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    test_pipeline_job_id_reuse()
