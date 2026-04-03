import unittest
from unittest.mock import MagicMock, patch
import json

# Minimal mocks to allow importing and testing api_server logic
with patch('ops_logger.ops_log'), \
     patch('storage.storage'), \
     patch('pipeline.Pipeline'):
    from api_server import create_finetune_job, retry_job, FinetuneRequest, JobStatus

class TestForceFlag(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline_mock = MagicMock()
        self.req = FinetuneRequest(
            dataset_s3_key="test.zip",
            speaker_name="Speaker1",
            job_id="job123",
            force=False
        )

    @patch('api_server.pipeline')
    def test_create_job_idempotent(self, mock_pipeline):
        # Setup: Job already exists
        existing_job = MagicMock()
        existing_job.to_dict.return_value = {"job_id": "job123", "status": "queued"}
        mock_pipeline.get_job.return_value = existing_job
        
        # Action: Create with force=False
        self.req.force = False
        response = create_finetune_job(self.req)
        
        # Verify: Returns 200 OK, does not call create_job
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["status"], "queued")
        mock_pipeline.create_job.assert_not_called()

    @patch('api_server.pipeline')
    def test_create_job_force(self, mock_pipeline):
        # Setup: Job already exists
        existing_job = MagicMock()
        mock_pipeline.get_job.return_value = existing_job
        
        # Action: Create with force=True
        self.req.force = True
        # We expect it to proceed to download and create. 
        # Mocking the rest of the function to avoid S3 calls.
        with patch('api_server.storage.download_file'), \
             patch('api_server.tempfile.NamedTemporaryFile') as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = "/tmp/test.zip"
            create_finetune_job(self.req)
        
        # Verify: create_job WAS called despite existing job
        mock_pipeline.create_job.assert_called_once()

    @patch('api_server.pipeline')
    async def test_retry_job_no_force(self, mock_pipeline):
        # Setup: retry_job (pipeline) returns None (e.g. status is QUEUED)
        mock_pipeline.retry_job.return_value = None
        existing_job = MagicMock()
        existing_job.status = JobStatus.QUEUED
        existing_job.to_dict.return_value = {"status": "queued"}
        mock_pipeline.get_job.return_value = existing_job
        
        # Action: Retry with force=False
        self.req.force = False
        response = await retry_job("job123", self.req)
        
        # Verify: Returns 200 OK, doesn't re-create
        self.assertEqual(response.status_code, 200)
        mock_pipeline.create_job.assert_not_called()

    @patch('api_server.pipeline')
    async def test_retry_job_with_force(self, mock_pipeline):
        # Setup: retry_job (pipeline) returns None
        mock_pipeline.retry_job.return_value = None
        existing_job = MagicMock()
        mock_pipeline.get_job.return_value = existing_job
        
        # Action: Retry with force=True
        self.req.force = True
        with patch('api_server.storage.download_file'), \
             patch('api_server.tempfile.NamedTemporaryFile') as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = "/tmp/test.zip"
            await retry_job("job123", self.req)
        
        # Verify: create_job WAS called
        mock_pipeline.create_job.assert_called_once()

    @patch('api_server.pipeline')
    async def test_retry_job_failed_status_no_force(self, mock_pipeline):
        # Setup: Job is FAILED, so even without force=False, it should restart 
        # (because it's not in QUEUED/TRAINING)
        mock_pipeline.retry_job.return_value = None
        existing_job = MagicMock()
        existing_job.status = JobStatus.FAILED
        mock_pipeline.get_job.return_value = existing_job
        
        # Action: Retry with force=False
        self.req.force = False
        with patch('api_server.storage.download_file'), \
             patch('api_server.tempfile.NamedTemporaryFile') as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = "/tmp/test.zip"
            await retry_job("job123", self.req)
        
        # Verify: create_job WAS called despite force=False
        mock_pipeline.create_job.assert_called_once()

if __name__ == "__main__":
    unittest.main()
