import json
import unittest
from unittest.mock import patch, MagicMock
import os
import io

# Mocking the configuration and logger
class MockLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass

logger = MockLogger()

def is_api_busy_v3(running_ops_data) -> bool:
    data = running_ops_data
    if isinstance(data, list) and len(data) > 0:
        active_ops = []
        for op in data:
            op_name = op.get("op_name", "")
            extra = op.get("extra", {})
            url = extra.get("url", "")
            
            is_status_check = (
                op_name == "api_request" and (
                    "/ops/running" in url or 
                    "/gpu/status" in url or 
                    "/gpu/vram" in url or
                    "/ops/averages" in url or
                    "/ops/history" in url or
                    "/storage/status" in url or
                    "/session/" in url or
                    "/sessions" in url or
                    "/gpu/terminate" in url
                )
            ) or (
                op_name in ("session_teardown", "session_auto_cleanup")
            )
            
            if not is_status_check:
                active_ops.append(op)

        if active_ops:
            return True
    return False

def resolve_instance_uuid_mock(env_dict) -> str | None:
    uuid_env = (
        env_dict.get("GPU_INSTANCE_ID", "").strip() or 
        env_dict.get("MASSED_COMPUTE_INSTANCE_UUID", "").strip() or
        env_dict.get("nGPU_INSTANCE_UUID", "").strip()
    )
    return uuid_env if uuid_env else None

class TestWatchdogRefined(unittest.TestCase):
    def test_terminate_endpoint_ignored(self):
        data = [{"op_name": "api_request", "extra": {"url": "http://localhost:8000/gpu/terminate"}}]
        self.assertFalse(is_api_busy_v3(data), "/gpu/terminate should be ignored")

    def test_ngpu_uuid_resolution(self):
        env = {"nGPU_INSTANCE_UUID": "test-ngpu-123"}
        self.assertEqual(resolve_instance_uuid_mock(env), "test-ngpu-123")

    def test_massed_compute_uuid_resolution(self):
        env = {"MASSED_COMPUTE_INSTANCE_UUID": "test-mc-123"}
        self.assertEqual(resolve_instance_uuid_mock(env), "test-mc-123")

    def test_priority_uuid_resolution(self):
        env = {
            "GPU_INSTANCE_ID": "priority-1",
            "MASSED_COMPUTE_INSTANCE_UUID": "priority-2",
            "nGPU_INSTANCE_UUID": "priority-3"
        }
        self.assertEqual(resolve_instance_uuid_mock(env), "priority-1")

if __name__ == "__main__":
    unittest.main()
