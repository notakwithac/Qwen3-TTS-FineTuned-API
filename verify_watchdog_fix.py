import json
import unittest
from unittest.mock import patch, MagicMock
import io

# Mocking the configuration and logger as they exist in gpu_idle_watchdog.py
class MockLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass

logger = MockLogger()

def is_api_busy_v2(running_ops_data) -> bool:
    """A version of is_api_busy that takes the raw data instead of calling the URL."""
    data = running_ops_data
    if isinstance(data, list) and len(data) > 0:
        active_ops = []
        for op in data:
            op_name = op.get("op_name", "")
            extra = op.get("extra", {})
            url = extra.get("url", "")
            
            # THE LOGIC TO TEST
            is_status_check = (
                op_name == "api_request" and (
                    "/ops/running" in url or 
                    "/gpu/status" in url or 
                    "/gpu/vram" in url or
                    "/ops/averages" in url or
                    "/ops/history" in url or
                    "/storage/status" in url or
                    "/session/" in url or
                    "/sessions" in url
                )
            ) or (
                op_name in ("session_teardown", "session_auto_cleanup")
            )
            
            if not is_status_check:
                active_ops.append(op)

        if active_ops:
            return True
    return False

class TestWatchdogLogic(unittest.TestCase):
    def test_status_checks_ignored(self):
        # Case 1: Only status checks running
        data = [
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/gpu/status"}},
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/session/ch_123/status"}}
        ]
        self.assertFalse(is_api_busy_v2(data), "Status checks should be ignored")

    def test_teardown_ignored(self):
        # Case 2: Teardown running
        data = [
            {"op_name": "session_teardown", "extra": {"session_id": "ch_123"}}
        ]
        self.assertFalse(is_api_busy_v2(data), "Session teardown should be ignored")

    def test_real_work_detected(self):
        # Case 3: Real work running
        data = [
            {"op_name": "inference_api", "extra": {"text_length": 100}},
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/gpu/status"}}
        ]
        self.assertTrue(is_api_busy_v2(data), "Real work (inference) should be detected")

    def test_finetune_detected(self):
        # Case 4: Finetune running
        data = [
            {"op_name": "finetune_job_create", "extra": {"speaker": "Voice1"}}
        ]
        self.assertTrue(is_api_busy_v2(data), "Finetune creation should be detected")

if __name__ == "__main__":
    unittest.main()
