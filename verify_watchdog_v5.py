import json
import unittest
from unittest.mock import MagicMock, patch

# Mocking the constants from the script
IDLE_THRESHOLD = 5
TERMINATE_MINUTES = 20

def is_api_busy_logic(data):
    """Extracted logic from gpu_idle_watchdog.py for testing."""
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
                    "/gpu/terminate" in url or
                    "/docs" in url or
                    "/redoc" in url or
                    "/openapi.json" in url or
                    "/favicon.ico" in url or
                    url.endswith("/")
                )
            ) or (
                op_name in ("session_teardown", "session_auto_cleanup")
            )
            
            if not is_status_check:
                active_ops.append(op)

        if active_ops:
            return True, active_ops
    return False, []

class TestWatchdogLogic(unittest.TestCase):
    def test_status_endpoints_ignored(self):
        test_cases = [
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/ops/running"}},
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/gpu/status"}},
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/docs"}},
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/favicon.ico"}},
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/"}},
            {"op_name": "session_teardown", "extra": {}},
        ]
        busy, active = is_api_busy_logic(test_cases)
        self.assertFalse(busy, f"Should be IDLE but found active ops: {active}")

    def test_real_work_detected(self):
        test_cases = [
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/infer/abc123"}},
            {"op_name": "api_request", "extra": {"url": "http://localhost:8000/finetune"}},
            {"op_name": "training", "extra": {"job_id": "job1"}},
        ]
        busy, active = is_api_busy_logic(test_cases)
        self.assertTrue(busy, "Should be BUSY for actual work")
        self.assertEqual(len(active), 3)

if __name__ == "__main__":
    unittest.main()
