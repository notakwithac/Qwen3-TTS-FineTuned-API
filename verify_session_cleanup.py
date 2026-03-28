import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

# Mock Session and SessionStatus
class SessionStatus:
    PREPARING = "preparing"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Session:
    def __init__(self, sid):
        self.session_id = sid
        self.last_active = time.time() - 4000  # Expired (timeout is 3600)
        self.status = SessionStatus.READY

async def test_cleanup_loop_logic():
    # Mock SessionManager logic
    sessions = {
        "ch_1": Session("ch_1"),
        "ch_2": Session("ch_2")
    }
    session_timeout = 3600
    
    def get_expired():
        now = time.time()
        return [
            sid for sid, s in sessions.items()
            if (now - s.last_active) > session_timeout
            and s.status not in (SessionStatus.PREPARING, SessionStatus.PROCESSING, SessionStatus.CANCELLED)
        ]

    # First pass: find both ch_1 and ch_2
    expired = get_expired()
    print(f"Pass 1 found: {expired}")
    if len(expired) != 2:
        print("FAIL: Expected 2 expired sessions")
        return

    # Simulate teardown for ch_1
    sessions["ch_1"].status = SessionStatus.CANCELLED
    print("Tore down ch_1 (status = CANCELLED)")

    # Second pass: should only find ch_2
    expired = get_expired()
    print(f"Pass 2 found: {expired}")
    if len(expired) != 1 or expired[0] != "ch_2":
        print("FAIL: Expected only ch_2 to be expired in second pass")
        return

    print("SUCCESS: Session cleanup loop logic verified")

if __name__ == "__main__":
    asyncio.run(test_cleanup_loop_logic())
