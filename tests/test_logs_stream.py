import asyncio
import logging
import sys
import threading
import types


class _DummyBroadcast:
    def __init__(self, *_args, **_kwargs):
        self.messages = []

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def publish(self, channel, message):
        self.messages.append((channel, message))

    def subscribe(self, *_args, **_kwargs):
        class _Subscriber:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _Subscriber()


sse_module = types.ModuleType("sse_starlette.sse")
sse_module.EventSourceResponse = object
sys.modules.setdefault("sse_starlette", types.ModuleType("sse_starlette"))
sys.modules["sse_starlette.sse"] = sse_module

broadcast_module = types.ModuleType("broadcaster")
broadcast_module.Broadcast = _DummyBroadcast
sys.modules["broadcaster"] = broadcast_module

import api_server


def test_ops_logger_relies_on_root_handler_for_streaming():
    assert api_server.stream_handler in logging.getLogger().handlers
    assert api_server.stream_handler not in logging.getLogger("ops").handlers


def test_log_stream_handler_emits_background_thread_logs(monkeypatch):
    handler = api_server.LogStreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    broadcast = _DummyBroadcast()
    monkeypatch.setattr(api_server, "broadcast", broadcast)

    original_history = list(api_server.log_history)
    api_server.log_history.clear()

    async def run_test():
        handler.loop = asyncio.get_running_loop()
        record = logging.LogRecord(
            name="ops",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="[START] | op=inference_voice_clone_flexible_batch | op_id=test123",
            args=(),
            exc_info=None,
        )

        thread = threading.Thread(target=handler.emit, args=(record,))
        thread.start()
        thread.join()
        await asyncio.sleep(0.05)

    try:
        asyncio.run(run_test())
        history = list(api_server.log_history)
        assert any("inference_voice_clone_flexible_batch" in message for message in history)
        assert any(
            channel == "logs" and "inference_voice_clone_flexible_batch" in message
            for channel, message in broadcast.messages
        )
    finally:
        api_server.log_history.clear()
        api_server.log_history.extend(original_history)
