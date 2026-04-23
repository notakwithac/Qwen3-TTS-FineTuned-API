import sys
import types

import pytest
from pydantic import ValidationError


class _DummyBroadcast:
    def __init__(self, *_args, **_kwargs):
        pass

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def publish(self, *_args, **_kwargs):
        return None

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


def _build_items(count: int) -> list[dict[str, str]]:
    return [
        {
            "text": f"Prompt text {index + 1}",
            "filename": f"clone_batch_{index + 1:04d}.wav",
        }
        for index in range(count)
    ]


def test_voice_clone_batch_request_accepts_72_items():
    assert api_server.MAX_CLONE_BATCH_ITEMS >= 72

    req = api_server.VoiceCloneBatchRequest(
        ref_audio_url="https://example.com/ref.wav",
        ref_text="Reference text",
        items=_build_items(72),
    )

    assert len(req.items) == 72


def test_voice_clone_batch_request_rejects_items_above_limit():
    with pytest.raises(ValidationError, match="items exceeds max batch size"):
        api_server.VoiceCloneBatchRequest(
            ref_audio_url="https://example.com/ref.wav",
            ref_text="Reference text",
            items=_build_items(api_server.MAX_CLONE_BATCH_ITEMS + 1),
        )
