import asyncio

import pytest

from session_manager import CharacterProgress, CharacterWorker, InferenceMessage


class _StubInferenceManager:
    max_models = 1


def _msg(text: str) -> InferenceMessage:
    return InferenceMessage(
        session_id="session-1",
        job_id="job-1",
        character_name="Narrator",
        text=text,
    )


@pytest.mark.asyncio
async def test_worker_groups_messages_until_text_budget_is_reached():
    queue = asyncio.Queue()
    worker = CharacterWorker(
        worker_id="worker-1",
        queue=queue,
        inference_manager=_StubInferenceManager(),
        worker_semaphore=asyncio.Semaphore(1),
        cache_key="checkpoint",
        speaker_name="Narrator",
        batch_size=10,
        batch_timeout_ms=1,
        batch_text_budget=6,
        progress=CharacterProgress(),
    )

    seen_batches: list[list[str]] = []

    async def _capture(batch, _loop):
        seen_batches.append([msg.text for msg in batch])
        if len(seen_batches) >= 2:
            worker._running = False

    worker._process_batch = _capture  # type: ignore[method-assign]

    await queue.put(_msg("aaa"))
    await queue.put(_msg("bbb"))
    await queue.put(_msg("ccc"))

    worker.start()
    await asyncio.wait_for(worker._task, timeout=1.0)

    assert seen_batches == [["aaa", "bbb"], ["ccc"]]


@pytest.mark.asyncio
async def test_worker_allows_single_oversized_dialogue_as_its_own_batch():
    queue = asyncio.Queue()
    worker = CharacterWorker(
        worker_id="worker-2",
        queue=queue,
        inference_manager=_StubInferenceManager(),
        worker_semaphore=asyncio.Semaphore(1),
        cache_key="checkpoint",
        speaker_name="Narrator",
        batch_size=10,
        batch_timeout_ms=1,
        batch_text_budget=10,
        progress=CharacterProgress(),
    )

    seen_batches: list[list[str]] = []

    async def _capture(batch, _loop):
        seen_batches.append([msg.text for msg in batch])
        if len(seen_batches) >= 2:
            worker._running = False

    worker._process_batch = _capture  # type: ignore[method-assign]

    await queue.put(_msg("x" * 25))
    await queue.put(_msg("tail"))

    worker.start()
    await asyncio.wait_for(worker._task, timeout=1.0)

    assert seen_batches == [["x" * 25], ["tail"]]
