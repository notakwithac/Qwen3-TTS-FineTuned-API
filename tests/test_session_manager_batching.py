import asyncio
import time
from threading import Event

import pytest

from session_manager import CharacterProgress, CharacterWorker, InferenceMessage


class _StubInferenceManager:
    max_models = 1

    def __init__(self):
        self.calls = []

    def generate_batch(self, texts, checkpoint_path, speaker_name, languages=None, instructs=None, **kwargs):
        self.calls.append(
            {
                "texts": texts,
                "checkpoint_path": checkpoint_path,
                "speaker_name": speaker_name,
                "languages": languages,
                "instructs": instructs,
                "kwargs": kwargs,
            }
        )
        return [b"wav" for _ in texts], 24000


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
        initial_batch_size=10,
        progress=CharacterProgress(),
    )

    seen_batches: list[list[str]] = []

    async def _capture(batch, _loop):
        seen_batches.append([msg.text for msg in batch])
        worker._first_batch_pending = False
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
async def test_worker_uses_single_item_initial_batch_for_faster_first_result():
    queue = asyncio.Queue()
    worker = CharacterWorker(
        worker_id="worker-initial",
        queue=queue,
        inference_manager=_StubInferenceManager(),
        worker_semaphore=asyncio.Semaphore(1),
        cache_key="checkpoint",
        speaker_name="Narrator",
        batch_size=10,
        batch_timeout_ms=1,
        batch_text_budget=100,
        progress=CharacterProgress(),
    )

    seen_batches: list[list[str]] = []

    async def _capture(batch, _loop):
        seen_batches.append([msg.text for msg in batch])
        worker._first_batch_pending = False
        if len(seen_batches) >= 2:
            worker._running = False

    worker._process_batch = _capture  # type: ignore[method-assign]

    await queue.put(_msg("first"))
    await queue.put(_msg("second"))
    await queue.put(_msg("third"))

    worker.start()
    await asyncio.wait_for(worker._task, timeout=1.0)

    assert seen_batches == [["first"], ["second", "third"]]


@pytest.mark.asyncio
async def test_worker_splits_when_padded_batch_cost_exceeds_budget():
    queue = asyncio.Queue()
    worker = CharacterWorker(
        worker_id="worker-pad",
        queue=queue,
        inference_manager=_StubInferenceManager(),
        worker_semaphore=asyncio.Semaphore(1),
        cache_key="checkpoint",
        speaker_name="Narrator",
        batch_size=10,
        batch_timeout_ms=1,
        batch_text_budget=100,
        batch_padded_text_budget=12,
        initial_batch_size=10,
        progress=CharacterProgress(),
    )

    seen_batches: list[list[str]] = []

    async def _capture(batch, _loop):
        seen_batches.append([msg.text for msg in batch])
        worker._first_batch_pending = False
        if len(seen_batches) >= 2:
            worker._running = False

    worker._process_batch = _capture  # type: ignore[method-assign]

    await queue.put(_msg("aaaaaa"))
    await queue.put(_msg("bb"))
    await queue.put(_msg("cc"))

    worker.start()
    await asyncio.wait_for(worker._task, timeout=1.0)

    assert seen_batches == [["aaaaaa", "bb"], ["cc"]]


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
        batch_padded_text_budget=10,
        initial_batch_size=10,
        progress=CharacterProgress(),
    )

    seen_batches: list[list[str]] = []

    async def _capture(batch, _loop):
        seen_batches.append([msg.text for msg in batch])
        worker._first_batch_pending = False
        if len(seen_batches) >= 2:
            worker._running = False

    worker._process_batch = _capture  # type: ignore[method-assign]

    await queue.put(_msg("x" * 25))
    await queue.put(_msg("tail"))

    worker.start()
    await asyncio.wait_for(worker._task, timeout=1.0)

    assert seen_batches == [["x" * 25], ["tail"]]


@pytest.mark.asyncio
async def test_worker_stop_waits_for_active_batch_completion():
    queue = asyncio.Queue()
    started = Event()

    class _SlowInferenceManager(_StubInferenceManager):
        def generate_batch(self, texts, checkpoint_path, speaker_name, languages=None, instructs=None, **kwargs):
            started.set()
            time.sleep(0.05)
            return super().generate_batch(
                texts,
                checkpoint_path,
                speaker_name,
                languages,
                instructs,
                **kwargs,
            )

    worker = CharacterWorker(
        worker_id="worker-3",
        queue=queue,
        inference_manager=_SlowInferenceManager(),
        worker_semaphore=asyncio.Semaphore(1),
        cache_key="checkpoint",
        speaker_name="Narrator",
        batch_size=10,
        batch_timeout_ms=1,
        batch_text_budget=100,
        initial_batch_size=10,
        progress=CharacterProgress(),
    )

    await queue.put(_msg("wait for me"))

    worker.start()
    started_ok = await asyncio.to_thread(started.wait, 1.0)
    assert started_ok

    await worker.stop()

    assert worker.progress.completed == 1
    assert worker._task is not None
    assert worker._task.done()


@pytest.mark.asyncio
async def test_worker_applies_bounded_deterministic_generation_defaults():
    queue = asyncio.Queue()
    inference = _StubInferenceManager()
    worker = CharacterWorker(
        worker_id="worker-generate-kwargs",
        queue=queue,
        inference_manager=inference,
        worker_semaphore=asyncio.Semaphore(1),
        cache_key="checkpoint",
        speaker_name="Narrator",
        batch_size=1,
        batch_timeout_ms=1,
        generation_kwargs={"do_sample": False, "subtalker_dosample": False},
        min_new_tokens=512,
        max_new_tokens=1536,
        max_new_tokens_per_char=4,
        progress=CharacterProgress(),
    )

    await worker._process_batch([_msg("x" * 325)], asyncio.get_running_loop())

    assert inference.calls
    assert inference.calls[-1]["kwargs"] == {
        "do_sample": False,
        "subtalker_dosample": False,
        "max_new_tokens": 1300,
    }
