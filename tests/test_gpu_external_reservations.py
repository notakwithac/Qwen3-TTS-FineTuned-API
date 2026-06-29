import threading

from inference_manager import RuntimeAdjustableLimiter


def test_external_reservation_blocks_normal_work_until_released():
    limiter = RuntimeAdjustableLimiter(2)
    reservation = limiter.acquire_all("external:dots")
    acquired = threading.Event()

    worker = threading.Thread(target=lambda: (limiter.acquire("qwen"), acquired.set()))
    worker.start()
    assert not acquired.wait(0.05)

    limiter.release_many("external:dots", reservation)
    assert acquired.wait(1)
    limiter.release("qwen")
    worker.join(1)


def test_reservation_releases_count_captured_at_acquire_time():
    limiter = RuntimeAdjustableLimiter(2)
    reservation = limiter.acquire_all("external:dots")
    limiter.update_capacity(3)
    limiter.release_many("external:dots", reservation)
    assert limiter.snapshot()["active"] == 0
