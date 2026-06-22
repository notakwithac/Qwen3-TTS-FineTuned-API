import argparse
import concurrent.futures
import sys
import time
import uuid
from dataclasses import dataclass

import requests


DEFAULT_TIMEOUT_SECONDS = 600


@dataclass
class PhaseResult:
    name: str
    duration_seconds: float
    successes: int
    failures: int
    session_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise shared VoiceDesign/VoiceClone replicas against a live server."
    )
    parser.add_argument("--base-url", required=True, help="Example: http://<ip>:8000")
    parser.add_argument("--design-count", type=int, default=3)
    parser.add_argument("--mixed-clone-count", type=int, default=3)
    parser.add_argument("--clone-only-count", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_success(response: requests.Response, context: str) -> dict:
    if response.status_code != 200:
        raise RuntimeError(
            f"{context} failed with {response.status_code}: {response.text[:500]}"
        )
    return response.json()


def _pick_audio_url(result: dict) -> str:
    url = result.get("presigned_url") or result.get("s3_url")
    if not url:
        raise RuntimeError(f"Design result did not include an audio URL: {result}")
    return url


def submit_voice_design(
    base_url: str,
    *,
    index: int,
    overwrite: bool,
    text_suffix: str,
) -> dict:
    payload = {
        "text": f"Stress design sample {index}: {text_suffix}.",
        "instruct": f"A clear narrator voice for stress run {index}, steady and natural.",
        "character_name": f"stress_designer_{index:02d}",
        "s3_filename": f"stress_design_{index:02d}.wav",
        "upload_to_s3": True,
        "overwrite": overwrite,
    }
    response = requests.post(f"{base_url}/voice-design", json=payload, timeout=300)
    body = _require_success(response, f"voice-design[{index}]")
    body["ref_text"] = payload["text"]
    return body


def submit_clone_batch(
    base_url: str,
    *,
    session_id: str,
    ref_audio_url: str,
    ref_text: str,
    item_count: int,
    overwrite: bool,
    text_prefix: str,
) -> dict:
    payload = {
        "session_id": session_id,
        "ref_audio_url": ref_audio_url,
        "ref_text": ref_text,
        "language": "English",
        "use_xvec": False,
        "upload_to_s3": True,
        "overwrite": overwrite,
        "items": [
            {
                "text": f"{text_prefix} line {idx + 1}.",
                "filename": f"{session_id}_{idx + 1:02d}.wav",
            }
            for idx in range(item_count)
        ],
    }
    response = requests.post(f"{base_url}/voice-clone/batch", json=payload, timeout=60)
    return _require_success(response, f"voice-clone/batch[{session_id}]")


def poll_clone_batch(
    base_url: str,
    *,
    session_id: str,
    expected_count: int,
    poll_interval: float,
    timeout_seconds: float,
) -> dict:
    deadline = time.time() + timeout_seconds
    last_body = None

    while time.time() < deadline:
        response = requests.get(f"{base_url}/voice-clone/batch/{session_id}", timeout=60)
        last_body = _require_success(response, f"voice-clone/batch/{session_id} status")
        status = last_body.get("status")
        if status in {"completed", "completed_with_errors", "failed"}:
            break
        time.sleep(poll_interval)

    if last_body is None:
        raise RuntimeError(f"voice-clone batch {session_id} returned no status payloads")

    if last_body.get("status") == "processing":
        raise RuntimeError(f"voice-clone batch {session_id} never left processing state")
    if last_body.get("status") == "failed":
        raise RuntimeError(
            f"voice-clone batch {session_id} failed: {last_body.get('error')}"
        )

    results = last_body.get("results") or last_body.get("clones") or []
    if len(results) != expected_count:
        raise RuntimeError(
            f"voice-clone batch {session_id} returned {len(results)} results, expected {expected_count}"
        )

    failed_results = [result for result in results if result.get("status") == "failed"]
    if failed_results:
        raise RuntimeError(
            f"voice-clone batch {session_id} had failed results: {failed_results[:2]}"
        )

    return last_body


def run_phase_voice_design(
    base_url: str,
    *,
    design_count: int,
    overwrite: bool,
) -> tuple[list[dict], PhaseResult]:
    started = time.perf_counter()
    suffix = uuid.uuid4().hex[:8]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(design_count, 8)) as executor:
        futures = [
            executor.submit(
                submit_voice_design,
                base_url,
                index=index + 1,
                overwrite=overwrite,
                text_suffix=suffix,
            )
            for index in range(design_count)
        ]
        results = [future.result() for future in futures]

    duration = time.perf_counter() - started
    return results, PhaseResult(
        name="voice_design_seed",
        duration_seconds=duration,
        successes=len(results),
        failures=0,
        session_ids=[],
    )


def run_phase_clone_wave(
    base_url: str,
    *,
    design_results: list[dict],
    batch_size: int,
    poll_interval: float,
    timeout_seconds: float,
    overwrite: bool,
    wave_name: str,
) -> PhaseResult:
    started = time.perf_counter()
    session_ids = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(design_results), 4)) as executor:
        submissions = []
        for index, result in enumerate(design_results, start=1):
            session_id = f"{wave_name}_{index:02d}_{uuid.uuid4().hex[:6]}"
            session_ids.append(session_id)
            submissions.append(
                executor.submit(
                    submit_clone_batch,
                    base_url,
                    session_id=session_id,
                    ref_audio_url=_pick_audio_url(result),
                    ref_text=result["ref_text"],
                    item_count=batch_size,
                    overwrite=overwrite,
                    text_prefix=f"{wave_name} clone batch {index}",
                )
            )
        for future in submissions:
            future.result()

    for session_id in session_ids:
        poll_clone_batch(
            base_url,
            session_id=session_id,
            expected_count=batch_size,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )

    duration = time.perf_counter() - started
    return PhaseResult(
        name=wave_name,
        duration_seconds=duration,
        successes=len(session_ids) * batch_size,
        failures=0,
        session_ids=session_ids,
    )


def run_phase_mixed_load(
    base_url: str,
    *,
    prior_design: dict,
    clone_count: int,
    poll_interval: float,
    timeout_seconds: float,
    overwrite: bool,
) -> tuple[dict, PhaseResult]:
    started = time.perf_counter()
    session_id = f"mixed_clone_{uuid.uuid4().hex[:6]}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        design_future = executor.submit(
            submit_voice_design,
            base_url,
            index=99,
            overwrite=overwrite,
            text_suffix=f"mixed-{uuid.uuid4().hex[:8]}",
        )
        clone_future = executor.submit(
            submit_clone_batch,
            base_url,
            session_id=session_id,
            ref_audio_url=_pick_audio_url(prior_design),
            ref_text=prior_design["ref_text"],
            item_count=clone_count,
            overwrite=overwrite,
            text_prefix="mixed load clone",
        )
        new_design = design_future.result()
        clone_future.result()

    poll_clone_batch(
        base_url,
        session_id=session_id,
        expected_count=clone_count,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
    )

    duration = time.perf_counter() - started
    return new_design, PhaseResult(
        name="mixed_design_and_clone",
        duration_seconds=duration,
        successes=clone_count + 1,
        failures=0,
        session_ids=[session_id],
    )


def print_summary(phases: list[PhaseResult]) -> None:
    total_requests = sum(phase.successes + phase.failures for phase in phases)
    print("\nStress summary")
    print("==============")
    print(f"Total successful operations: {sum(phase.successes for phase in phases)}")
    print(f"Total failed operations: {sum(phase.failures for phase in phases)}")
    print(f"Total tracked operations: {total_requests}")
    for phase in phases:
        print(
            f"- {phase.name}: {phase.duration_seconds:.2f}s, "
            f"successes={phase.successes}, failures={phase.failures}, "
            f"session_ids={phase.session_ids or '[]'}"
        )


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    phases: list[PhaseResult] = []

    design_results, phase = run_phase_voice_design(
            base_url,
            design_count=args.design_count,
            overwrite=args.overwrite,
    )
    phases.append(phase)

    phases.append(
        run_phase_clone_wave(
                base_url,
                design_results=design_results,
                batch_size=max(1, args.mixed_clone_count),
                poll_interval=args.poll_interval,
                timeout_seconds=args.timeout_seconds,
                overwrite=args.overwrite,
                wave_name="clone_wave",
        )
    )

    new_design, phase = run_phase_mixed_load(
            base_url,
            prior_design=design_results[0],
            clone_count=max(1, args.mixed_clone_count),
            poll_interval=args.poll_interval,
            timeout_seconds=args.timeout_seconds,
            overwrite=args.overwrite,
    )
    phases.append(phase)

    phases.append(
        run_phase_clone_wave(
                base_url,
                design_results=[new_design],
                batch_size=max(1, args.clone_only_count),
                poll_interval=args.poll_interval,
                timeout_seconds=args.timeout_seconds,
                overwrite=args.overwrite,
                wave_name="clone_only_wave",
        )
    )

    print_summary(phases)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Stress run failed: {exc}", file=sys.stderr)
        raise
