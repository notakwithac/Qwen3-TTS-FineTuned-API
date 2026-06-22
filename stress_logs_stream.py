import argparse
import sys
import threading
import time
import uuid

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that /logs/stream captures live background inference activity."
    )
    parser.add_argument("--base-url", required=True, help="Example: http://<ip>:8000")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output-file", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_success(response: requests.Response, context: str) -> dict:
    if response.status_code != 200:
        raise RuntimeError(
            f"{context} failed with {response.status_code}: {response.text[:500]}"
        )
    return response.json()


def stream_logs(base_url: str, messages: list[str], stop_event: threading.Event, ready_event: threading.Event):
    try:
        with requests.get(
            f"{base_url}/logs/stream",
            stream=True,
            timeout=(10, 60),
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"/logs/stream failed with {response.status_code}: {response.text[:500]}"
                )
            ready_event.set()
            for raw_line in response.iter_lines(decode_unicode=True):
                if stop_event.is_set():
                    break
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    messages.append(raw_line[6:])
    finally:
        ready_event.set()


def submit_voice_design(base_url: str, overwrite: bool, token: str) -> dict:
    payload = {
        "text": f"Log stream design probe {token}.",
        "instruct": f"A calm narrator for log stream verification {token}.",
        "character_name": f"log_stream_{token}",
        "s3_filename": f"log_stream_design_{token}.wav",
        "upload_to_s3": True,
        "overwrite": overwrite,
    }
    response = requests.post(f"{base_url}/voice-design", json=payload, timeout=300)
    body = _require_success(response, "voice-design")
    body["ref_text"] = payload["text"]
    return body


def submit_clone_batch(base_url: str, overwrite: bool, session_id: str, ref_audio_url: str, ref_text: str) -> None:
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
                "text": f"Clone stream verification line {index + 1} for {session_id}.",
                "filename": f"{session_id}_{index + 1:02d}.wav",
            }
            for index in range(2)
        ],
    }
    response = requests.post(f"{base_url}/voice-clone/batch", json=payload, timeout=60)
    _require_success(response, "voice-clone/batch")


def poll_clone_batch(base_url: str, session_id: str, poll_interval: float, timeout_seconds: float) -> dict:
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
        raise RuntimeError(f"No status received for clone session {session_id}")
    if last_body.get("status") == "processing":
        raise RuntimeError(f"Clone session {session_id} never left processing")
    if last_body.get("status") == "failed":
        raise RuntimeError(f"Clone session {session_id} failed: {last_body.get('error')}")

    results = last_body.get("results") or last_body.get("clones") or []
    if len(results) != 2:
        raise RuntimeError(f"Clone session {session_id} returned {len(results)} results instead of 2")
    if any(result.get("status") == "failed" for result in results):
        raise RuntimeError(f"Clone session {session_id} returned failed result(s): {results}")
    return last_body


def wait_for_expected_messages(
    messages: list[str],
    *,
    expected_fragments: list[str],
    timeout_seconds: float,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        joined = "\n".join(messages)
        if all(fragment in joined for fragment in expected_fragments):
            return
        time.sleep(0.5)
    missing = [fragment for fragment in expected_fragments if fragment not in "\n".join(messages)]
    raise RuntimeError(f"Missing expected streamed log fragments: {missing}")


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    token = uuid.uuid4().hex[:8]
    session_id = f"logclone_{token}"
    streamed_messages: list[str] = []
    stop_event = threading.Event()
    ready_event = threading.Event()

    worker = threading.Thread(
        target=stream_logs,
        args=(base_url, streamed_messages, stop_event, ready_event),
        daemon=True,
    )
    worker.start()

    if not ready_event.wait(timeout=15):
        raise RuntimeError("Timed out waiting for /logs/stream connection")

    design_result = submit_voice_design(base_url, args.overwrite, token)
    ref_audio_url = design_result.get("presigned_url") or design_result.get("s3_url")
    if not ref_audio_url:
        raise RuntimeError(f"Voice design response did not include an audio URL: {design_result}")

    submit_clone_batch(
        base_url,
        args.overwrite,
        session_id,
        ref_audio_url,
        design_result["ref_text"],
    )
    poll_clone_batch(base_url, session_id, args.poll_interval, args.timeout_seconds)

    wait_for_expected_messages(
        streamed_messages,
        expected_fragments=[
            "VoiceDesign started",
            "VoiceDesign finished",
            "VoiceClone flexible started",
            "VoiceClone flexible finished",
            session_id,
        ],
        timeout_seconds=min(args.timeout_seconds, 60.0),
    )

    stop_event.set()
    worker.join(timeout=5)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(streamed_messages))

    print("Log stream verification passed.")
    print(f"Captured {len(streamed_messages)} streamed log messages.")
    print(f"Clone session id: {session_id}")
    if args.output_file:
        print(f"Saved stream transcript to {args.output_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Log stream verification failed: {exc}", file=sys.stderr)
        raise
