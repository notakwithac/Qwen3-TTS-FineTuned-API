#!/usr/bin/env python3
"""
GPU Idle Watchdog (Simplified) — auto-terminates the Massed Compute VM after 
sustained pipeline inactivity (S3 uploads, training, inference).
"""

import argparse
import logging
import os
import socket
import sys
import time
import urllib.request
import json
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from massed_compute_client import MassedComputeClient
from storage import storage
import glob

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gpu-watchdog] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gpu-watchdog")

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
TERMINATE_MINUTES = int(os.getenv("GPU_IDLE_TERMINATE_MINUTES", "20"))
POLL_INTERVAL = int(os.getenv("GPU_POLL_INTERVAL", "60"))
# Grace period after /gpu/terminate signal: wait this long for in-flight ops to drain,
# then force-terminate even if the API still reports busy.
SIGNAL_GRACE_MINUTES = int(os.getenv("GPU_SIGNAL_GRACE_MINUTES", "5"))
# How long to wait for the API server to come up before starting idle countdown.
# During this window, Connection refused is normal (uvicorn still loading models).
API_STARTUP_WAIT_MINUTES = int(os.getenv("GPU_API_STARTUP_WAIT_MINUTES", "5"))

# --- Heartbeat / Periodic Upload Config ---
# How often to upload artifacts even if the API is busy (to prevent data loss).
PERIODIC_UPLOAD_MINUTES = int(os.getenv("GPU_PERIODIC_UPLOAD_MINUTES", "5"))
# Whether to include training/finetuning weights in the upload.
UPLOAD_OUTPUTS = os.getenv("GPU_UPLOAD_OUTPUTS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_api_busy() -> bool:
    """Check if the API server has active background operations (S3 uploads, training, etc)."""
    try:
        # We check localhost since the watchdog runs on the same VM
        with urllib.request.urlopen("http://localhost:8000/ops/running", timeout=5) as response:
            if response.status == 200:
                raw = response.read().decode()
                data = json.loads(raw)
                logger.debug("Raw /ops/running response (%d ops): %s", 
                             len(data) if isinstance(data, list) else -1, raw[:2000])
                # If the list is not empty, check if any operation is actually heavy
                if isinstance(data, list) and len(data) > 0:
                    # Filter out purely status-related API requests
                    active_ops = []
                    for op in data:
                        op_name = op.get("op_name", "")
                        op_id = op.get("op_id", "?")
                        extra = op.get("extra", {})
                        url = extra.get("url", "")
                        start_ts = op.get("start_ts", "?")
                        
                        # We ignore these specific status/ops requests
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
                        
                        if is_status_check:
                            logger.debug("  IGNORED op: name=%s id=%s url=%s", op_name, op_id, url)
                        else:
                            logger.info("  KEEPING op: name=%s id=%s url=%s started=%s", 
                                        op_name, op_id, url, start_ts)
                            active_ops.append(op)

                    if active_ops:
                        op_summaries = [
                            f"{o.get('op_name','?')}(id={o.get('op_id','?')}, started={o.get('start_ts','?')})"
                            for o in active_ops
                        ]
                        logger.info("API is BUSY with %d active operations: %s", 
                                    len(active_ops), ", ".join(op_summaries))
                        return True
                else:
                    logger.debug("/ops/running returned empty list — server is idle")
            else:
                logger.warning("/ops/running returned status %d", response.status)
    except Exception as exc:
        # If API is down or not responding, we assume it's NOT busy.
        # Connection refused is normal during API startup — log at DEBUG level.
        err_str = str(exc)
        if "Connection refused" in err_str or "Connection reset" in err_str:
            logger.debug("is_api_busy(): API not reachable yet (Connection refused) — treating as NOT busy")
        else:
            logger.warning("is_api_busy() check failed (treating as NOT busy): %s", exc)
    return False


def get_local_ips() -> list[str]:
    """Return a list of the machine's local IP addresses."""
    ips = ["127.0.0.1"]
    try:
        hostname = socket.gethostname()
        ips.extend([addr[4][0] for addr in socket.getaddrinfo(hostname, None)])
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return list(set(ips))


def resolve_instance_uuid(client: MassedComputeClient) -> str | None:
    """Resolve the current VM's Massed Compute instance UUID."""
    env_gpu_id = os.getenv("GPU_INSTANCE_ID", "").strip()
    env_mc_uuid = os.getenv("MASSED_COMPUTE_INSTANCE_UUID", "").strip()
    env_gpu_uuid = os.getenv("GPU_INSTANCE_UUID", "").strip()
    logger.info("Env vars: GPU_INSTANCE_ID='%s', MASSED_COMPUTE_INSTANCE_UUID='%s', GPU_INSTANCE_UUID='%s'",
                env_gpu_id, env_mc_uuid, env_gpu_uuid)
    
    uuid_env = env_gpu_id or env_mc_uuid or env_gpu_uuid
    if uuid_env:
        logger.info("Using instance UUID from env: %s", uuid_env)
        return uuid_env
    
    logger.info("Instance ID not set in env. Attempting IP resolution...")
    local_ips = get_local_ips()
    logger.info("Local IPs detected: %s", local_ips)
    try:
        instances = client.list_instances()
        logger.info("list_instances raw response type=%s: %s", type(instances).__name__, str(instances)[:500])
        if isinstance(instances, dict):
            instances = instances.get("runningInstances", instances.get("instances", []))
        
        logger.info("Found %d instances to check for IP match", len(instances) if isinstance(instances, list) else -1)
        for inst in instances:
            ip = inst.get("ip")
            inst_uuid = inst.get("uuid", "?")
            logger.info("  Instance: uuid=%s ip=%s (match=%s)", inst_uuid, ip, ip in local_ips)
            if ip in local_ips:
                logger.info("Auto-resolved instance UUID via IP match (%s): %s", ip, inst_uuid)
                return inst_uuid
        logger.warning("No IP match found among %d instances", len(instances) if isinstance(instances, list) else 0)
    except Exception as exc:
        logger.error("Failed to list instances for IP matching: %s", exc, exc_info=True)
    
    return None


def upload_final_artifacts(instance_uuid: str, is_periodic: bool = False):
    """Upload API logs, instance logs (/tmp), and training outputs to S3.
    
    Args:
        instance_uuid: The Massed Compute instance ID.
        is_periodic: If True, this is a heartbeat upload (uses 'periodic' folder).
    """
    if not storage.is_configured:
        logger.warning("Storage not configured — skipping artifact upload.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if is_periodic:
        # For heartbeats, use a fixed 'latest' folder to avoid S3 bloat
        s3_prefix = f"backups/{instance_uuid}/periodic/latest"
    else:
        # For final termination, use a timestamped folder for complete history
        s3_prefix = f"backups/{instance_uuid}/final/{timestamp}"
    
    # 1. API Logs & Metrics
    artifact_paths = [
        "logs",    # Directory
        "metrics", # Directory
        "/tmp",    # Directory (for system logs)
    ]
    
    # Files to glob for in those paths
    log_patterns = ["*.log", "*.jsonl", "*.txt", "events.out.tfevents.*"]
    # Model patterns: include config/trainer state always, weights only for final upload
    model_patterns = ["config.json", "trainer_state.json", "*.pt"] if UPLOAD_OUTPUTS else []
    if UPLOAD_OUTPUTS and not is_periodic:
        model_patterns.append("*.safetensors")
        
    all_patterns = list(set(log_patterns + model_patterns))

    found_files = []
    for base in artifact_paths:
        if not os.path.exists(base):
            continue
        
        if os.path.isfile(base):
            found_files.append(base)
            continue

        # Recursively find files matching patterns
        for root, _, files in os.walk(base):
            for filename in files:
                # Basic pattern matching
                matches = any(
                    (filename.endswith(p.replace("*", "")) if p.startswith("*") else filename == p)
                    for p in all_patterns
                )
                if matches:
                    found_files.append(os.path.join(root, filename))

    # 3. Collect from API history if possible (only for final or long-interval periodic)
    if not is_periodic or timestamp.endswith("00"): # simplified throttle for history
        try:
            with urllib.request.urlopen("http://localhost:8000/ops/history", timeout=5) as response:
                if response.status == 200:
                    history_data = response.read()
                    storage.upload_bytes(
                        history_data, 
                        f"{s3_prefix}/ops_history.json", 
                        content_type="application/json"
                    )
                    logger.info("Uploaded ops history to S3.")
        except Exception as exc:
            logger.debug("Could not retrieve ops history from API: %s", exc)

    # 4. Upload all collected files
    upload_count = 0
    for local_path in list(set(found_files)):
        if os.path.isfile(local_path):
            # Maintain some structure in S3
            rel_path = os.path.relpath(local_path, start=".")
            # Sanitize rel_path for S3 (remove leading dots/slashes)
            rel_path = rel_path.replace("..", "up").replace(":", "_")
            if rel_path.startswith("/") or rel_path.startswith("\\"):
                rel_path = rel_path[1:]
            
            s3_key = f"{s3_prefix}/{rel_path}"
                
            try:
                # Basic size throttle for periodic uploads
                file_size = os.path.getsize(local_path)
                if is_periodic and file_size > 1024*1024*500:
                    logger.warning("Skipping massive file %s (%.1f MB) during periodic upload", local_path, file_size/1e6)
                    continue

                storage.upload_file(local_path, s3_key)
                upload_count += 1
            except Exception as exc:
                logger.error("Failed to upload %s to S3: %s", local_path, exc)
    
    logger.info("Artifact upload complete (%d files -> %s)", upload_count, s3_prefix)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPU Idle Watchdog (Simplified)")
    parser.add_argument("--dry-run", action="store_true", help="Log only")
    parser.add_argument("--skip-auth", action="store_true", help="Testing only")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GPU Simplified Watchdog starting")
    logger.info("  Terminate after  : %d min of API idle", TERMINATE_MINUTES)
    logger.info("  Poll interval    : %ds", POLL_INTERVAL)
    logger.info("  Signal grace     : %d min (max wait after /gpu/terminate signal)", SIGNAL_GRACE_MINUTES)
    logger.info("  Dry-run          : %s", args.dry_run)
    logger.info("=" * 60)

    client = MassedComputeClient()
    if not args.skip_auth:
        logger.info("Validating Massed Compute API token...")
        if client.authenticate():
            logger.info("API token validation: OK")
        else:
            # Check if we already know the UUID — if so, we can still attempt termination
            # even if the token validation endpoint returns 4xx (it might be a different
            # endpoint auth scheme, or the validation endpoint itself has issues).
            uuid_from_env = (
                os.getenv("GPU_INSTANCE_ID", "").strip() or
                os.getenv("MASSED_COMPUTE_INSTANCE_UUID", "").strip() or
                os.getenv("GPU_INSTANCE_UUID", "").strip()
            )
            if uuid_from_env:
                logger.warning(
                    "API token validation FAILED (status 4xx) — but GPU_INSTANCE_UUID is set (%s). "
                    "Continuing anyway. The terminate call will fail if the token is truly invalid.",
                    uuid_from_env
                )
            else:
                logger.error("API token validation FAILED and no instance UUID found in env — cannot terminate safely. Exiting.")
                sys.exit(1)

    initial_uuid = resolve_instance_uuid(client)
    if not initial_uuid:
        logger.warning("Could not resolve instance UUID. Termination will fail.")
    else:
        logger.info("Termination target: %s", initial_uuid)

    idle_seconds = 0
    terminate_after = TERMINATE_MINUTES * 60
    signal_grace_seconds = SIGNAL_GRACE_MINUTES * 60
    signal_file = "terminate_signal.tmp"
    start_time = time.time()
    api_startup_wait_seconds = API_STARTUP_WAIT_MINUTES * 60

    # Track when the signal file was first detected (for grace period)
    signal_first_seen_at: float = 0.0

    logger.info("  Signal grace period  : %d min (force-terminate after this even if API reports busy)",
                SIGNAL_GRACE_MINUTES)
    logger.info("  API startup window   : %d min (Connection refused suppressed + idle timer paused)",
                API_STARTUP_WAIT_MINUTES)
    logger.info("  Periodic heartbeat   : %d min (upload artifacts even if busy)",
                PERIODIC_UPLOAD_MINUTES)

    last_periodic_upload_at = time.time()

    while True:
        api_busy = is_api_busy()
        termination_requested = os.path.exists(signal_file)
        uptime_seconds = time.time() - start_time
        in_startup_window = uptime_seconds < api_startup_wait_seconds
        
        # --- Periodic Heartbeat Upload ---
        now = time.time()
        if (now - last_periodic_upload_at) >= (PERIODIC_UPLOAD_MINUTES * 60):
            if initial_uuid:
                logger.info(">>> PERIODIC HEARTBEAT UPLOAD TRIGGERED (even if busy) <<<")
                try:
                    upload_final_artifacts(initial_uuid, is_periodic=True)
                    last_periodic_upload_at = now
                except Exception as exc:
                    logger.error("Periodic heartbeat upload failed: %s", exc)
            else:
                logger.warning("Periodic heartbeat skipped: instance UUID not resolved yet")

        if termination_requested:
            now = time.time()
            if signal_first_seen_at == 0.0:
                signal_first_seen_at = now
                logger.warning("!!! TERMINATION SIGNAL FILE DETECTED — starting %d-min drain grace period !!!",
                               SIGNAL_GRACE_MINUTES)

            elapsed_since_signal = now - signal_first_seen_at
            remaining = max(0.0, signal_grace_seconds - elapsed_since_signal)

            if api_busy and remaining > 0:
                logger.warning("Termination signal active — API still busy, waiting for ops to drain. "
                               "%.1f/%.0f min elapsed (%.1f min remaining in grace period).",
                               elapsed_since_signal / 60, SIGNAL_GRACE_MINUTES, remaining / 60)
            elif api_busy and remaining <= 0:
                logger.warning("!!! GRACE PERIOD EXPIRED — forcing termination despite API busy state !!!")
                logger.warning("A stuck operation is blocking shutdown. Force-terminating now.")
                idle_seconds = terminate_after
            else:
                logger.warning("!!! Termination signal active + API is IDLE — triggering termination !!!")
                idle_seconds = terminate_after
        elif not api_busy:
            if in_startup_window:
                logger.info("API not reachable yet — startup window active (%.0f/%.0f min). Idle timer paused.",
                            uptime_seconds / 60, API_STARTUP_WAIT_MINUTES)
            else:
                signal_first_seen_at = 0.0  # Reset if signal file disappears
                idle_seconds += POLL_INTERVAL
                logger.info("System IDLE (No active S3/inference/training) — %d/%d min until termination",
                            idle_seconds // 60, TERMINATE_MINUTES)
        else:
            if idle_seconds > 0:
                logger.info("System ACTIVE — Activity detected in API. Resetting idle timer (was %d min)",
                            idle_seconds // 60)
            signal_first_seen_at = 0.0
            idle_seconds = 0

        if idle_seconds >= terminate_after:
            logger.error("=" * 60)
            logger.error("TERMINATION TRIGGERED")
            logger.error("=" * 60)

            logger.info("Resolving instance UUID for termination...")
            uuid = resolve_instance_uuid(client)
            if not uuid:
                logger.error("FATAL: Cannot terminate — No instance UUID found. Will retry next cycle.")
                logger.error("  Check that GPU_INSTANCE_ID, MASSED_COMPUTE_INSTANCE_UUID, or GPU_INSTANCE_UUID is set in .env")
                idle_seconds = 0
                time.sleep(POLL_INTERVAL)
                continue

            if args.dry_run:
                logger.warning("[DRY RUN] Would call Massed Compute API to terminate %s", uuid)
                idle_seconds = 0
                time.sleep(POLL_INTERVAL)
                continue

            logger.error(">>> PRE-TERMINATE: uuid=%s, api_token=%s..., base_url=%s <<<",
                         uuid, client.api_token[:12] if client.api_token else "NONE", client.BASE_URL)
            
            # --- Upload artifacts before termination ---
            try:
                logger.info(">>> UPLOADING FINAL ARTIFACTS BEFORE TERMINATION <<<")
                upload_final_artifacts(uuid)
            except Exception as exc:
                logger.error("Artifact upload failed: %s", exc, exc_info=True)

            logger.error(">>> CALLING client.terminate_instance([%s]) NOW <<<", uuid)
            try:
                result = client.terminate_instance([uuid])
                logger.error(">>> POST-TERMINATE: API returned: %s <<<", result)
                logger.info("Termination request sent successfully. Goodbye. 👋")
                sys.exit(0)
            except Exception as exc:
                logger.error("FAILED TO TERMINATE INSTANCE: %s", exc, exc_info=True)
                # Log the full HTTP response if it's a requests exception
                if hasattr(exc, 'response') and exc.response is not None:
                    logger.error("  HTTP status: %s", exc.response.status_code)
                    logger.error("  Response body: %s", exc.response.text[:1000])
                idle_seconds = 0

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
