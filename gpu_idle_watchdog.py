#!/usr/bin/env python3
"""
GPU Idle Watchdog — auto-terminates the Massed Compute VM after sustained GPU
inactivity to avoid burning money on an idle instance.

Usage:
    python gpu_idle_watchdog.py              # production mode
    python gpu_idle_watchdog.py --dry-run    # logs only, no termination

Environment variables (all optional except the API token):
    MASSED_COMPUTE_API_TOKEN       – Required.  Bearer token for the MC API.
    MASSED_COMPUTE_INSTANCE_UUID   – If set, skips auto-detection of the
                                     instance UUID via IP matching.
    GPU_IDLE_THRESHOLD             – Utilization % below which the GPU is
                                     considered idle (default: 5).
    GPU_IDLE_TERMINATE_MINUTES     – Minutes of continuous idle before
                                     termination (default: 20).
    GPU_POLL_INTERVAL              – Seconds between nvidia-smi polls
                                     (default: 60).
"""

import argparse
import logging
import os
import socket
import subprocess
import sys
import time

from massed_compute_client import MassedComputeClient

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
IDLE_THRESHOLD = int(os.getenv("GPU_IDLE_THRESHOLD", "5"))
TERMINATE_MINUTES = int(os.getenv("GPU_IDLE_TERMINATE_MINUTES", "20"))
POLL_INTERVAL = int(os.getenv("GPU_POLL_INTERVAL", "60"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_gpu_utilization() -> int:
    """Return current GPU utilization % via nvidia-smi, or 0 on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi exited with code %d: %s",
                           result.returncode, result.stderr.strip())
            return 0
        # Multi-GPU: take the max utilization across all GPUs
        values = [int(v.strip()) for v in result.stdout.strip().split("\n") if v.strip()]
        return max(values) if values else 0
    except FileNotFoundError:
        logger.warning("nvidia-smi not found — assuming 0%% utilization")
        return 0
    except Exception as exc:
        logger.warning("Failed to read GPU utilization: %s", exc)
        return 0


def get_local_ips() -> list[str]:
    """Return a list of the machine's local IP addresses."""
    ips = []
    try:
        hostname = socket.gethostname()
        ips = list({addr[4][0] for addr in socket.getaddrinfo(hostname, None)})
    except Exception:
        pass

    # Also try the outbound-route trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    return list(set(ips))


def resolve_instance_uuid(client: MassedComputeClient) -> str | None:
    """Resolve the current VM's Massed Compute instance UUID.

    Priority:
      1. MASSED_COMPUTE_INSTANCE_UUID env var
      2. Match local IP against running instances from the API
    """
    uuid_env = os.getenv("MASSED_COMPUTE_INSTANCE_UUID", "").strip()
    if uuid_env:
        logger.info("Instance UUID from env: %s", uuid_env)
        return uuid_env

    logger.info("MASSED_COMPUTE_INSTANCE_UUID not set — resolving via IP match…")
    local_ips = get_local_ips()
    logger.info("Local IPs: %s", local_ips)

    try:
        instances = client.list_instances()
        # The API may return a list directly or nested under a key
        if isinstance(instances, dict):
            instances = instances.get("runningInstances", instances.get("instances", []))
        for inst in instances:
            if inst.get("ip") in local_ips:
                logger.info("Matched instance '%s' (uuid=%s, ip=%s)",
                            inst.get("name"), inst["uuid"], inst["ip"])
                return inst["uuid"]
    except Exception as exc:
        logger.error("Failed to list instances for IP matching: %s", exc)

    logger.error("Could not resolve instance UUID — no IP match found")
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPU Idle Watchdog")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions without actually terminating")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GPU Idle Watchdog starting")
    logger.info("  Idle threshold   : <%d%% utilization", IDLE_THRESHOLD)
    logger.info("  Terminate after  : %d min of sustained idle", TERMINATE_MINUTES)
    logger.info("  Poll interval    : %ds", POLL_INTERVAL)
    logger.info("  Dry-run          : %s", args.dry_run)
    logger.info("=" * 60)

    # Validate token early so we fail fast
    client = MassedComputeClient()
    if not client.authenticate():
        logger.error("API token validation failed — exiting")
        sys.exit(1)
    logger.info("API token validated ✓")

    idle_seconds = 0
    terminate_after = TERMINATE_MINUTES * 60

    while True:
        util = get_gpu_utilization()

        if util < IDLE_THRESHOLD:
            idle_seconds += POLL_INTERVAL
            logger.info(
                "GPU utilization: %d%% (idle) — idle for %d/%d min",
                util, idle_seconds // 60, TERMINATE_MINUTES,
            )
        else:
            if idle_seconds > 0:
                logger.info(
                    "GPU utilization: %d%% (active) — idle counter reset "
                    "(was %d min)",
                    util, idle_seconds // 60,
                )
            else:
                logger.info("GPU utilization: %d%% (active)", util)
            idle_seconds = 0

        # --- Termination trigger ---
        if idle_seconds >= terminate_after:
            logger.warning(
                "🛑  GPU idle for %d minutes — triggering instance termination",
                idle_seconds // 60,
            )

            uuid = resolve_instance_uuid(client)
            if not uuid:
                logger.error(
                    "Cannot terminate: instance UUID unknown. "
                    "Set MASSED_COMPUTE_INSTANCE_UUID and restart."
                )
                # Reset and keep trying — don't exit, in case the API
                # recovers or user sets the env var.
                idle_seconds = 0
                time.sleep(POLL_INTERVAL)
                continue

            if args.dry_run:
                logger.warning(
                    "DRY RUN: Would terminate instance %s — skipping", uuid
                )
                idle_seconds = 0
                time.sleep(POLL_INTERVAL)
                continue

            # Flush logs before the VM disappears
            logger.warning("Terminating instance %s NOW", uuid)
            sys.stdout.flush()
            sys.stderr.flush()

            try:
                result = client.terminate_instance([uuid])
                logger.info("Terminate API response: %s", result)
            except Exception as exc:
                logger.error("Terminate API call failed: %s", exc)
                # The VM may already be going down; exit cleanly
            finally:
                sys.stdout.flush()
                sys.stderr.flush()

            logger.info("Goodbye 👋")
            sys.exit(0)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
