#!/usr/bin/env python3
"""
GPU Idle Watchdog — auto-terminates the Massed Compute VM after sustained GPU
inactivity and API idle to avoid burning money on an idle instance.
"""

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
import json

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
def is_api_busy() -> bool:
    """Check if the API server has active background operations."""
    try:
        # We check localhost since the watchdog runs on the same VM
        with urllib.request.urlopen("http://localhost:8000/ops/running", timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                # If the list is not empty, there are operations in progress
                if isinstance(data, list) and len(data) > 0:
                    logger.info("API is BUSY with %d active operations (e.g., %s)", 
                                len(data), data[0].get("op", "unknown"))
                    return True
    except Exception:
        # If API is down or not responding, we assume it's NOT busy
        pass
    return False


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
    uuid_env = os.getenv("MASSED_COMPUTE_INSTANCE_UUID", "").strip()
    if uuid_env:
        return uuid_env
    local_ips = get_local_ips()
    try:
        instances = client.list_instances()
        if isinstance(instances, dict):
            instances = instances.get("runningInstances", instances.get("instances", []))
        for inst in instances:
            if inst.get("ip") in local_ips:
                return inst["uuid"]
    except Exception as exc:
        logger.error("Failed to list instances for IP matching: %s", exc)
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

    client = MassedComputeClient()
    if not client.authenticate():
        logger.error("API token validation failed — exiting")
        sys.exit(1)
    logger.info("API token validated ✓")

    idle_seconds = 0
    terminate_after = TERMINATE_MINUTES * 60
    signal_file = "terminate_signal.tmp"

    while True:
        util = get_gpu_utilization()
        api_busy = is_api_busy()

        # Check for immediate termination signal from API
        termination_requested = os.path.exists(signal_file)

        if termination_requested and not api_busy:
            logger.warning("🚀 Immediate termination requested via signal file — triggering now")
            idle_seconds = terminate_after  # Force termination trigger
        elif util < IDLE_THRESHOLD and not api_busy:
            idle_seconds += POLL_INTERVAL
            logger.info(
                "System IDLE: GPU %d%%, API idle — idle for %d/%d min",
                util, idle_seconds // 60, TERMINATE_MINUTES,
            )
        else:
            if idle_seconds > 0:
                reason = "API BUSY" if api_busy else f"GPU active ({util}%)"
                logger.info(
                    "System ACTIVE: %s — idle counter reset (was %d min)",
                    reason, idle_seconds // 60,
                )
            else:
                if api_busy:
                    logger.info("System status: API BUSY, GPU %d%%", util)
                else:
                    logger.info("System status: GPU active (%d%%)", util)
            idle_seconds = 0

        if idle_seconds >= terminate_after:
            logger.warning("🛑 Sustained idle — triggering instance termination")
            uuid = resolve_instance_uuid(client)
            if not uuid:
                logger.error("Cannot resolve instance UUID — skipping termination")
                idle_seconds = 0
                time.sleep(POLL_INTERVAL)
                continue

            if args.dry_run:
                logger.warning("DRY RUN: Would terminate instance %s", uuid)
                idle_seconds = 0
                time.sleep(POLL_INTERVAL)
                continue

            logger.warning("Terminating instance %s NOW", uuid)
            try:
                client.terminate_instance([uuid])
            except Exception as exc:
                logger.error("Terminate API call failed: %s", exc)
            finally:
                sys.stdout.flush()
                sys.stderr.flush()

            logger.info("Goodbye 👋")
            sys.exit(0)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
