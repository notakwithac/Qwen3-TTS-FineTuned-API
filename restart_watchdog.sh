#!/bin/bash
# Script to safely restart the GPU Idle Watchdog.

echo "🔍 Finding existing gpu_idle_watchdog.py processes..."
PIDS=$(pgrep -f "gpu_idle_watchdog.py")

if [ -n "$PIDS" ]; then
    echo "🛑 Killing PIDs: $PIDS"
    kill $PIDS
    sleep 2
else
    echo "✅ No existing watchdog found."
fi

echo "🚀 Starting fresh GPU Idle Watchdog..."
mkdir -p logs
nohup python gpu_idle_watchdog.py >> logs/watchdog.log 2>&1 &

NEW_PID=$(pgrep -f "gpu_idle_watchdog.py")
echo "✅ Watchdog started successfully with PID: $NEW_PID"
echo "📄 Logs available at: logs/watchdog.log"
