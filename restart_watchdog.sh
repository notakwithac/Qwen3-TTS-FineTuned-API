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
nohup python gpu_idle_watchdog.py 2>&1 | tee -a logs/watchdog.log &
sleep 3

NEW_PID=$(pgrep -f "gpu_idle_watchdog.py")
if [ -n "$NEW_PID" ]; then
    echo "✅ Watchdog started successfully with PID: $NEW_PID"
    echo "📄 Logs mirrored to stdout AND logs/watchdog.log"
else
    echo "❌ ERROR: Watchdog failed to start! Check logs/watchdog.log:"
    tail -20 logs/watchdog.log
fi
