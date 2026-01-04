#!/usr/bin/env bash
set -e

echo "▶ Starting Python livestream..."
python3 /app/livestream.py &
PY_PID=$!
echo "python PID=$PY_PID"

trap 'echo "Stopping..."; kill $PY_PID ; wait; exit 0' SIGTERM SIGINT

# Keep container alive as long as nginx is alive
wait $PY_PID

