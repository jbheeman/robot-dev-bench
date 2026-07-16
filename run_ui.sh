#!/bin/bash

echo "Starting BPS API Backend on port 8000..."
export PYTHONPATH=$(pwd)
python src/web/bps_api.py &
BACKEND_PID=$!

echo "Starting Web UI Frontend on port 4000..."
# Using Python's built-in HTTP server to host the static files
python -m http.server 4000 --directory web-ui/ &
FRONTEND_PID=$!

echo "Both servers are running."
echo "Access the UI at: http://localhost:4000"
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID" SIGINT
wait
