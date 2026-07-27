#!/bin/bash

echo "Starting AV Backend on port 3000..."
export PYTHONPATH=$(pwd)
python src/web/app.py &
BACKEND_PID=$!

echo "Server is running."
echo "Access the UI at: http://localhost:3000"
echo "Press Ctrl+C to stop the server."

trap "kill $BACKEND_PID" SIGINT
wait
