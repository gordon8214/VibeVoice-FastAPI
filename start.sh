#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=3
# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8001}
WORKERS=${API_WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-info}

echo "============================================================"
echo "Starting VibeVoice API Server"
echo "============================================================"
echo ""
echo "Server: http://$HOST:$PORT"
echo "API Docs: http://$HOST:$PORT/docs"
echo "Workers: $WORKERS"
echo "Log Level: $LOG_LEVEL"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

# Start the server
uvicorn api.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level $LOG_LEVEL


