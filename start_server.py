"""Launcher script for VibeVoice API server. Used by start.bat on Windows."""

import os
import subprocess
import sys
from pathlib import Path

def load_env(env_path):
    """Load .env file into os.environ, handling JSON values and comments."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ[key] = value.strip()


def main():
    script_dir = Path(__file__).resolve().parent
    load_env(script_dir / ".env")

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = os.environ.get("API_PORT", "8001")
    workers = os.environ.get("API_WORKERS", "1")
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    print("=" * 60)
    print("Starting VibeVoice API Server")
    print("=" * 60)
    print()
    print(f"Server:    http://{host}:{port}")
    print(f"API Docs:  http://{host}:{port}/docs")
    print(f"Workers:   {workers}")
    print(f"Log Level: {log_level}")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    sys.exit(subprocess.call([
        "uvicorn", "api.main:app",
        "--host", host,
        "--port", port,
        "--workers", workers,
        "--log-level", log_level,
    ]))


if __name__ == "__main__":
    main()
