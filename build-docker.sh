#!/bin/bash
set -e

echo "=========================================="
echo "Building VibeVoice API Docker Image"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ Warning: NVIDIA GPU not detected. Container will use CPU."
fi

echo ""
echo "Building Docker image..."
docker build -t vibevoice-api:latest .

echo ""
echo "=========================================="
echo "✓ Build complete!"
echo "=========================================="
echo ""
echo "To run with docker-compose:"
echo "  1. Copy docker-env.example to .env"
echo "  2. Edit .env with your configuration"
echo "  3. Run: docker-compose up -d"
echo ""
echo "To run with docker:"
echo "  docker run -d --gpus all -p 8000:8000 \\"
echo "    -v /path/to/voices:/app/voices:ro \\"
echo "    -v ~/.cache/huggingface:/root/.cache/huggingface:rw \\"
echo "    vibevoice-api:latest"
echo ""

