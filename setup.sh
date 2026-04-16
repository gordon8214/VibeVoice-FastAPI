#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "VibeVoice API Setup Script"
echo "============================================================"
echo ""

# ============================================================
# Python Version Check (3.10 - 3.12, prefer 3.12 to match Docker)
# ============================================================
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR_MIN=10
REQUIRED_PYTHON_MINOR_MAX=12

echo "Checking for compatible Python version..."

# Find suitable Python version (prefer newest first)
PYTHON_CMD=""
for version in python3.12 python3.11 python3.10 python3; do
    if command -v $version &> /dev/null; then
        PY_VERSION=$($version -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
        PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

        if [ "$PY_MAJOR" -eq "$REQUIRED_PYTHON_MAJOR" ] && \
           [ "$PY_MINOR" -ge "$REQUIRED_PYTHON_MINOR_MIN" ] && \
           [ "$PY_MINOR" -le "$REQUIRED_PYTHON_MINOR_MAX" ]; then
            PYTHON_CMD=$version
            echo "✓ Found compatible Python: $PYTHON_CMD (version $PY_VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ ERROR: Python 3.10, 3.11, or 3.12 is required."
    echo ""
    echo "Please install Python 3.12 (recommended):"
    echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    echo "  macOS: brew install python@3.12"
    echo ""
    exit 1
fi

# ============================================================
# Create Python Virtual Environment
# ============================================================
echo ""
echo "Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# ============================================================
# Upgrade pip and install build tools
# ============================================================
echo ""
echo "Upgrading pip and installing build tools..."
pip install --upgrade pip wheel setuptools

# ============================================================
# Set environment variables to prevent compilation hangs
# ============================================================
export PIP_ONLY_BINARY=:all:
export MAX_JOBS=2
export MAKEFLAGS="-j2"
echo "✓ Set MAX_JOBS=2 to prevent system hangs if compilation occurs"

# ============================================================
# Detect CUDA and install PyTorch
# ============================================================
echo ""
echo "Detecting CUDA and installing PyTorch..."

PY_VERSION_SHORT=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    echo "✓ Detected CUDA $CUDA_VERSION"

    # Select appropriate PyTorch CUDA index
    if [ "$CUDA_MAJOR" -gt "12" ] || { [ "$CUDA_MAJOR" -eq "12" ] && [ "$CUDA_MINOR" -ge "8" ]; }; then
        TORCH_INDEX="cu128"
    elif [ "$CUDA_MAJOR" -eq "12" ]; then
        TORCH_INDEX="cu124"
    elif [ "$CUDA_MAJOR" -ge "11" ]; then
        TORCH_INDEX="cu118"
    else
        echo "⚠️  WARNING: CUDA $CUDA_VERSION may not be fully supported"
        TORCH_INDEX="cu118"
    fi

    echo "Installing PyTorch 2.8.x with CUDA support ($TORCH_INDEX)..."
    pip install --only-binary=:all: torch==2.8.* torchaudio --index-url https://download.pytorch.org/whl/$TORCH_INDEX

    # Install torchao for INT8/INT4 quantization support
    echo ""
    echo "Installing torchao for quantization support..."
    pip install --only-binary=:all: torchao==0.13.0 || {
        echo "⚠️  WARNING: torchao installation failed. Quantization will not be available."
    }

    # Install flash-attn from pre-built wheel (NO COMPILATION)
    echo ""
    echo "Installing flash-attn from pre-built wheel..."
    echo "Note: This is optional but recommended for better performance"

    # Select flash-attn wheel matching PyTorch 2.8 and detected CUDA
    FLASH_WHEEL_URL=""
    if [ "$TORCH_INDEX" = "cu128" ] || [ "$TORCH_INDEX" = "cu124" ]; then
        FLASH_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-${PY_VERSION_SHORT}-${PY_VERSION_SHORT}-linux_x86_64.whl"
    elif [ "$TORCH_INDEX" = "cu118" ]; then
        FLASH_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu118torch2.8cxx11abiFALSE-${PY_VERSION_SHORT}-${PY_VERSION_SHORT}-linux_x86_64.whl"
    else
        echo "⚠️  WARNING: No pre-built flash-attn wheel for CUDA $CUDA_VERSION"
    fi

    if [ -n "$FLASH_WHEEL_URL" ]; then
        pip install "$FLASH_WHEEL_URL" && \
        echo "✓ flash-attn installed from pre-built wheel" || {
            echo "⚠️  WARNING: flash-attn pre-built wheel installation failed. Will use SDPA fallback."
            echo "Available wheels: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3"
        }
    fi
else
    echo "ℹ️  No CUDA detected, installing CPU-only PyTorch"
    pip install --only-binary=:all: torch==2.8.* torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "Note: flash-attn requires CUDA and will not be installed."
fi

# ============================================================
# Install VibeVoice package in editable mode
# ============================================================
echo ""
echo "Installing VibeVoice package..."
pip install -e .

# ============================================================
# Install API dependencies
# ============================================================
echo ""
echo "Installing API dependencies..."
pip install -r requirements-api.txt

# ============================================================
# Check for ffmpeg (system dependency)
# ============================================================
echo ""
echo "Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  WARNING: ffmpeg not found. Required for audio format conversion."
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo ""
else
    echo "✓ ffmpeg is installed"
fi

# ============================================================
# Create .env file if it doesn't exist
# ============================================================
echo ""
if [ ! -f .env ]; then
    if [ -f env.example ]; then
        cp env.example .env
        echo "✓ Created .env file from env.example"
    else
        echo "# VibeVoice API Configuration" > .env
        echo "VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B" >> .env
        echo "VIBEVOICE_DEVICE=cuda" >> .env
        echo "✓ Created basic .env file"
    fi
    echo "Please review and edit .env file as needed"
else
    echo "ℹ️  .env file already exists, skipping creation"
fi

# ============================================================
# Setup Complete
# ============================================================
echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review and edit .env file if needed"
echo "  2. Start the API server:"
echo "     ./start.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8001"
echo ""
echo "The model will be downloaded automatically on first run."
echo "============================================================"
