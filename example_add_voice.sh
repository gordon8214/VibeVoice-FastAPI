#!/bin/bash
# Example script to add a custom voice to VibeVoice API

set -e

echo "=========================================="
echo "VibeVoice - Add Custom Voice Example"
echo "=========================================="
echo ""

# Check if voices directory exists
VOICES_DIR="${VOICES_DIR:-demo/voices}"

if [ ! -d "$VOICES_DIR" ]; then
    echo "Error: Voices directory not found at $VOICES_DIR"
    echo "Please set VOICES_DIR environment variable or create the directory"
    exit 1
fi

echo "Voices directory: $VOICES_DIR"
echo ""

# Example: Download a sample voice (you would use your own recording)
echo "Step 1: Prepare your voice sample"
echo "  - Record 5-10 seconds of clear speech"
echo "  - Save as WAV, MP3, or other supported format"
echo "  - Name it descriptively (e.g., en-YourName_gender.wav)"
echo ""

# Check if user provided a voice file
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path-to-voice-file> [preset-name]"
    echo ""
    echo "Example:"
    echo "  $0 my-recording.wav en-John_man"
    echo "  $0 voice.mp3 customer-service"
    echo ""
    echo "Current voices in $VOICES_DIR:"
    ls -1 "$VOICES_DIR" | grep -E '\.(wav|mp3|flac|ogg|m4a|aac)$' || echo "  (none found)"
    exit 0
fi

VOICE_FILE="$1"
PRESET_NAME="${2:-$(basename "$VOICE_FILE" | sed 's/\.[^.]*$//')}"

# Check if voice file exists
if [ ! -f "$VOICE_FILE" ]; then
    echo "Error: Voice file not found: $VOICE_FILE"
    exit 1
fi

echo "Step 2: Copy voice to voices directory"
echo "  Source: $VOICE_FILE"
echo "  Preset name: $PRESET_NAME"
echo ""

# Get file extension
EXT="${VOICE_FILE##*.}"
TARGET_FILE="$VOICES_DIR/${PRESET_NAME}.${EXT}"

# Copy the file
cp "$VOICE_FILE" "$TARGET_FILE"
echo "✓ Copied to: $TARGET_FILE"
echo ""

# Optional: Convert to optimal format using ffmpeg if available
if command -v ffmpeg &> /dev/null; then
    echo "Step 3: Optimize audio (optional)"
    OPTIMIZED_FILE="$VOICES_DIR/${PRESET_NAME}.wav"
    
    if [ "$TARGET_FILE" != "$OPTIMIZED_FILE" ]; then
        echo "  Converting to WAV format at 24kHz mono..."
        ffmpeg -i "$TARGET_FILE" -ar 24000 -ac 1 -y "$OPTIMIZED_FILE" 2>&1 | grep -E '(Duration|Stream|Output)' || true
        
        # Remove original if conversion successful
        if [ -f "$OPTIMIZED_FILE" ]; then
            rm "$TARGET_FILE"
            TARGET_FILE="$OPTIMIZED_FILE"
            echo "✓ Optimized: $OPTIMIZED_FILE"
        fi
    else
        echo "  Already in WAV format, skipping conversion"
    fi
    echo ""
else
    echo "Step 3: Optimize audio (skipped - ffmpeg not found)"
    echo "  Install ffmpeg for automatic audio optimization"
    echo ""
fi

echo "Step 4: Verify the file"
ls -lh "$TARGET_FILE"
if command -v file &> /dev/null; then
    file "$TARGET_FILE"
fi
echo ""

echo "Step 5: Restart the API server"
echo "  Run: ./start.sh"
echo ""

echo "Step 6: Test your new voice"
echo "  List voices:"
echo "    curl http://localhost:8001/v1/audio/voices?show_all=true | jq '.voices[] | select(.name == \"$PRESET_NAME\")'"
echo ""
echo "  Generate speech:"
echo "    curl -X POST http://localhost:8001/v1/audio/speech \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"tts-1\",\"input\":\"Hello from my custom voice!\",\"voice\":\"$PRESET_NAME\",\"response_format\":\"mp3\"}' \\"
echo "      --output test_${PRESET_NAME}.mp3"
echo ""

echo "=========================================="
echo "✓ Voice added successfully!"
echo "=========================================="
echo ""
echo "Preset name: $PRESET_NAME"
echo "File: $TARGET_FILE"
echo ""
echo "Don't forget to restart the server: ./start.sh"


