# VibeVoice FastAPI Service Startup Script
# Called by WinSW to start the TTS service

$ErrorActionPreference = "Stop"

# Get project root (parent of service directory)
$projectRoot = Split-Path -Parent $PSScriptRoot

# Helper function for logging
function Write-ServiceLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "[$timestamp] [$Level] $Message"
}

Write-ServiceLog "Starting VibeVoice FastAPI service..."
Write-ServiceLog "Project root: $projectRoot"

# Set environment variables
$env:PYTHONUTF8 = "1"
$env:PROJECT_ROOT = $projectRoot

# Add ffmpeg to PATH (required for audio processing)
$env:PATH = "C:\Users\Gordon\bin;$env:PATH"

# Python executable in venv
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"

Write-ServiceLog "Python executable: $pythonExe"

# Validate venv exists
if (-not (Test-Path $pythonExe)) {
    Write-ServiceLog "ERROR: Python not found at: $pythonExe" "ERROR"
    Write-ServiceLog "Please ensure the virtual environment exists." "ERROR"
    exit 1
}

# Change to project directory
Set-Location $projectRoot

# Validate main module exists
$mainModule = Join-Path $projectRoot "api\main.py"
if (-not (Test-Path $mainModule)) {
    Write-ServiceLog "ERROR: Main module not found at: $mainModule" "ERROR"
    exit 1
}

Write-ServiceLog "Starting uvicorn on port 8881..."

# Start uvicorn using the venv's python
# Using -u for unbuffered output so logs appear immediately
try {
    & $pythonExe -u -m uvicorn api.main:app --host 0.0.0.0 --port 8881 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-ServiceLog "Uvicorn exited with code: $exitCode" "ERROR"
        exit $exitCode
    }
} catch {
    Write-ServiceLog "Failed to start uvicorn: $_" "ERROR"
    exit 1
}
