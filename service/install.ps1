#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Installs the VibeVoice FastAPI TTS Windows service.

.DESCRIPTION
    Installs the service and configures failure recovery.

.PARAMETER Force
    Skip confirmation prompts (for automation).

.EXAMPLE
    .\install.ps1
    Interactive installation with prompts.

.EXAMPLE
    .\install.ps1 -Force
    Automated installation without prompts.
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Configuration
$ServiceName = "VibeVoiceFastAPI"
$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir

# Change to script directory
Set-Location $ScriptDir

# Check if service already exists
$existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existingService) {
    if ($PSCmdlet.ShouldProcess($ServiceName, "Stop and uninstall existing service")) {
        Write-Host "Service already exists. Stopping and uninstalling..."
        & .\VibeVoiceFastAPI.exe stop 2>$null
        Start-Sleep -Seconds 2
        & .\VibeVoiceFastAPI.exe uninstall
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to uninstall existing service" -ForegroundColor Red
            exit 1
        }

        # Wait for service to be fully removed from SCM
        Write-Host "Waiting for service removal to complete..."
        $timeout = 30
        $elapsed = 0
        while ((Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) -and ($elapsed -lt $timeout)) {
            Start-Sleep -Seconds 1
            $elapsed++
        }
        if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
            Write-Host "Timeout waiting for service removal" -ForegroundColor Red
            exit 1
        }
    }
}

if ($PSCmdlet.ShouldProcess($ServiceName, "Install service")) {
    Write-Host "Installing VibeVoice FastAPI TTS service..."
    & .\VibeVoiceFastAPI.exe install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install service" -ForegroundColor Red
        exit 1
    }
}

if ($PSCmdlet.ShouldProcess($ServiceName, "Configure failure recovery")) {
    Write-Host "Configuring failure recovery..."
    # Configure SCM to: restart service (5s, 30s, 60s delays)
    & sc.exe failure $ServiceName reset= 3600 actions= restart/5000/restart/30000/restart/60000
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to configure failure recovery" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Service installed successfully." -ForegroundColor Green

if ($PSCmdlet.ShouldProcess($ServiceName, "Start service")) {
    Write-Host "Starting service..."
    Write-Host "Note: First startup will download the model (~14GB) and may take several minutes." -ForegroundColor Yellow
    try {
        Start-Service -Name $ServiceName -ErrorAction Stop
        Start-Sleep -Seconds 5
        $service = Get-Service -Name $ServiceName
        if ($service.Status -eq 'Running') {
            Write-Host "Service is running." -ForegroundColor Green
            Write-Host ""
            Write-Host "The model is loading. Check status with:" -ForegroundColor Cyan
            Write-Host "  curl http://localhost:8881/v1/vibevoice/health"
        } else {
            Write-Host "Warning: Service status is $($service.Status)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Warning: Service installed but failed to start: $_" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Check status:    sc.exe query $ServiceName"
Write-Host "  View logs:       Get-Content `"$ProjectRoot\logs\vibevoice.log`""
Write-Host "  Stop service:    Stop-Service $ServiceName"
Write-Host "  Start service:   Start-Service $ServiceName"
Write-Host "  Uninstall:       .\VibeVoiceFastAPI.exe uninstall"
