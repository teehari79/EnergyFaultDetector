<#!
.SYNOPSIS
    Start all local webhook receiver scripts on Windows.
.DESCRIPTION
    Launches each webhook helper in the `scripts` directory so that webhook
    payloads emitted by the asynchronous prediction API can be inspected.
    Processes are terminated automatically when this script exits.
#>

[CmdletBinding()]
param(
    [string[]]
    $AdditionalArgs = @()
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

$webhookScripts = @(
    'webhook_anomalies.py',
    'webhook_events.py',
    'webhook_criticality.py',
    'webhook_root_cause.py',
    'webhook_narrative.py'
)

$python = (Get-Command python).Source
$processes = @()

foreach ($script in $webhookScripts) {
    Write-Host "Starting $script"
    $argumentList = @($script)
    if ($AdditionalArgs.Count -gt 0) {
        $argumentList += $AdditionalArgs
    }

    $processes += Start-Process -FilePath $python -ArgumentList $argumentList -PassThru -NoNewWindow
    Start-Sleep -Milliseconds 200
}

try {
    Write-Host 'Webhook listeners are running. Press Ctrl+C to stop.'
    Wait-Process -Id ($processes | ForEach-Object { $_.Id })
}
finally {
    foreach ($proc in $processes) {
        if ($proc -and -not $proc.HasExited) {
            try {
                Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
            } catch {
                # Ignore shutdown errors
            }
        }
    }
}
