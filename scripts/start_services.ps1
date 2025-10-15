<#!
.SYNOPSIS
    Start both the synchronous and asynchronous Energy Fault Detector APIs on Windows.
.DESCRIPTION
    Launches two Uvicorn processes: `energy_fault_detector.api.app:app` and
    `energy_fault_detector.api.prediction_api:app`. The service configuration path
    is validated and exported via the `EFD_SERVICE_CONFIG` environment variable.
#>

[CmdletBinding()]
param(
    [string]
    $ConfigPath = "energy_fault_detector/api/service_config.yaml",

    [string]
    $ApiHost = "0.0.0.0",

    [int]
    $ApiPort = 8000,

    [string]
    $PredictionHost = "0.0.0.0",

    [int]
    $PredictionPort = 8001
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path (Join-Path $scriptDir '..')
Set-Location $projectRoot

if (-not (Test-Path -Path $ConfigPath -PathType Leaf)) {
    throw "Configuration file '$ConfigPath' was not found."
}

$resolvedConfigPath = (Resolve-Path $ConfigPath).Path
$env:EFD_SERVICE_CONFIG = $resolvedConfigPath

$uvicorn = (Get-Command uvicorn).Source

$apiArgs = @(
    'energy_fault_detector.api.app:app',
    '--host', $ApiHost,
    '--port', $ApiPort
)

$predictionArgs = @(
    'energy_fault_detector.api.prediction_api:app',
    '--host', $PredictionHost,
    '--port', $PredictionPort
)

Write-Host "Starting synchronous API on $ApiHost`:$ApiPort"
$apiProcess = Start-Process -FilePath $uvicorn -ArgumentList $apiArgs -PassThru -NoNewWindow

Write-Host "Starting asynchronous prediction API on $PredictionHost`:$PredictionPort"
$predictionProcess = Start-Process -FilePath $uvicorn -ArgumentList $predictionArgs -PassThru -NoNewWindow

try {
    Write-Host 'Services are running. Press Ctrl+C to stop.'
    Wait-Process -Id @($apiProcess.Id, $predictionProcess.Id)
}
finally {
    foreach ($proc in @($apiProcess, $predictionProcess)) {
        if ($proc -and -not $proc.HasExited) {
            try {
                Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
            } catch {
                # Ignore shutdown errors
            }
        }
    }
}
