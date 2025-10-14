<#
.SYNOPSIS
    Launches the Energy Fault Detector FastAPI service on Windows.
.DESCRIPTION
    Mirrors scripts/run_api.sh by changing to the repository root, validating
    the service configuration path and starting Uvicorn. Additional arguments
    after the configuration path are forwarded to Uvicorn.
#>

param(
    [string]
    $ConfigPath = "energy_fault_detector/api/service_config.yaml",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]
    $UvicornArgs
)

$ErrorActionPreference = 'Stop'

# Change to the repository root (the directory containing this script).
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path (Join-Path $scriptDir '..')
Set-Location $projectRoot

if (-not (Test-Path -Path $ConfigPath -PathType Leaf)) {
    Write-Error "Configuration file '$ConfigPath' was not found."
}

$resolvedConfigPath = Resolve-Path $ConfigPath
$env:EFD_SERVICE_CONFIG = $resolvedConfigPath.Path

# Start the FastAPI application with Uvicorn.
& uvicorn 'energy_fault_detector.api.app:app' @UvicornArgs
