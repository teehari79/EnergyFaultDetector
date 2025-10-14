<#
.SYNOPSIS
    Sends a Farm C prediction request to the Energy Fault Detector API on Windows.
.DESCRIPTION
    Mirrors scripts/predict_farm_c.sh by posting the JSON payload to the
    configured API endpoint. The payload is validated before submission and the
    formatted JSON response is printed to the console.
#>

param(
    [string]
    $ApiUrl = 'http://127.0.0.1:8000/predict',

    [string]
    $PayloadFile = 'docs/examples/farm_c_prediction.json'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $PayloadFile -PathType Leaf)) {
    Write-Error "Payload file '$PayloadFile' was not found."
}

$payload = Get-Content -Path $PayloadFile -Raw

$response = Invoke-RestMethod -Uri $ApiUrl -Method Post -ContentType 'application/json' -Body $payload

$response | ConvertTo-Json -Depth 10
