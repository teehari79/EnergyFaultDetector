#!/usr/bin/env pwsh
# Utility script to launch the Energy Fault Detector stack in Docker (PowerShell edition).

param(
    [switch]$Help,
    [string]$ProjectName,
    [string]$ComposeFile,
    [string]$MongoContainerName,
    [Nullable[int]]$MongoPort,
    [string]$MongoDataDir,
    [string]$MongoLogDir,
    [string]$MongoRootUser,
    [string]$MongoRootPassword,
    [string]$MongoAppDb,
    [string]$MongoAppUser,
    [string]$MongoAppPassword,
    [string]$ApiContainerName,
    [Nullable[int]]$ApiPort,
    [Nullable[int]]$PredictionPort,
    [string]$ResultsDir,
    [string]$ModelsDir,
    [string]$WebServiceContainerName,
    [Nullable[int]]$WebServicePort,
    [string]$WebUiContainerName,
    [Nullable[int]]$WebUiPort,
    [string[]]$ComposeArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Show-Usage {
    @'
Usage: ./scripts/setup_docker_local.ps1 [options]

Bootstraps the Energy Fault Detector API, Node service, React UI, and MongoDB
using Docker Compose. Extra docker compose arguments can be supplied via
-ComposeArgs.

Parameters (environment variables may also be used):
  -Help                         Show this help message and exit
  -ProjectName <name>           Docker Compose project name (default: energy-fault-detector)
  -ComposeFile <path>           Path to the docker compose file (default: ./docker/docker-compose.local.yml)
  -MongoContainerName <name>    MongoDB container name (default: energy-fault-detector-mongo)
  -MongoPort <port>             Host port for MongoDB (default: 27017)
  -MongoDataDir <path>          Directory for MongoDB data volume (default: ./.mongodb/data)
  -MongoLogDir <path>           Directory for MongoDB logs (default: ./.mongodb/logs)
  -MongoRootUser <user>         MongoDB root username (default: efd_root)
  -MongoRootPassword <pass>     MongoDB root password (default: efd_root_password)
  -MongoAppDb <name>            MongoDB application database (default: energy_fault_detector)
  -MongoAppUser <user>          MongoDB application username (default: efd_app)
  -MongoAppPassword <pass>      MongoDB application password (default: efd_app_password)
  -ApiContainerName <name>      API container name (default: energy-fault-detector-api)
  -ApiPort <port>               Host port for the synchronous FastAPI service (default: 8000)
  -PredictionPort <port>        Host port for the asynchronous prediction API (default: 8001)
  -ResultsDir <path>            Directory to persist API results (default: ./results)
  -ModelsDir <path>             Directory containing trained models (default: ./models)
  -WebServiceContainerName <name>  Node service container name (default: energy-fault-detector-web-service)
  -WebServicePort <port>        Host port for the Node service (default: 4000)
  -WebUiContainerName <name>    React UI container name (default: energy-fault-detector-web-ui)
  -WebUiPort <port>             Host port for the React UI (default: 5173)
  -ComposeArgs <array>          Additional arguments passed to 'docker compose up'

Environment overrides:
  EFD_COMPOSE_PROJECT, EFD_COMPOSE_FILE, MONGO_CONTAINER_NAME, MONGO_PORT,
  MONGO_DATA_DIR, MONGO_LOG_DIR, MONGO_INITDB_ROOT_USERNAME, MONGO_INITDB_ROOT_PASSWORD,
  MONGO_APP_DB, MONGO_APP_USER, MONGO_APP_PASSWORD, API_CONTAINER_NAME,
  API_PORT, PREDICTION_PORT, HOST_RESULTS_DIR, HOST_MODELS_DIR,
  WEB_SERVICE_CONTAINER_NAME, WEB_SERVICE_PORT,
  WEB_UI_CONTAINER_NAME, WEB_UI_PORT
'@
}

function Get-ConfigValue {
    param(
        [object]$ParameterValue,
        [string]$EnvName,
        [string]$DefaultValue
    )

    if ($null -ne $ParameterValue) {
        $stringValue = [string]$ParameterValue
        if (-not [string]::IsNullOrEmpty($stringValue)) {
            return $stringValue
        }
    }

    $envValue = [Environment]::GetEnvironmentVariable($EnvName)
    if (-not [string]::IsNullOrEmpty($envValue)) {
        return $envValue
    }

    return $DefaultValue
}

if ($Help) {
    Show-Usage
    exit 0
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir

$DefaultComposeFile = Join-Path -Path $RootDir -ChildPath 'docker'
$DefaultComposeFile = Join-Path -Path $DefaultComposeFile -ChildPath 'docker-compose.local.yml'
$DefaultMongoDataDir = Join-Path -Path $RootDir -ChildPath '.mongodb'
$DefaultMongoDataDir = Join-Path -Path $DefaultMongoDataDir -ChildPath 'data'
$DefaultMongoLogDir = Join-Path -Path $RootDir -ChildPath '.mongodb'
$DefaultMongoLogDir = Join-Path -Path $DefaultMongoLogDir -ChildPath 'logs'
$DefaultResultsDir = Join-Path -Path $RootDir -ChildPath 'results'
$DefaultModelsDir = Join-Path -Path $RootDir -ChildPath 'models'

$ProjectName = Get-ConfigValue -ParameterValue $ProjectName -EnvName 'EFD_COMPOSE_PROJECT' -DefaultValue 'energy-fault-detector'
$ComposeFile = Get-ConfigValue -ParameterValue $ComposeFile -EnvName 'EFD_COMPOSE_FILE' -DefaultValue $DefaultComposeFile
$MongoContainerName = Get-ConfigValue -ParameterValue $MongoContainerName -EnvName 'MONGO_CONTAINER_NAME' -DefaultValue 'energy-fault-detector-mongo'
$MongoPort = [int](Get-ConfigValue -ParameterValue $MongoPort -EnvName 'MONGO_PORT' -DefaultValue '27017')
$MongoDataDir = Get-ConfigValue -ParameterValue $MongoDataDir -EnvName 'MONGO_DATA_DIR' -DefaultValue $DefaultMongoDataDir
$MongoLogDir = Get-ConfigValue -ParameterValue $MongoLogDir -EnvName 'MONGO_LOG_DIR' -DefaultValue $DefaultMongoLogDir
${'$'}defaultRootUser = [Environment]::GetEnvironmentVariable('MONGO_INITDB_ROOT_USERNAME')
if ([string]::IsNullOrEmpty(${ '$'}defaultRootUser)) {
    ${'$'}defaultRootUser = [Environment]::GetEnvironmentVariable('MONGO_ROOT_USER')
}
if ([string]::IsNullOrEmpty(${ '$'}defaultRootUser)) {
    ${'$'}defaultRootUser = 'efd_root'
}
${'$'}MongoRootUser = Get-ConfigValue -ParameterValue ${'$'}MongoRootUser -EnvName 'MONGO_INITDB_ROOT_USERNAME' -DefaultValue ${'$'}defaultRootUser
${'$'}defaultRootPassword = [Environment]::GetEnvironmentVariable('MONGO_INITDB_ROOT_PASSWORD')
if ([string]::IsNullOrEmpty(${ '$'}defaultRootPassword)) {
    ${'$'}defaultRootPassword = [Environment]::GetEnvironmentVariable('MONGO_ROOT_PASSWORD')
}
if ([string]::IsNullOrEmpty(${ '$'}defaultRootPassword)) {
    ${'$'}defaultRootPassword = 'efd_root_password'
}
${'$'}MongoRootPassword = Get-ConfigValue -ParameterValue ${'$'}MongoRootPassword -EnvName 'MONGO_INITDB_ROOT_PASSWORD' -DefaultValue ${'$'}defaultRootPassword
$MongoAppDb = Get-ConfigValue -ParameterValue $MongoAppDb -EnvName 'MONGO_APP_DB' -DefaultValue 'energy_fault_detector'
$MongoAppUser = Get-ConfigValue -ParameterValue $MongoAppUser -EnvName 'MONGO_APP_USER' -DefaultValue 'efd_app'
$MongoAppPassword = Get-ConfigValue -ParameterValue $MongoAppPassword -EnvName 'MONGO_APP_PASSWORD' -DefaultValue 'efd_app_password'
$ApiContainerName = Get-ConfigValue -ParameterValue $ApiContainerName -EnvName 'API_CONTAINER_NAME' -DefaultValue 'energy-fault-detector-api'
$ApiPort = [int](Get-ConfigValue -ParameterValue $ApiPort -EnvName 'API_PORT' -DefaultValue '8000')
$PredictionPort = [int](Get-ConfigValue -ParameterValue $PredictionPort -EnvName 'PREDICTION_PORT' -DefaultValue '8001')
$ResultsDir = Get-ConfigValue -ParameterValue $ResultsDir -EnvName 'HOST_RESULTS_DIR' -DefaultValue $DefaultResultsDir
$ModelsDir = Get-ConfigValue -ParameterValue $ModelsDir -EnvName 'HOST_MODELS_DIR' -DefaultValue $DefaultModelsDir
$WebServiceContainerName = Get-ConfigValue -ParameterValue $WebServiceContainerName -EnvName 'WEB_SERVICE_CONTAINER_NAME' -DefaultValue 'energy-fault-detector-web-service'
$WebServicePort = [int](Get-ConfigValue -ParameterValue $WebServicePort -EnvName 'WEB_SERVICE_PORT' -DefaultValue '4000')
$WebUiContainerName = Get-ConfigValue -ParameterValue $WebUiContainerName -EnvName 'WEB_UI_CONTAINER_NAME' -DefaultValue 'energy-fault-detector-web-ui'
$WebUiPort = [int](Get-ConfigValue -ParameterValue $WebUiPort -EnvName 'WEB_UI_PORT' -DefaultValue '5173')

$ComposeFile = [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($ComposeFile))
$MongoDataDir = [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($MongoDataDir))
$MongoLogDir = [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($MongoLogDir))
$ResultsDir = [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($ResultsDir))
$ModelsDir = [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($ModelsDir))

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error 'Error: docker is required but was not found in PATH.'
    exit 1
}

$composeCommand = $null
if (& docker compose version *> $null) {
    $composeCommand = @('docker', 'compose')
}
elseif (Get-Command docker-compose -ErrorAction SilentlyContinue) {
    $composeCommand = @('docker-compose')
}
else {
    Write-Error 'Error: docker compose plugin or docker-compose binary is required.'
    exit 1
}

if (-not (Test-Path -Path $ComposeFile -PathType Leaf)) {
    Write-Error "Error: docker compose file '$ComposeFile' not found."
    exit 1
}

[System.IO.Directory]::CreateDirectory($MongoDataDir) | Out-Null
[System.IO.Directory]::CreateDirectory($MongoLogDir) | Out-Null
[System.IO.Directory]::CreateDirectory($ResultsDir) | Out-Null
[System.IO.Directory]::CreateDirectory($ModelsDir) | Out-Null

Write-Host "Using docker compose file: $ComposeFile"
Write-Host "Project name: $ProjectName"
Write-Host "MongoDB container: $MongoContainerName (port $MongoPort)"
Write-Host "API container: $ApiContainerName (ports $ApiPort/$PredictionPort)"
Write-Host "Node service container: $WebServiceContainerName (port $WebServicePort)"
Write-Host "Web UI container: $WebUiContainerName (port $WebUiPort)"

if (-not $ComposeArgs -or $ComposeArgs.Count -eq 0) {
    $ComposeArgs = @('--build', '-d')
}

$previousEnv = @{}
$envKeys = @(
    'COMPOSE_FILE','COMPOSE_PROJECT_NAME','MONGO_CONTAINER_NAME','MONGO_PORT',
    'MONGO_DATA_DIR','MONGO_LOG_DIR','MONGO_ROOT_USER','MONGO_INITDB_ROOT_USERNAME','MONGO_ROOT_PASSWORD','MONGO_INITDB_ROOT_PASSWORD',
    'MONGO_APP_DB','MONGO_APP_USER','MONGO_APP_PASSWORD','API_CONTAINER_NAME',
    'API_PORT','PREDICTION_PORT','HOST_RESULTS_DIR','HOST_MODELS_DIR',
    'WEB_SERVICE_CONTAINER_NAME','WEB_SERVICE_PORT','WEB_UI_CONTAINER_NAME','WEB_UI_PORT'
)

foreach ($key in $envKeys) {
    $previousEnv[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}

try {
    $env:COMPOSE_FILE = $ComposeFile
    $env:COMPOSE_PROJECT_NAME = $ProjectName
    $env:MONGO_CONTAINER_NAME = $MongoContainerName
    $env:MONGO_PORT = [string]$MongoPort
    $env:MONGO_DATA_DIR = $MongoDataDir
    $env:MONGO_LOG_DIR = $MongoLogDir
    $env:MONGO_ROOT_USER = $MongoRootUser
    $env:MONGO_INITDB_ROOT_USERNAME = $MongoRootUser
    $env:MONGO_ROOT_PASSWORD = $MongoRootPassword
    $env:MONGO_INITDB_ROOT_PASSWORD = $MongoRootPassword
    $env:MONGO_APP_DB = $MongoAppDb
    $env:MONGO_APP_USER = $MongoAppUser
    $env:MONGO_APP_PASSWORD = $MongoAppPassword
    $env:API_CONTAINER_NAME = $ApiContainerName
    $env:API_PORT = [string]$ApiPort
    $env:PREDICTION_PORT = [string]$PredictionPort
    $env:HOST_RESULTS_DIR = $ResultsDir
    $env:HOST_MODELS_DIR = $ModelsDir
    $env:WEB_SERVICE_CONTAINER_NAME = $WebServiceContainerName
    $env:WEB_SERVICE_PORT = [string]$WebServicePort
    $env:WEB_UI_CONTAINER_NAME = $WebUiContainerName
    $env:WEB_UI_PORT = [string]$WebUiPort

    & $composeCommand up @ComposeArgs
    Write-Host
    & $composeCommand ps
}
finally {
    foreach ($key in $envKeys) {
        if ($null -eq $previousEnv[$key]) {
            Remove-Item -Path "Env:$key" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($key, $previousEnv[$key], 'Process')
        }
    }
}

Write-Host
Write-Host 'Stack is up! Services are reachable at:'
Write-Host "  FastAPI (sync):        http://localhost:$ApiPort"
Write-Host "  FastAPI (prediction):  http://localhost:$PredictionPort"
Write-Host "  Node proxy service:    http://localhost:$WebServicePort"
Write-Host "  React UI:              http://localhost:$WebUiPort"
Write-Host "  MongoDB:               mongodb://localhost:$MongoPort"
Write-Host
Write-Host "Use '$($composeCommand -join ' ') -p $ProjectName down' to stop the stack."
