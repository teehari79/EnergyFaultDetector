#!/usr/bin/env pwsh
# Utility script to launch a MongoDB instance for local development (PowerShell edition).

param(
    [switch]$Help,
    [string]$ContainerName,
    [Nullable[int]]$Port,
    [string]$DataDir,
    [string]$LogDir,
    [string]$RootUser,
    [string]$RootPassword,
    [string]$AppDb,
    [string]$AppUser,
    [string]$AppPassword
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Show-Usage {
    @'
Usage: ./scripts/setup_mongodb_local.ps1 [-Help] [-ContainerName <name>] [-Port <port>] [-DataDir <path>] [-LogDir <path>] \
                                         [-RootUser <user>] [-RootPassword <pass>] [-AppDb <db>] [-AppUser <user>] [-AppPassword <pass>]

Bootstraps a MongoDB instance in Docker for local development and testing.

Parameters (environment variables may also be used):
  -Help                   Show this help message and exit
  -ContainerName          Name for the MongoDB container (default: energy-fault-detector-mongo)
  -Port                   Host port to expose MongoDB on (default: 27017)
  -DataDir                Host directory to persist MongoDB data (default: ../.mongodb/data)
  -LogDir                 Host directory to persist MongoDB logs (default: ../.mongodb/logs)
  -RootUser               MongoDB root username (default: efd_root)
  -RootPassword           MongoDB root password (default: efd_root_password)
  -AppDb                  Application database name (default: energy_fault_detector)
  -AppUser                Application database user (default: efd_app)
  -AppPassword            Application user password (default: efd_app_password)

Environment overrides:
  MONGO_CONTAINER_NAME, MONGO_PORT, MONGO_DATA_DIR, MONGO_LOG_DIR,
  MONGO_INITDB_ROOT_USERNAME, MONGO_INITDB_ROOT_PASSWORD,
  MONGO_APP_DB, MONGO_APP_USER, MONGO_APP_PASSWORD
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

$DefaultDataDir = Join-Path -Path $RootDir -ChildPath '.mongodb'
$DefaultDataDir = Join-Path -Path $DefaultDataDir -ChildPath 'data'
$DefaultLogDir = Join-Path -Path $RootDir -ChildPath '.mongodb'
$DefaultLogDir = Join-Path -Path $DefaultLogDir -ChildPath 'logs'

$ContainerName = Get-ConfigValue -ParameterValue $ContainerName -EnvName 'MONGO_CONTAINER_NAME' -DefaultValue 'energy-fault-detector-mongo'
$Port = [int](Get-ConfigValue -ParameterValue $Port -EnvName 'MONGO_PORT' -DefaultValue '27017')
$DataDir = Get-ConfigValue -ParameterValue $DataDir -EnvName 'MONGO_DATA_DIR' -DefaultValue $DefaultDataDir
$LogDir = Get-ConfigValue -ParameterValue $LogDir -EnvName 'MONGO_LOG_DIR' -DefaultValue $DefaultLogDir
$RootUser = Get-ConfigValue -ParameterValue $RootUser -EnvName 'MONGO_INITDB_ROOT_USERNAME' -DefaultValue 'efd_root'
$RootPassword = Get-ConfigValue -ParameterValue $RootPassword -EnvName 'MONGO_INITDB_ROOT_PASSWORD' -DefaultValue 'efd_root_password'
$AppDb = Get-ConfigValue -ParameterValue $AppDb -EnvName 'MONGO_APP_DB' -DefaultValue 'energy_fault_detector'
$AppUser = Get-ConfigValue -ParameterValue $AppUser -EnvName 'MONGO_APP_USER' -DefaultValue 'efd_app'
$AppPassword = Get-ConfigValue -ParameterValue $AppPassword -EnvName 'MONGO_APP_PASSWORD' -DefaultValue 'efd_app_password'

$DataDir = [Environment]::ExpandEnvironmentVariables($DataDir)
$LogDir = [Environment]::ExpandEnvironmentVariables($LogDir)
$DataDir = (New-Object System.IO.DirectoryInfo($DataDir)).FullName
$LogDir = (New-Object System.IO.DirectoryInfo($LogDir)).FullName

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error 'Error: docker is required but was not found in PATH.'
    exit 1
}

Write-Host "Using container name: $ContainerName"
Write-Host "Exposing MongoDB on host port: $Port"
Write-Host "Persisting data in: $DataDir"
Write-Host "Persisting logs in: $LogDir"
Write-Host "Root user: $RootUser"
Write-Host "Application database: $AppDb"
Write-Host "Application user: $AppUser"

[System.IO.Directory]::CreateDirectory($DataDir) | Out-Null
[System.IO.Directory]::CreateDirectory($LogDir) | Out-Null

$existingContainers = @(@(& docker ps -a --format '{{.Names}}') | Where-Object { $_ })
if ($existingContainers -contains $ContainerName) {
    Write-Host "Found existing container named $ContainerName."
    $runningContainers = @(@(& docker ps --format '{{.Names}}') | Where-Object { $_ })
    if ($runningContainers -contains $ContainerName) {
        Write-Host 'Container is already running; skipping recreation.'
    }
    else {
        Write-Host "Removing stopped container $ContainerName."
        & docker rm $ContainerName | Out-Null
    }
}

$runningContainers = @(@(& docker ps --format '{{.Names}}') | Where-Object { $_ })
if (-not ($runningContainers -contains $ContainerName)) {
    Write-Host 'Starting MongoDB container...'
    $dockerRunArgs = @(
        'run', '-d',
        '--name', $ContainerName,
        '-p', "${Port}:27017",
        '-v', "${DataDir}:/data/db",
        '-v', "${LogDir}:/var/log/mongodb",
        '-e', "MONGO_INITDB_ROOT_USERNAME=$RootUser",
        '-e', "MONGO_INITDB_ROOT_PASSWORD=$RootPassword",
        'mongo:7.0',
        '--wiredTigerCacheSizeGB', '1',
        '--logpath', '/var/log/mongodb/mongod.log',
        '--bind_ip_all'
    )
    & docker @dockerRunArgs | Out-Null
}
else {
    Write-Host 'MongoDB container already running; skipping docker run.'
}

Write-Host 'Waiting for MongoDB to accept connections...'
$attempts = 0
while ($attempts -lt 20) {
    $null = & docker exec $ContainerName mongosh --quiet --eval "db.adminCommand('ping')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        break
    }
    Start-Sleep -Seconds 1
    $attempts++
}

if ($LASTEXITCODE -ne 0) {
    Write-Error 'MongoDB did not become ready in time.'
    exit 1
}

Write-Host 'MongoDB is ready. Ensuring application user exists...'
$createUserScript = @"
const targetDb = db.getSiblingDB('$AppDb');
if (!targetDb.getUser('$AppUser')) {
  targetDb.createUser({ user: '$AppUser', pwd: '$AppPassword', roles: [{ role: 'readWrite', db: '$AppDb' }] });
  print('Created application user $AppUser in database $AppDb.');
} else {
  print('Application user $AppUser already exists in database $AppDb.');
}
"@
$createUserScript = ($createUserScript -split "`r?`n") -join ' '

$null = & docker exec $ContainerName mongosh --quiet --username $RootUser --password $RootPassword --authenticationDatabase admin --eval $createUserScript 2>$null

Write-Host ''
Write-Host 'MongoDB local setup complete!'
Write-Host ''
Write-Host 'Connection details:'
Write-Host "  MongoDB URI (admin): mongodb://${RootUser}:${RootPassword}@localhost:${Port}/admin"
Write-Host "  MongoDB URI (app):   mongodb://${AppUser}:${AppPassword}@localhost:${Port}/${AppDb}"
Write-Host ''
Write-Host 'Remember to keep your credentials safe and rotate them for production usage.'

