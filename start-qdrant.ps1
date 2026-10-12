# Runs Qdrant natively (no Docker/WSL required).
# Binary fetched from https://github.com/qdrant/qdrant/releases into tools\qdrant
$ErrorActionPreference = "Stop"
$dir = Join-Path $PSScriptRoot "tools\qdrant"
$exe = Join-Path $dir "qdrant.exe"

if (-not (Test-Path $exe)) {
    Write-Host "Qdrant binary not found - downloading..."
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $zip = "$env:TEMP\qdrant.zip"
    curl.exe -L -o $zip "https://github.com/qdrant/qdrant/releases/download/v1.19.2/qdrant-x86_64-pc-windows-msvc.zip" --retry 3
    Expand-Archive -Path $zip -DestinationPath $dir -Force
}

$qdrantStorage = Join-Path $PSScriptRoot "data\qdrant_storage"
New-Item -ItemType Directory -Force -Path $qdrantStorage | Out-Null

$env:QDRANT__STORAGE__STORAGE_PATH = $qdrantStorage
Write-Host "Starting Qdrant on http://127.0.0.1:6333 (storage: $qdrantStorage)"
& $exe
