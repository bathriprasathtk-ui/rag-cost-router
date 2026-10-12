# Starts the RAG router API + dashboard.
# Prerequisite: start Qdrant first in another terminal (.\start-qdrant.ps1), optional.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
