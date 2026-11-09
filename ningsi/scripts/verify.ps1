# One-shot verification: run all tests, run one end-to-end demo, list artifacts.
# Keep this file ASCII-only so Windows PowerShell 5.1 (ANSI default) reads it correctly.
param([string]$Participant = "p01")

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "src"
$ErrorActionPreference = "Continue"

Write-Host "[1/2] run tests ..." -ForegroundColor Cyan
python -m unittest discover -s tests -t .
if ($LASTEXITCODE -ne 0) { Write-Host "tests FAILED" -ForegroundColor Red; exit 1 }

Write-Host "[2/2] end-to-end demo ..." -ForegroundColor Cyan
python -m ningsi.app.cli demo --root var/verify --participant $Participant
if ($LASTEXITCODE -ne 0) { Write-Host "demo FAILED" -ForegroundColor Red; exit 1 }

Write-Host "artifacts under var/verify:" -ForegroundColor Cyan
Get-ChildItem -Recurse -File var/verify | ForEach-Object { $_.FullName.Replace($root + '\', '') }
Write-Host "done: tests passed, demo artifacts written to var/verify" -ForegroundColor Green
