# Suite verification; no device or Internet needed. Some tests use loopback HTTP.
param(
    [switch]$ArchitectureOnly,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$suiteRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $workspacePython = Join-Path $suiteRoot '.venv\Scripts\python.exe'
    $Python = if (Test-Path -LiteralPath $workspacePython) { $workspacePython } else { 'python' }
}
# Resolve a caller-supplied relative interpreter before changing package cwd.
$Python = (Get-Command $Python -ErrorAction Stop).Source
$originalPythonPath = $env:PYTHONPATH
$originalPythonEncoding = $env:PYTHONIOENCODING

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python verification failed (exit $LASTEXITCODE)." }
}

try {
    Invoke-CheckedPython -Arguments @('-c', 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"')
    $env:PYTHONIOENCODING = 'utf-8'
    $sourceRoots = @('ningsi\src', 'ningsi-studio\src', 'ningsi-capture\src') | ForEach-Object { Join-Path $suiteRoot $_ }
    $env:PYTHONPATH = $sourceRoots -join [IO.Path]::PathSeparator
    Invoke-CheckedPython -Arguments @((Join-Path $PSScriptRoot 'check_architecture.py'))
    Invoke-CheckedPython -Arguments @('-m', 'unittest', 'discover', '-s', (Join-Path $PSScriptRoot 'tests'), '-p', 'test_*.py')
    if (-not $ArchitectureOnly) {
        Invoke-CheckedPython -Arguments @('-m', 'ningsi_studio.tools.static_check')
        foreach ($package in @('ningsi', 'ningsi-studio', 'ningsi-capture')) {
            Write-Host "Testing $package ..."
            $packageRoot = Join-Path $suiteRoot $package
            Push-Location $packageRoot
            try {
                Invoke-CheckedPython -Arguments @('-m', 'unittest', 'discover', '-s', 'tests', '-t', '.')
            } finally {
                Pop-Location
            }
        }
    }
    Write-Host 'Suite verification passed.'
} finally {
    $env:PYTHONPATH = $originalPythonPath
    $env:PYTHONIOENCODING = $originalPythonEncoding
}
