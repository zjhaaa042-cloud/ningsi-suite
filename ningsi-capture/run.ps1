param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CaptureArgs)
$ErrorActionPreference = "Stop"
$CaptureRoot = $PSScriptRoot
$RepoRoot = Split-Path -Parent $CaptureRoot
$CapturePython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $CapturePython)) {
    $CapturePython = (Get-Command python -ErrorAction Stop).Source
}
$env:PYTHONPATH = (Join-Path $CaptureRoot "src") + ";" + (Join-Path $RepoRoot "ningsi\src")
& $CapturePython -m ningsi_capture @CaptureArgs
exit $LASTEXITCODE
