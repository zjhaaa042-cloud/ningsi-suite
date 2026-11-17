param([ValidateSet("onefile", "onedir")][string]$Mode = "onefile", [string]$PythonExe)
$ErrorActionPreference = "Continue"
$CaptureRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $CaptureRoot
$CapturePython = if ($PythonExe) { $PythonExe } else { Join-Path $RepoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $CapturePython)) { throw "Workspace Python was not found." }
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BuildRoot = Join-Path $CaptureRoot "build"
$env:PYINSTALLER_CONFIG_DIR = Join-Path $BuildRoot "pyinstaller-cache"
$Stage = Join-Path $BuildRoot ("stage_" + $Stamp)
$CaptureStage = Join-Path $Stage "capture"
$EngineStage = Join-Path $Stage "engine"
New-Item -ItemType Directory -Force -Path $CaptureStage, $EngineStage | Out-Null
foreach ($Pair in @(@((Join-Path $CaptureRoot "src\ningsi_capture"), (Join-Path $CaptureStage "ningsi_capture")), @((Join-Path $RepoRoot "ningsi\src\ningsi"), (Join-Path $EngineStage "ningsi")))) {
    $SourceDir = [System.IO.Path]::GetFullPath($Pair[0])
    $TargetDir = [System.IO.Path]::GetFullPath($Pair[1])
    foreach ($SourceFile in (Get-ChildItem -LiteralPath $SourceDir -Recurse -File -Filter '*.py')) {
        $RelativeFile = $SourceFile.FullName.Substring($SourceDir.Length + 1)
        $TargetFile = Join-Path $TargetDir $RelativeFile
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $TargetFile) | Out-Null
        Copy-Item -LiteralPath $SourceFile.FullName -Destination $TargetFile
    }
}
$env:PYTHONPATH = $CaptureStage + ";" + $EngineStage
$EngineSourceData = (Join-Path $EngineStage "ningsi") + ";capture_engine_source"
$ModeFlag = if ($Mode -eq "onefile") { "--onefile" } else { "--onedir" }
$BuildArgs = @(
    "-m", "PyInstaller", "--noconfirm", "--log-level", "WARN", $ModeFlag, "--windowed",
    "--name", "NingsiCapture", "--distpath", (Join-Path $CaptureRoot "dist"),
    "--workpath", (Join-Path $BuildRoot ("work_" + $Stamp)), "--specpath", $BuildRoot,
    "--paths", $CaptureStage, "--paths", $EngineStage,
    "--collect-all", "pylsl", "--collect-submodules", "ningsi_capture",
    "--collect-submodules", "ningsi.signal", "--collect-submodules", "ningsi.models",
    "--hidden-import", "ningsi.behavior.sart", "--hidden-import", "ningsi.config",
    "--add-data", $EngineSourceData,
    "--exclude-module", "matplotlib", "--exclude-module", "pandas", "--exclude-module", "scipy",
    (Join-Path $PSScriptRoot "frozen_entry.py")
)
foreach ($ModelName in @("attlapse_healthy_model.json", "attlapse_shadow_250.json")) {
    $ModelFile = Join-Path $RepoRoot ("ningsi\src\ningsi\models\" + $ModelName)
    if (-not (Test-Path -LiteralPath $ModelFile -PathType Leaf)) { throw "Missing capture model: $ModelName" }
    $BuildArgs += @("--add-data", ($ModelFile + ";ningsi/models"))
}
$Log = Join-Path $BuildRoot ("build_" + $Stamp + ".log")
& $CapturePython @BuildArgs *> $Log
if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $Log -Tail 50; exit 1 }
$Exe = if ($Mode -eq "onefile") { Join-Path $CaptureRoot "dist\NingsiCapture.exe" } else { Join-Path $CaptureRoot "dist\NingsiCapture\NingsiCapture.exe" }
Get-Item -LiteralPath $Exe | Select-Object FullName,Length
$ExeHash = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
Write-Output $ExeHash
$PackageVersion = & $CapturePython -c "from ningsi_capture import __version__; print(__version__)"
$BuildRecord = [ordered]@{ app = 'ningsi-capture'; version = $PackageVersion; built_utc = (Get-Date).ToUniversalTime().ToString('o'); sha256 = $ExeHash; mode = $Mode; stage = $Stage }
$BuildRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path (Split-Path -Parent $Exe) 'BUILD.json') -Encoding UTF8
exit 0
