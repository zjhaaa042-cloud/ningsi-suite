# 生成演示视频逐镜中文旁白（Windows 内置 SAPI，zh-CN 女声 Microsoft Huihui Desktop）。
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File ningsi\tools\build_demo_voice.ps1
param([string]$BuildDir = "D:\Desktop\ningsi\var\video_build", [string]$VoiceName = "Microsoft Huihui Desktop")

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$scenesPath = Join-Path $BuildDir "scenes.json"
if (-not (Test-Path $scenesPath)) { throw "缺少 $scenesPath（先生成 scenes.json）" }
$scenes = Get-Content $scenesPath -Raw -Encoding UTF8 | ConvertFrom-Json

$voiceDir = Join-Path $BuildDir "voice"
New-Item -ItemType Directory -Force -Path $voiceDir | Out-Null

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$available = $synth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }
if ($available -contains $VoiceName) { $synth.SelectVoice($VoiceName) } else { Write-Warning "未找到 $VoiceName，使用默认语音：$($available -join ', ')" }
$synth.Rate = -1        # 略慢于默认语速，便于看画面

foreach ($scene in $scenes) {
    $path = Join-Path $voiceDir ($scene.id + ".wav")
    $synth.SetOutputToWaveFile($path)
    $synth.Speak($scene.text)
    $synth.SetOutputToNull()
    $size = [math]::Round((Get-Item $path).Length / 1KB, 0)
    Write-Output ("  {0}  {1} KB  {2}" -f $scene.id, $size, $scene.text.Substring(0, [Math]::Min(24, $scene.text.Length)))
}
$synth.Dispose()
Write-Output "旁白完成：$voiceDir"
