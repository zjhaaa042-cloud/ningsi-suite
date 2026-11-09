"""生成演示视频逐镜中文旁白（edge-tts 自然语音 + ffmpeg 柔化后处理）。

用法（在 D:\\Desktop\\ningsi 下运行）：
    python ningsi\\tools\\build_demo_voice.py                       # 全部镜头
    python ningsi\\tools\\build_demo_voice.py --only s01 s06        # 只重做指定镜头

相比 Windows 内置 SAPI（Microsoft Huihui Desktop），这里改用与 bsense-lsl 语音提示同源的
神经元语音 zh-CN-XiaoxiaoNeural：语调自然、气息柔和；再统一做三件事，去掉机械感：

  1. 语速 -8%、音高 -3 Hz：放慢并压低，听起来更从容；
  2. 高频柔化（7 kHz 以上 -3.5 dB）+ 250 Hz 轻微加暖：削掉齿音与刺耳感；
  3. 轻压缩 + 响度归一化到 -16 LUFS，并在句首留 0.35 秒、句尾留 0.55 秒静音，
     避免紧贴画面切换的突兀感。

产物：var\\video_build\\voice\\sNN.wav（48 kHz 立体声），供 build_demo_video.py 合成使用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "var" / "video_build"
VOICE = BUILD / "voice"

# 柔美向参数：神经元女声 + 略慢语速 + 略低音高
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TTS_RATE = "-14%"
TTS_PITCH = "-3Hz"
TTS_VOLUME = "+0%"

# 柔化后处理链：削齿音 → 加暖 → 轻压缩 → 响度归一化 → 首尾留白
# 注意：不要加低通滤波器——实测 ffmpeg 的 lowpass 会在截止频率附近留下一条持续窄带线，
# 听感上像底噪鸣音，反而更“生硬”。
POLISH = (
    "highshelf=f=6500:g=-4.5,"
    "equalizer=f=250:width_type=o:width=1.2:g=1.2,"
    "acompressor=threshold=-20dB:ratio=2.5:attack=10:release=200:makeup=2,"
    "loudnorm=I=-16:TP=-1.5:LRA=11,"
    "adelay=350:all=1,"
    "apad=pad_dur=0.55"
)


def synthesize(text: str, voice: str, rate: str, pitch: str, volume: str, mp3: Path) -> None:
    import edge_tts

    async def run() -> None:
        await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume).save(str(mp3))

    asyncio.run(run())


def polish(mp3: Path, wav: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-af", POLISH,
         "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(wav)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg 后处理失败：{result.stderr[-600:]}")


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="只重做指定镜号，例如 --only s01 s06")
    parser.add_argument("--voice", default=TTS_VOICE)
    parser.add_argument("--rate", default=TTS_RATE)
    parser.add_argument("--pitch", default=TTS_PITCH)
    args = parser.parse_args()

    scenes = json.loads((BUILD / "scenes.json").read_text(encoding="utf-8"))
    if args.only:
        scenes = [item for item in scenes if item["id"] in set(args.only)]
    VOICE.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ningsi-voice-") as temp:
        for scene in scenes:
            mp3 = Path(temp) / f"{scene['id']}.mp3"
            wav = VOICE / f"{scene['id']}.wav"
            synthesize(scene["text"], args.voice, args.rate, args.pitch, TTS_VOLUME, mp3)
            polish(mp3, wav)
            print(f"  {scene['id']}  {duration(wav):5.1f}s  {wav.stat().st_size/1024:6.0f} KB  {scene['text'][:22]}…")

    total = sum(duration(VOICE / f"{scene['id']}.wav") for scene in scenes)
    print(f"旁白完成：{len(scenes)} 段 / {total:.1f}s = {total/60:.2f} 分钟 → {VOICE}")
    print(f"音色：{args.voice}｜语速 {args.rate}｜音高 {args.pitch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
