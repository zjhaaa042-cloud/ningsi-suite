"""Create a new, data-free Windows showcase release from fixed build inputs.

Run after PyInstaller completes. Existing releases and archives are never
overwritten or removed. --check-only validates inputs without writing files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STUDIO_VERSION = "0.1.9"
CAPTURE_VERSION = "0.4.0"
STUDIO_EXE = "NingsiStudio-0.1.9-Showcase.exe"
STUDIO_BUILD = ROOT / "ningsi-studio/dist/NingsiStudio-0.1.9-Showcase"
CAPTURE_EXE = ROOT / "dist/biomulti-preproc-0.4.0/NingsiCapture.exe"
DEFAULT_LABEL = "competition-showcase-0.1.9-20261117"
PRIVATE_EXTENSIONS = {".sqlite", ".sqlite3", ".db", ".npz", ".npy", ".xdf", ".edf", ".bdf", ".rec", ".jsonl"}
PRIVATE_DIRECTORIES = {"var", ".git", ".venv"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path, parent):
    resolved = path.resolve()
    resolved.relative_to(parent.resolve())
    return resolved


def _git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _version(package):
    return tomllib.loads((ROOT / package / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def validate_inputs(label):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", label) or label in (".", ".."):
        raise ValueError("Release label must be a simple directory name")
    release = _inside(ROOT / "dist" / label, ROOT / "dist")
    archive = release.with_suffix(release.suffix + ".zip")
    archive_checksum = archive.with_suffix(archive.suffix + ".sha256")
    if any(path.exists() for path in (release, archive, archive_checksum)):
        raise FileExistsError("Release directory or archive exists; choose a new label. Nothing was overwritten.")
    for package, expected in (("ningsi-studio", STUDIO_VERSION), ("ningsi-capture", CAPTURE_VERSION)):
        if _version(package) != expected:
            raise ValueError(f"Unexpected {package} source version; update the packaging recipe deliberately")
    for path in (STUDIO_BUILD / STUDIO_EXE, CAPTURE_EXE, ROOT / "ningsi-studio/docs/SHOWCASE.md", ROOT / "LICENSE"):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing build input: {path}")
    for executable in (STUDIO_BUILD / STUDIO_EXE, CAPTURE_EXE):
        with executable.open("rb") as handle:
            if handle.read(2) != b"MZ":
                raise ValueError(f"Input is not a Windows executable: {executable.name}")
    required_assets = (
        "_internal/web/index.html", "_internal/web/js/views/showcase.js",
        "_internal/web/css/showcase.css",
        "_internal/ningsi_studio/domain/showcase_evidence_snapshot.json",
    )
    for name in required_assets:
        if not (STUDIO_BUILD / name).is_file():
            raise FileNotFoundError(f"Missing frozen showcase asset: {name}")
    members = []
    for path in sorted(STUDIO_BUILD.rglob("*")):
        _inside(path, STUDIO_BUILD)
        if path.is_symlink():
            raise ValueError("Build input must not contain symbolic links")
        relative = path.relative_to(STUDIO_BUILD)
        if any(part.casefold() in PRIVATE_DIRECTORIES for part in relative.parts):
            continue  # Runtime directories are never copied, even if a doctor run created one.
        if path.is_file():
            if path.suffix.casefold() in PRIVATE_EXTENSIONS:
                raise ValueError(f"Potential raw data is not a release asset: {relative}")
            members.append(path)
    snapshot = json.loads((STUDIO_BUILD / required_assets[-1]).read_text(encoding="utf-8"))
    if snapshot.get("evidence_origin") != "bundled_development_snapshot":
        raise ValueError("Frozen evidence must be the anonymized development snapshot")
    encoded = json.dumps(snapshot, ensure_ascii=False)
    if any(value in encoded for value in ('"samples"', '"participant"', '"session_id"', "P001")):
        raise ValueError("Evidence snapshot contains a private field or identifier")
    # Validate the reused Capture binary against its original distribution record.
    capture_sha = sha(CAPTURE_EXE)
    original_checksum = CAPTURE_EXE.parent / "SHA256SUMS.txt"
    checksum_lines = original_checksum.read_text(encoding="utf-8-sig").splitlines()
    expected = [line.split()[0] for line in checksum_lines if line.strip().endswith("NingsiCapture.exe")]
    if expected != [capture_sha]:
        raise ValueError("Capture binary does not match its original distribution checksum")
    return release, archive, archive_checksum, members, capture_sha


def _source_manifest():
    values = {}
    for folder in ("ningsi/src", "ningsi-studio/src", "ningsi-studio/web"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".sql", ".js", ".mjs", ".css", ".html"}:
                values[path.relative_to(ROOT).as_posix()] = sha(path)
    return values


def _instructions():
    return """凝思 A09 比赛展示版 0.1.9（2026 年 11 月 17 日）

软件：Studio 0.1.9 + 独立采集器 Capture 0.4.0，Windows x64。
完整解压后运行，不需要安装 Python。保留 _internal 整个目录，不要只复制 EXE。

一、进入比赛展示
1. 双击 01-Start-Studio.bat，程序启动本机服务并打开浏览器。
2. 在左侧导航点击“赛题展示”。默认地址通常为 http://127.0.0.1:8765/#/showcase；
   端口被占用时可能顺延，请以控制台地址为准。亦可双击 Studio EXE 后进入同一页面。
3. 展示“需求总览、设备采集、信号预处理、PSD与特征、模型训练、联合评估与报告、历史趋势”。
   处理分析支持明确标注的模拟信号和用户选择的原始 JSON，不会把模拟结果标成真实设备结果。
4. 设备表单记录实际型号、厂家、导联、参考、单位及资料出处。填写完成仍是操作员声明，
   不会自动认证合规。证据卡中的开发快照不代表本次现场采集。
5. 模型训练、正式评估报告、实时监测、反馈训练和历史趋势均可从页面入口进入。
   选择或创建会话后查看对应现场结果；没有结果时保留缺失，不补造真人分数。

二、真实采集
双击 02-Start-Capture.bat，或直接运行 NingsiCapture.exe。
先由设备厂商软件发布 EEG 流，核对实际导联、参考和单位，再开始连续记录。
原始记录和任务事件保存在操作者选择的位置，停止后导出，再到 Studio 导入和训练。
Capture 沿用已发布 0.4.0，本次没有改动其程序。设备时标、单位和质量条件不满足时保留拒评原因。

三、数据与关闭
本压缩包不含真人原始脑电、会话数据库、参与者资料或个人训练模型。
Studio 首次运行会创建 var/studio 保存本机数据；请保留自己的新数据并单独备份。
不能把该目录用于再次分发。退出浏览器不会停止服务；在 Studio 控制台按 Ctrl+C 停止。
Capture 按界面完成停止、保存和关闭。不要把旧版本 _internal 混入本目录。

四、验证范围
本轮整仓 506 项软件测试通过：架构工具 10、引擎 93、Studio 302、Capture 101。
Capture GUI 测试退出时出现 Tk 回调/主题销毁日志，未导致用例失败。
测试支持软件功能与链路，不代表新的真人识别准确率或训练效果。
现有压力、情绪和专注识别仍需目标设备独立验证；伪迹标记/拒评与重建校正分别解释。
程序未做代码签名。模型和源代码来自存在未提交修改的工作区，详见 BUILD.json。
本包准备完成后的 EXE 实际启动检查由交付验收记录另行确认；本文件不预先宣称已通过。

五、文件校验
FILES.sha256 为包内逐文件 SHA-256；ZIP 同目录 .zip.sha256 用于校验下载包。
SHOWCASE.md 为源码说明与演示步骤，LICENSE.txt 为自研代码许可。
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    release, archive, archive_checksum, members, capture_sha = validate_inputs(args.label)
    if args.check_only:
        print(json.dumps({"ready": True, "studio_files": len(members), "release": str(release),
                          "capture_sha256": capture_sha}, ensure_ascii=False, indent=2))
        return 0
    # copytree itself refuses an existing output; no recursive delete/move is used.
    shutil.copytree(STUDIO_BUILD, release, ignore=shutil.ignore_patterns(*PRIVATE_DIRECTORIES))
    shutil.copy2(CAPTURE_EXE, release / "NingsiCapture.exe")
    shutil.copy2(ROOT / "ningsi-studio/docs/SHOWCASE.md", release / "SHOWCASE.md")
    shutil.copy2(ROOT / "LICENSE", release / "LICENSE.txt")
    (release / "使用说明.txt").write_text(_instructions().replace("\n", "\r\n"), encoding="utf-8-sig", newline="")
    for name, command in (("01-Start-Studio.bat", f'"{STUDIO_EXE}" serve --open'),
                          ("02-Start-Capture.bat", '"NingsiCapture.exe"')):
        (release / name).write_text("@echo off\r\nsetlocal\r\ncd /d \"%~dp0\"\r\n" + command +
                                    "\r\nif errorlevel 1 pause\r\n", encoding="ascii", newline="")
    dirty = _git("status", "--porcelain")
    dependencies = {}
    for name in ("numpy", "pylsl", "pyinstaller"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    build = {
        "release": args.label, "packaged_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "platform": "Windows x64; Studio onedir + Capture onefile",
        "versions": {"studio": STUDIO_VERSION, "capture": CAPTURE_VERSION},
        "executables": {STUDIO_EXE: sha(release / STUDIO_EXE), "NingsiCapture.exe": capture_sha},
        "source_state": "working tree with uncommitted changes; not a clean tagged release",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_dirty": None if dirty is None else bool(dirty),
        "source_changed_entries": None if dirty is None else len(dirty.splitlines()),
        "source_sha256_at_packaging": _source_manifest(),
        "dependencies": dependencies, "python_packager": sys.version,
        "software_verification": {"total": 506, "architecture": 10, "engine": 93, "studio": 302, "capture": 101,
                                  "passed": True, "scope": "software and synthetic fixtures; independent human/device validity not established",
                                  "gui_shutdown_log": "Tk callbacks/theme events after window destruction; no failed assertions"},
        "frozen_service_smoke": "not yet recorded by packaging script",
        "code_signed": False, "capture_reused_without_modification": True,
        "contains_raw_eeg": False, "contains_participant_records": False,
        "contains_runtime_database": False, "contains_personal_trained_models": False,
        "bundled_evidence": "anonymized development snapshot, not current-device realtime results",
    }
    smoke_path = ROOT / "var/showcase-20261117/release-smoke.json"
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("passed") is not True or smoke.get("hardware_or_human_validity_tested") is not False:
            raise ValueError("Unexpected frozen-service verification record")
        build["frozen_service_smoke"] = {"passed": True, "checks": smoke["check_count"],
                                         "verified_at": smoke["verified_at"]}
        (release / "验收记录.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (release / "BUILD.json").write_text(json.dumps(build, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    files = [path for path in sorted(release.rglob("*")) if path.is_file()]
    checksum_values = {path.relative_to(release).as_posix(): sha(path) for path in files}
    (release / "FILES.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in checksum_values.items()), encoding="utf-8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                zipped.write(path, f"{release.name}/{path.relative_to(release).as_posix()}")
    archive_sha = sha(archive)
    archive_checksum.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    # Verify every archived byte against the delivered file manifest without extracting.
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        for name, digest in checksum_values.items():
            if hashlib.sha256(zipped.read(f"{release.name}/{name}")).hexdigest() != digest:
                raise ValueError(f"ZIP member checksum mismatch: {name}")
    result = {"release": str(release), "zip": str(archive), "zip_sha256_file": str(archive_checksum),
              "zip_sha256": archive_sha, "files": len(checksum_values) + 1,
              "zip_bytes": archive.stat().st_size, "member_checksums_verified": len(checksum_values),
              "no_raw_eeg_or_runtime_databases": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
