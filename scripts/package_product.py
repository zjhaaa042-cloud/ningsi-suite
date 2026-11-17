"""Package a new, data-free Windows product release from verified frozen builds.

Inputs are explicit and existing releases are never overwritten. --check-only
checks the receipt, suite log, frozen assets and delivery paths without copying.
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
import sys
import zipfile

from package_showcase import ROOT, _git, _inside, _source_manifest, _version, sha


PRIVATE_DIRECTORIES = {
    "var", ".git", ".venv", "venv", "qa", "_analysis", ".pytest_cache", "__pycache__",
    "human_models", "personal_models", "device_profiles", "recordings", "datasets", "exports",
}
PRIVATE_EXTENSIONS = {
    ".sqlite", ".sqlite3", ".db", ".npz", ".npy", ".xdf", ".edf", ".bdf", ".rec",
    ".jsonl", ".csv", ".tsv", ".log", ".pfx", ".key",
    ".sqlite-wal", ".sqlite-shm", ".sqlite3-wal", ".sqlite3-shm", ".db-wal", ".db-shm",
}
PUBLIC_JSON = {
    "_internal/ningsi/models/frontal_affect.json",
    "_internal/ningsi_studio/domain/showcase_evidence_snapshot.json",
}
REQUIRED_ASSETS = (
    "_internal/web/index.html", "_internal/web/js/main.js", "_internal/web/js/views/start.js",
    "_internal/web/js/views/device-profile.js", "_internal/web/js/views/session-preflight.js",
    "_internal/web/js/views/state-heatmap.js",
    "_internal/web/css/workspace.css", "_internal/ningsi_studio/db/schema.sql",
)
SHA_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def validate_label(label):
    if (not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", label)
            or label.endswith(".") or label.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("Release label must be a simple, non-reserved directory name")
    return label


def output_paths(label, dist=None):
    validate_label(label)
    dist = Path(dist) if dist is not None else ROOT / "dist"
    release = _inside(dist / label, dist)
    archive, checksum = dist / f"{label}.zip", dist / f"{label}.zip.sha256"
    if any(path.exists() for path in (release, archive, checksum)):
        raise FileExistsError("Release directory or archive exists; choose a new label. Nothing was overwritten.")
    return release, archive, checksum


def _check_link(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError(f"Build input must not contain links or junctions: {path.name}")


def build_members(folder):
    """Return only validated frozen assets; private runtime directories are skipped."""
    folder = Path(folder)
    _check_link(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing Studio build directory: {folder}")
    members = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if any(part.casefold() in PRIVATE_DIRECTORIES or part.casefold().startswith(".chart-data-")
               for part in relative.parts):
            continue
        _check_link(path)
        _inside(path, folder)
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        if (suffix in PRIVATE_EXTENSIONS or path.name.casefold().startswith(".env")
                or path.name.casefold() in {"credentials", "secrets.json"}):
            raise ValueError(f"Potential private data is not a release asset: {relative}")
        if suffix == ".json" and relative.as_posix() not in PUBLIC_JSON:
            raise ValueError(f"Unapproved JSON is not a frozen public asset: {relative}")
        if relative.as_posix().endswith("showcase_evidence_snapshot.json"):
            snapshot = json.loads(path.read_text(encoding="utf-8-sig"))
            encoded = json.dumps(snapshot, ensure_ascii=False)
            if (snapshot.get("evidence_origin") != "bundled_development_snapshot"
                    or any(value in encoded for value in ('"samples"', '"participant"', '"session_id"', "P001"))):
                raise ValueError("Frozen evidence must contain only the anonymized development snapshot")
        members.append(path)
    return members


def parse_suite(path):
    """Read actual passed unittest groups; never hard-code a suite test count."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("A bounded, complete suite verification log is required")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if (not text.rstrip().endswith("Suite verification passed.")
            or re.search(r"^FAILED(?:\s|$)", text, flags=re.MULTILINE)):
        raise ValueError("Suite log does not finish successfully")
    completed = list(re.finditer(r"^Ran (\d+) tests? in [^\r\n]+\r?\n\s*OK[^\r\n]*$", text, re.MULTILINE))
    if not completed:
        raise ValueError("Suite log has no passed unittest counts")
    markers = list(re.finditer(r"^Testing (ningsi|ningsi-studio|ningsi-capture) \.\.\.\s*$", text, re.MULTILINE))
    seen = set()
    groups = []
    for match in completed:
        before = [marker for marker in markers if marker.start() < match.start()]
        package = before[-1].group(1) if before else "architecture"
        seen.add(package)
        groups.append({"package": package, "tests": int(match.group(1)), "passed": True})
    if not {"ningsi", "ningsi-studio", "ningsi-capture"}.issubset(seen):
        raise ValueError("Suite log must include all three product packages")
    return {"passed": True, "total": sum(group["tests"] for group in groups), "groups": groups,
            "log_sha256": sha(path), "scope": "software and synthetic fixtures; independent device/human validity not tested"}


def read_verification(path, executable_hashes, versions, capture_build=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("A bounded frozen-product verification receipt is required")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(data, dict) or data.get("passed") is not True
            or data.get("hardware_or_human_validity_tested") is not False
            or type(data.get("check_count")) is not int or data["check_count"] <= 0):
        raise ValueError("Frozen verification must pass and explicitly exclude device/human validity")
    checks = data.get("checks")
    if checks is not None and (not isinstance(checks, list) or len(checks) != data["check_count"]):
        raise ValueError("Frozen verification check count is inconsistent")
    declared = data.get("exe_sha256")
    if isinstance(declared, str):
        declared = {"studio": declared, "capture": data.get("capture_exe_sha256")}
    if not isinstance(declared, dict):
        raise ValueError("Frozen receipt must bind the verified executable SHA-256")
    if declared.get("capture") is None and capture_build:
        declared = {**declared, "capture": capture_build.get("sha256")}
    for name, digest in executable_hashes.items():
        claimed = declared.get(name)
        if not isinstance(claimed, str) or not SHA_PATTERN.fullmatch(claimed) or claimed.lower() != digest.lower():
            raise ValueError(f"Frozen verification does not match {name} executable")
    receipt_versions = data.get("versions")
    if receipt_versions is not None:
        if not isinstance(receipt_versions, dict) or any(receipt_versions.get(name) != value for name, value in versions.items()):
            raise ValueError("Frozen receipt versions do not match current package versions")
    # Publish only bounded verification metadata, never arbitrary API fixture payloads.
    names = []
    for check in checks or []:
        if isinstance(check, str):
            names.append(check)
        elif isinstance(check, dict) and isinstance(check.get("name"), str) and check.get("passed", True) is True:
            names.append(check["name"])
        else:
            raise ValueError("Frozen checks must be names or passed named checks")
    return {"passed": True, "verified_at": data.get("verified_at"), "check_count": data["check_count"],
            "checks": names, "exe_sha256": executable_hashes, "versions": receipt_versions,
            "receipt_sha256": sha(path), "hardware_or_human_validity_tested": False,
            "scope": "packaged software and explicit fixtures; no new human accuracy or feedback-effect evidence"}


def _capture_build_receipt(executable, version):
    path = executable.parent / "BUILD.json"
    if not path.is_file():
        return None
    if path.stat().st_size > 128 * 1024:
        raise ValueError("Capture build record is too large")
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(record, dict) or record.get("version") != version
            or str(record.get("sha256", "")).lower() != sha(executable)):
        raise ValueError("Capture executable does not match its build version/checksum")
    return {key: record.get(key) for key in ("app", "version", "built_utc", "sha256", "mode")}


def _source_hashes():
    values = _source_manifest()
    for folder in ("ningsi-capture/src", "scripts", "docs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".md", ".txt"}:
                values[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ("ningsi/pyproject.toml", "ningsi-studio/pyproject.toml", "ningsi-capture/pyproject.toml", "LICENSE"):
        values[name] = sha(ROOT / name)
    return values


def _instructions(versions, studio_exe, suite, receipt):
    return f"""凝思 Windows 产品版
Studio {versions['studio']} / Capture {versions['capture']}

完整解压到可写目录后运行，无需安装 Python。保留 _internal 整个目录；不要只复制 EXE。

一、评估与训练
1. 双击 01-Start-Studio.bat，程序启动本机服务并打开浏览器。端口以控制台地址为准。
2. 主页面选择评估方案、参与者与 EEG 数据源。首次接入先在厂商软件开启 EEG 推流。
3. 展开“设置与管理 → 设备连接”，核对来源并填写厂家、型号、电极、参考与单位依据。
   档案保存为操作员声明，始终待现场核验；同名流的档案不能证明当前设备已经合规。
4. 开始前查看“本次评分与模型适配”。基础指数与真人模型分别解释；质量和基线需采集后检查。
   设备未声明单位时按本次配置明确确认；采用档案单位前仍需核对当前硬件与软件设置。
5. 跟随阶段引导完成任务与训练。缺失或不合格信号不填成低专注，调整后按提示继续或重采。
6. 完成后查看评估报告、下载本次产物，在“历史与趋势”回顾可比较的同人记录。
7. 无设备时仅使用“运行模拟演示”；模拟数据始终单独标记，不代表真人准确率。
实时监测中的热力图区分信号合格、等待基线与可评分窗口；无评分时先显示原因摘要。
可展开逐窗记录、翻页或仅看可评分片段，筛选不会改变全会话覆盖率，窗口时间不是连续有效时长。

二、连续采集、模型与外部留证
双击 02-Start-Capture.bat 打开独立采集器。按预定研究协议记录匿名编号和开发/封存测试分组，
确认电极、参考、单位与时标后开始连续采集；完成任务后正常停止并导出，再到 Studio 的
“设置与管理 → 模型管理”导入验收、训练、独立评估，并按设备和处理版本匹配后启用。
详细模型说明见 HUMAN_MODEL_TRAINING.md，现场清单见 FIELD_ACCEPTANCE_CHECKLIST.txt。
Capture 可按需显式开启 LSL 任务 Marker（默认关闭），配合独立 LabRecorder 留证。
Marker 只含白名单任务事件，不含探针答案；另行检查 XDF 的流与完整性。
普通实时 LSL 不具备原生 REC 离线核对的全部时钟证据，不强行启用要求原生 REC 的模型。

三、数据保存与升级
本软件包不含真人原始 EEG、会话数据库、参与者资料或个人训练模型。
Studio 默认在程序旁 var/studio 保存数据；目录不可写时使用用户应用数据目录。
Capture 使用界面所选保存位置。正常停止并关闭程序后再备份整个数据目录。
升级请使用新目录，不将旧 _internal 混入本包。需要旧平台数据时先备份，再用命令行：
"{studio_exe}" serve --data "原数据目录" --open
浏览器关闭不会停止服务；在 Studio 控制台按 Ctrl+C 关闭。Capture 按界面停止并保存后关闭。
切勿把运行后生成的 var 目录重新分发。

四、验证范围
本包关联整仓 {suite['total']} 项软件测试及 {receipt['check_count']} 项冻结程序检查，实际结果见 BUILD.json
与 FROZEN_VERIFICATION.json。测试使用软件和显式夹具，不包含新的真人佩戴、识别有效性或反馈疗效证明。
压力、情绪和专注识别仍需目标设备、新参与者及跨天独立验证；反馈效果还需匹配对照。
原指数、模型估计、自评与信号质量分别保存，不作为医学诊断。程序未做代码签名。

五、文件校验
FILES.sha256 为包内逐文件 SHA-256；ZIP 同目录的 .zip.sha256 用于核对下载包。
LICENSE.txt 为自研代码许可，_internal 中的依赖许可保持随库分发。
BUILD.json 记录实际源码提交、工作区状态、源文件与 EXE 校验值，旧版本发布包保持不变。
"""


def validate_inputs(args):
    release, archive, checksum = output_paths(args.label)
    versions = {"studio": _version("ningsi-studio"), "capture": _version("ningsi-capture")}
    studio_build = _inside(args.studio_build, ROOT)
    capture_exe = _inside(args.capture_exe, ROOT)
    for path in (args.studio_build, args.capture_exe):
        _check_link(Path(path))
    studio_exes = sorted(studio_build.glob("*.exe"))
    if len(studio_exes) != 1:
        raise ValueError("Studio onedir build must contain exactly one root executable")
    studio_exe = studio_exes[0]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.exe", studio_exe.name, flags=re.IGNORECASE):
        raise ValueError("Studio executable needs a simple ASCII name for the Windows launcher")
    for executable in (studio_exe, capture_exe):
        if not executable.is_file() or executable.stat().st_size < 2:
            raise FileNotFoundError(f"Missing executable: {executable}")
        with executable.open("rb") as handle:
            if handle.read(2) != b"MZ":
                raise ValueError(f"Not a Windows executable: {executable.name}")
    members = build_members(studio_build)
    member_names = {path.relative_to(studio_build).as_posix() for path in members}
    for name in REQUIRED_ASSETS:
        if name not in member_names:
            raise FileNotFoundError(f"Missing frozen product asset: {name}")
    # The UI copied to the bundle must be the UI being delivered, not an older build.
    web_files = [path for path in members if path.relative_to(studio_build).as_posix().startswith("_internal/web/")]
    for path in web_files:
        relative = path.relative_to(studio_build / "_internal/web")
        source = ROOT / "ningsi-studio/web" / relative
        if not source.is_file() or sha(path) != sha(source):
            raise ValueError(f"Frozen web asset differs from current product source: {relative}")
    required_docs = {
        "FIELD_ACCEPTANCE_CHECKLIST.txt": ROOT / "docs/FIELD_ACCEPTANCE_CHECKLIST.txt",
        "HUMAN_MODEL_TRAINING.md": ROOT / "ningsi-studio/docs/HUMAN_MODEL_TRAINING.md",
        "LICENSE.txt": ROOT / "LICENSE",
    }
    for path in required_docs.values():
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing delivery document: {path}")
    executable_hashes = {"studio": sha(studio_exe), "capture": sha(capture_exe)}
    capture_build = _capture_build_receipt(capture_exe, versions["capture"])
    receipt = read_verification(args.verification, executable_hashes, versions, capture_build)
    suite = parse_suite(args.suite_log)
    return {"release": release, "archive": archive, "checksum": checksum, "versions": versions,
            "studio_build": studio_build, "studio_exe": studio_exe, "capture_exe": capture_exe,
            "members": members, "documents": required_docs, "executables": executable_hashes,
            "capture_build": capture_build, "receipt": receipt, "suite": suite, "web_files_verified": len(web_files)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-build", type=Path, default=ROOT / "ningsi-studio/dist" / f"NingsiStudio-{_version('ningsi-studio')}")
    parser.add_argument("--capture-exe", type=Path, default=ROOT / "ningsi-capture/dist/NingsiCapture.exe")
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--suite-log", type=Path, required=True)
    parser.add_argument("--label", default=f"ningsi-product-{_version('ningsi-studio')}-{datetime.now(timezone(timedelta(hours=8))):%Y%m%d}")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    inputs = validate_inputs(args)
    release, archive, checksum = (inputs[key] for key in ("release", "archive", "checksum"))
    if args.check_only:
        print(json.dumps({"ready": True, "release": str(release), "studio_files": len(inputs["members"]),
                          "web_files_verified": inputs["web_files_verified"], "versions": inputs["versions"],
                          "suite_tests": inputs["suite"]["total"], "frozen_checks": inputs["receipt"]["check_count"]}, ensure_ascii=False, indent=2))
        return 0
    # Exclusive creation: no existing release is overwritten and no recursive deletion is used.
    release.mkdir(parents=True, exist_ok=False)
    for path in inputs["members"]:
        target = release / path.relative_to(inputs["studio_build"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(inputs["capture_exe"], release / "NingsiCapture.exe")
    for name, path in inputs["documents"].items():
        shutil.copy2(path, release / name)
    (release / "使用说明.txt").write_text(_instructions(inputs["versions"], inputs["studio_exe"].name,
                                                      inputs["suite"], inputs["receipt"]).replace("\n", "\r\n"), encoding="utf-8-sig", newline="")
    for name, command in (("01-Start-Studio.bat", f'"{inputs["studio_exe"].name}" serve --open'),
                          ("02-Start-Capture.bat", '"NingsiCapture.exe"')):
        (release / name).write_text('@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\n' + command + '\r\nif errorlevel 1 pause\r\n', encoding="ascii", newline="")
    dirty = _git("status", "--porcelain")
    dependencies = {}
    for name in ("numpy", "pylsl", "pyinstaller"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    build = {"release": args.label, "packaged_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
             "platform": "Windows x64; Studio onedir + Capture onefile", "versions": inputs["versions"],
             "version_basis": "frozen receipt and source pyproject" if inputs["receipt"]["versions"] else "source pyproject; frozen receipt does not supply versions",
             "executables": {inputs["studio_exe"].name: inputs["executables"]["studio"], "NingsiCapture.exe": inputs["executables"]["capture"]},
             "source_commit": _git("rev-parse", "HEAD"), "source_dirty": None if dirty is None else bool(dirty),
             "source_changed_entries": None if dirty is None else len(dirty.splitlines()),
             "source_state": "not available" if dirty is None else "working tree with uncommitted changes" if dirty else "clean checkout",
             "source_sha256_at_packaging": _source_hashes(), "dependencies": dependencies, "python_packager": sys.version,
             "software_verification": inputs["suite"], "frozen_product_verification": inputs["receipt"],
             "capture_build": inputs["capture_build"], "frozen_web_files_verified": inputs["web_files_verified"],
             "code_signed": False, "hardware_or_human_validity_tested": False,
             "contains_raw_eeg": False, "contains_participant_records": False, "contains_runtime_database": False,
             "contains_personal_trained_models": False}
    for name, value in (("BUILD.json", build), ("FROZEN_VERIFICATION.json", inputs["receipt"])):
        (release / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    files = [path for path in sorted(release.rglob("*")) if path.is_file()]
    hashes = {path.relative_to(release).as_posix(): sha(path) for path in files}
    (release / "FILES.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                zipped.write(path, f"{release.name}/{path.relative_to(release).as_posix()}")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        expected = {f"{release.name}/{name}" for name in (*hashes, "FILES.sha256")}
        if set(zipped.namelist()) != expected:
            raise ValueError("ZIP member list differs from the delivery manifest")
        for name, digest in hashes.items():
            if hashlib.sha256(zipped.read(f"{release.name}/{name}")).hexdigest() != digest:
                raise ValueError(f"ZIP member checksum mismatch: {name}")
        if zipped.read(f"{release.name}/FILES.sha256") != (release / "FILES.sha256").read_bytes():
            raise ValueError("ZIP file manifest differs from the delivered file")
    archive_sha = sha(archive)
    with checksum.open("x", encoding="ascii") as handle:
        handle.write(f"{archive_sha}  {archive.name}\n")
    print(json.dumps({"release": str(release), "zip": str(archive), "zip_sha256_file": str(checksum),
                      "zip_sha256": archive_sha, "files": len(hashes) + 1, "zip_bytes": archive.stat().st_size,
                      "member_checksums_verified": len(hashes), "suite_tests": inputs["suite"]["total"],
                      "frozen_checks": inputs["receipt"]["check_count"], "no_raw_eeg_or_runtime_databases": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
