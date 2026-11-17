"""Package the desktop-accepted app, instructions and validation metadata."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DIST = PROJECT / "dist"
QA = WORKSPACE / "var/human-model-desktop-20261114"
PACKAGED_EXE_NAME = "ningsi-studio-human-data.exe"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path):
    log = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Ran (\d+) tests? in [^\r\n]+\s+OK\s*$", log)
    if not matches:
        raise ValueError(f"Suite did not finish successfully: {path}")
    return {"tests": int(matches[-1]), "passed": True, "log_sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DIST / "ningsi-studio-human-data-desktop.exe")
    parser.add_argument("--frozen-receipt", type=Path, default=QA / "frozen-release-verified/verification.json")
    parser.add_argument("--desktop-receipt", type=Path, default=QA / "desktop-verification.json")
    parser.add_argument("--label", default="human-data-desktop-final-20261114")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("Release label must be a simple directory name")
    executable = args.exe.resolve()
    release = DIST / args.label
    package = DIST / f"ningsi-studio-{args.label}-win-x64.zip"
    if release.exists() or package.exists():
        raise FileExistsError("Use a new release label; existing releases are never overwritten")
    receipt = json.loads(args.frozen_receipt.read_text(encoding="utf-8"))
    desktop = json.loads(args.desktop_receipt.read_text(encoding="utf-8"))
    exe_sha = sha(executable)
    assert exe_sha == receipt["exe_sha256"] == desktop["exe_sha256"]
    assert receipt["api_import_train_report_export_activation"]
    assert desktop["complete_desktop_workflow"] and desktop["active_model_matches_exported_model"]
    assert len(desktop["downloaded_files_match_run_artifacts"]) == 5
    suites = {
        "studio_suite": suite(QA / "studio-regression.log"),
        "engine_suite": suite(WORKSPACE / "var/human-model-acceptance/engine-tests.log"),
        "capture_export_suite": suite(WORKSPACE / "var/human-model-acceptance/capture-export-tests.log"),
        "targeted_human_models_suite": suite(QA / "targeted-final.log"),
    }
    static_log = QA / "static-check.log"
    assert "静态自检通过" in static_log.read_text(encoding="utf-8", errors="replace")
    release.mkdir(exist_ok=False)
    shutil.copy2(executable, release / PACKAGED_EXE_NAME)
    shutil.copy2(PROJECT / "docs/HUMAN_MODEL_TRAINING.md", release / "HUMAN_MODEL_TRAINING.md")
    shutil.copy2(WORKSPACE / "LICENSE", release / "LICENSE.txt")
    text = r"""凝思 Studio：真人数据训练桌面验收版（2026-11-14）

无需安装 Python。解压整个压缩包，双击 ningsi-studio-human-data.exe。
程序启动后会打开本机平台；在“更多工具 → 模型训练”进入“真人数据与模型”。

1. 在独立采集器完成正式研究协议、停止并导出。
2. 在“本机采集目录”粘贴完整路径，例如 D:\实验数据\采集导出；也可点击“选择文件”选择文件夹。
   支持单个导出目录、多个导出的父目录，或采集器合并的数据集目录。原始采集目录需先导出。
3. 点“导入并验收”，核对开发组/封存测试组的人数、标签数量和排除原因。
   点“一键训练与独立评估”，等待出现“训练与独立测试已完成”。也可在填入路径后直接一键训练。
4. 点“预览独立测试报告”在新标签页查看结果；下载完整成果ZIP、模型JSON、HTML报告、评估JSON或预测CSV。
   下载文件保存到浏览器设置的下载目录。
5. 点击“启用此模型（新会话）”；在新会话实时监测页查看模型判断、质量状态和特征依据。

开发组至少5位有效参与者；独立测试组必须有两类有效标签。
建议按前期方案采集30位开发组、30位封存测试组。同一人的所有到访始终属于同一组。
标签0=当前任务专注，1=内部走神。仿真、练习、缺失和不合格探针不进入真人训练。
原始 recording.sqlite3 目录请先在采集器导出，不直接上传原始脑电。
旧合并数据集缺少设备/引擎清单时可训练评估；实时接入请使用完整导出目录。

本版采用可解释的六特征逻辑回归、按被试五折验证、开发组校准与阈值选择，冻结后独立测试。
主要分析中每类至少5个有效标签的测试参与者不足20人时，报告明确标为探索性结果。
原有频带比值指数、预警和神经反馈目标保留，真人模型概率单独展示，不代表绝对能力或疗效。

数据默认保存在EXE同目录的 var\studio，不要把该目录当作临时文件清理。
导入的数据集、后台任务、训练成果和已启用模型都会保存；离开页面再进入仍可查看。
同一研究的数据请使用同一数据目录。同一数据集重复点击会复用任务；封存测试人员不能转入开发组。
沿用旧平台数据时，请先关闭旧版，再从命令行执行：
ningsi-studio-human-data.exe serve --data "你的旧平台数据目录" --open
关闭程序控制台可停止服务。正在运行的会话保留启动时模型，模型启用/停用只影响新会话。
已有模型可在“导入已有模型”选择 attention_model.json 并点击“导入并启用模型”。
设备、采样率、通道、单位或算法版本不兼容时会暂停模型输出，请按提示检查。

命令行复现：
ningsi-studio-human-data.exe train-real --input "采集导出目录" --output "新的训练成果目录"

详细流程、字段、统计口径、模型适用范围及限制见 HUMAN_MODEL_TRAINING.md。
本版已完成桌面验收：路径导入 → 一键训练与独立评估 → 报告预览 → 五类成果下载 → 模型启用及页面重入。
五类下载文件与后台成果逐字节一致，完整成果ZIP的成员校验值全部通过，启用版本与导出的模型一致。
最终EXE已在没有源码路径的隔离工作目录完成命令行训练和HTTP链路验证，无需安装Python。
Studio {studio_tests}项、引擎 {engine_tests}项、采集导出 {capture_tests}项自动化测试通过。
验收使用人为构造的软件测试样例，不是真人准确率结果；发布包不包含样例、原始脑电或已训练测试模型。
真人硬件表现、跨设备泛化与训练疗效仍需按实验方案验证。
"""
    text = text.replace("{studio_tests}", str(suites["studio_suite"]["tests"]))
    text = text.replace("{engine_tests}", str(suites["engine_suite"]["tests"]))
    text = text.replace("{capture_tests}", str(suites["capture_export_suite"]["tests"]))
    (release / "使用说明.txt").write_bytes(("\ufeff" + text.replace("\n", "\r\n")).encode("utf-8"))
    sources = {}
    for root in (WORKSPACE / "ningsi/src", PROJECT / "src", PROJECT / "web"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".js", ".mjs", ".html", ".css", ".sql"):
                sources[path.relative_to(WORKSPACE).as_posix()] = sha(path)
    build = {"app": "ningsi-studio", "product_version": "0.1.2", "build_label": args.label,
             "packaged_utc": datetime.now(timezone.utc).isoformat(), "platform": "Windows x64 onefile",
             "exe_sha256": exe_sha, "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pylsl", "pyinstaller")},
             "source_state": "working tree; source SHA256 manifest below", "source_sha256": sources,
             "signed": False}
    validation = {"exe_sha256": exe_sha, **suites,
                  "static_check": {"passed": True, "log_sha256": sha(static_log)},
                  "frozen_binary": {k: v for k, v in receipt.items() if k not in ("exe", "run_id")},
                  "desktop": desktop,
                  "real_person_or_hardware_model_validation": False,
                  "distribution_contains_eeg_or_test_models": False}
    for name, value in (("BUILD.json", build), ("VALIDATION.json", validation)):
        (release / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    names = {PACKAGED_EXE_NAME, "使用说明.txt", "HUMAN_MODEL_TRAINING.md", "LICENSE.txt", "BUILD.json", "VALIDATION.json"}
    assert {p.name for p in release.iterdir()} == names
    (release / "SHA256SUMS.txt").write_text("".join(f"{sha(release/name)}  {name}\n" for name in sorted(names)), encoding="utf-8")
    with zipfile.ZipFile(package, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names | {"SHA256SUMS.txt"}):
            archive.write(release / name, f"{release.name}/{name}")
    package.with_suffix(".zip.sha256").write_text(f"{sha(package)}  {package.name}\n", encoding="ascii")
    print(json.dumps({"exe": str(release / PACKAGED_EXE_NAME), "exe_sha256": exe_sha, "zip": str(package),
                      "zip_sha256": sha(package), "files": len(names) + 1, "bytes": package.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
