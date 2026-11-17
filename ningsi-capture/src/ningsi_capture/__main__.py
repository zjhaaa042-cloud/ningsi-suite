"""Source and EXE entry points use the same implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from . import __version__
from .bootstrap import ensure_engine


def main(argv=None):
    parser = argparse.ArgumentParser(description="凝思独立实验采集器")
    sub = parser.add_subparsers(dest="command")
    gui = sub.add_parser("gui", help="启动独立桌面采集器（默认）")
    gui.add_argument("--data", type=Path)
    gui.add_argument("--lsl-markers", action="store_true", help="显式开启可选 LSL 任务标记流，供 LabRecorder 另行录制")
    doctor = sub.add_parser("doctor", help="检查采集器和算法依赖")
    doctor.add_argument("--output", type=Path)
    demo = sub.add_parser("demo", help="生成明确标记的合成验收记录，不接真实设备")
    demo.add_argument("--protocol", choices=("fixture", "short"), default="fixture",
                      help="short 用模拟时钟走完短版研究任务，fixture 生成快速导出样例")
    demo.add_argument("--data", type=Path, required=True)
    demo.add_argument("--export", type=Path)
    demo.add_argument("--output", type=Path)
    export = sub.add_parser("export", help="导出一份已结束的采集记录")
    export.add_argument("session", type=Path)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--recover", action="store_true", help="对异常退出记录生成只读快照；不进入训练集")
    export.add_argument("--output", type=Path)
    export.add_argument("--biomulti-db", type=Path, help="BioMulti Lite 的 data.db（只读）")
    export.add_argument("--rec-id", type=int, help="与此段采集同时录制的 BioMulti 原生 REC 编号")
    export.add_argument("--confirmed-unit", choices=("uV", "mV", "V", "nV", "ADC"), help="补录厂商确认的 EEG 单位")
    export.add_argument("--unit-evidence", help="单位确认依据，如手册页或校准记录")
    export.add_argument("--uv-per-count", type=float, help="已确认的 ADC 转 µV 倍率")
    export.add_argument("--reprocess", action="store_true", help="用新版因果滤波和伪迹掩码重新导出旧记录")
    clock_check = sub.add_parser("check-clock", help="只读核对凝思已收到的样本与已保存的 BioMulti REC")
    clock_check.add_argument("session", type=Path)
    clock_check.add_argument("--rec-id", type=int, required=True)
    clock_check.add_argument("--biomulti-db", type=Path)
    clock_check.add_argument("--output", type=Path)
    dataset = sub.add_parser("dataset", help="按被试预分组汇总真实训练与测试数据")
    dataset.add_argument("--exports", nargs="+", type=Path, required=True)
    dataset.add_argument("--out", type=Path, required=True)
    dataset.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    ensure_engine()
    if args.command in (None, "gui"):
        from .gui import run
        run(getattr(args, "data", None), lsl_markers=getattr(args, "lsl_markers", False))
        return 0
    try:
        if args.command == "doctor":
            from ningsi.models.attlapse import load_candidate
            from ningsi.models.attlapse_shadow import load_candidates
            from ningsi.signal.conditioning import POLICY
            from .storage import engine_fingerprint, implementation_fingerprint
            candidate = load_candidate()
            shadow = load_candidates()
            result = {"app": "ningsi-capture", "version": __version__,
                      "attlapse_model_id": candidate["model_id"] if candidate else None,
                      "shadow_250_hz": {"bundle_id": shadow["bundle_id"], "model_ids": shadow["model_ids"]},
                      "modules": {name: importlib.util.find_spec(name) is not None
                                  for name in ("numpy", "pylsl", "tkinter", "ningsi")},
                      "engine_fingerprint": engine_fingerprint(),
                      "preprocessing_policy": POLICY, "implementation_fingerprint": implementation_fingerprint(POLICY)}
            # Test loading the native LSL library, not just finding a module spec.
            from pylsl import library_version
            result["liblsl_version"] = library_version()
        elif args.command == "demo":
            from .demo import create_demo, create_short_demo
            from .export import export_session
            path = create_short_demo(args.data) if args.protocol == "short" else create_demo(args.data)
            result = {"session": str(path), "data_origin": "simulation"}
            if args.export:
                result["export"] = export_session(path, args.export)
        elif args.command == "export":
            from .export import export_session
            calibration = None
            if args.confirmed_unit or args.unit_evidence or args.uv_per_count is not None:
                calibration = {"unit": args.confirmed_unit, "evidence": args.unit_evidence, "factor": args.uv_per_count}
            result = export_session(args.session, args.out, recover=args.recover, native_db=args.biomulti_db,
                                    native_record=args.rec_id, calibration=calibration, reprocess=args.reprocess)
        elif args.command == "check-clock":
            from contextlib import closing
            from .biomulti import align, default_database
            from .storage import open_readonly, read_metadata
            with closing(open_readonly(args.session)) as reader:
                reader.execute("BEGIN")
                _, result, _ = align(reader, read_metadata(reader)["source"],
                                     args.biomulti_db or default_database(), args.rec_id)
        elif args.command == "dataset":
            from .export import build_dataset
            result = build_dataset(args.exports, args.out)
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
        code = 1
    else:
        code = 0
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
