"""命令行入口：端到端演示、导入上游记录、启动桌面界面、运行自测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ningsi import config
from ningsi.app.pipeline import SessionConfig, import_studio_history, run_session


def _print_summary(artifacts) -> None:
    data = artifacts.as_dict()
    print(f"凝思会话完成：sub-{data['participant']} / ses-{data['session']} / run-{data['run']}")
    print(f"  设备质检：{artifacts.quality['usable']}/{artifacts.quality['windows']} 窗通过"
          f"（不可用原因 {artifacts.quality['reasons'] or '无'}），判定 {'通过' if artifacts.quality['passed'] else '未通过'}")
    open_base = artifacts.baseline["eyes_open"]
    closed_base = artifacts.baseline["eyes_closed"]
    print(f"  静息基线：睁眼 {'有效' if open_base['valid'] else '无效'}（{open_base['n_windows']} 窗）"
          f"＋闭眼 {'有效' if closed_base['valid'] else '无效'}（{closed_base['n_windows']} 窗）"
          f"；任务态以{artifacts.baseline['reference']}为参照")
    for code, result in sorted(artifacts.scales.items()):
        print(f"  量表 {code}：粗分 {result['raw_score']}，标准分 {result['standard_score']}，{result['level']}")
    sart = artifacts.behavior.get("sart", {})
    pvt = artifacts.behavior.get("pvt", {})
    print(f"  SART：正确率 {sart.get('go_accuracy', 0):.0%}，虚报率 {sart.get('commission_rate', 0):.1%}，"
          f"反应时 {sart.get('rt_mean', 0):.3f}s，序列集 {sart.get('sequence_set_id')}")
    print(f"  PVT-B：中位反应时 {pvt.get('rt_median', 0):.3f}s，慢反应率 {pvt.get('lapse_rate', 0):.1%}")
    summary = artifacts.indicators.get("summary", {})
    quality = artifacts.indicators.get("quality", {})
    print("  状态指标：" + "；".join(
        f"{label} {summary.get(key, {}).get('mean', float('nan')):.2f}"
        for key, label in (("focus", "专注度"), ("relax", "放松度"), ("load", "认知负荷"))
    ))
    print(f"  可用窗比例：{quality.get('valid_ratio', 0):.0%}"
          f"（{quality.get('windows_usable', 0)}/{quality.get('windows_total', 0)}），"
          f"不可用原因 {quality.get('unusable_reasons') or '无'}")
    print(f"  预警：{'、'.join(a['kind'] for a in artifacts.alerts) or '本次无触发'}")
    print(f"  联合评估：{artifacts.assessment.get('conclusion')}")
    model = artifacts.model or {}
    test_metrics = model.get("test") or {}
    if test_metrics:
        print(f"  基线模型：{model.get('spec')}｜被试级划分 {model.get('subject_split')}"
              f"｜留出被试准确率 {test_metrics.get('accuracy')}、AUC {test_metrics.get('auc')}"
              f"｜模型文件 {model.get('model_path')}")
    training = artifacts.training
    print(f"  训练闭环：目标 {training.get('final_target')}，达标时间占比 {training.get('on_target_ratio')}，"
          f"首末段变化 {training.get('first_to_last_change')}")
    print("  产出文件：")
    for key, value in artifacts.paths.items():
        if key != "root":
            print(f"    - {key}: {value}")


def _demo(args) -> int:
    artifacts = run_session(SessionConfig(
        participant=args.participant,
        session=args.session,
        run=args.run,
        root=args.root,
        training_mode=args.mode,
    ))
    _print_summary(artifacts)
    if args.json:
        print(json.dumps(artifacts.as_dict(), ensure_ascii=False, indent=2))
    return 0


def _import_studio(args) -> int:
    count = import_studio_history(args.dataset_root, args.root)
    print(f"已从 {args.dataset_root} 导入 {count} 条上游记录到 {Path(args.root) / 'history' / 'sessions.jsonl'}")
    return 0


def _ui(args) -> int:
    from ningsi.app.ui import launch
    launch(root=args.root, participant=args.participant)
    return 0


def _self_test(args) -> int:
    import unittest
    # app/cli.py -> app -> ningsi -> src -> <repo>/tests
    tests_dir = Path(__file__).resolve().parents[3] / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests_dir))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ningsi", description="凝思：专注力强化训练与心理状态评估系统")
    parser.add_argument("--version", action="version", version=f"ningsi {config.VERSION}")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="用仿真脑电源跑通一次端到端会话")
    demo.add_argument("--participant", default="p01")
    demo.add_argument("--session", default="01")
    demo.add_argument("--run", default="001")
    demo.add_argument("--root", default="var/session")
    demo.add_argument("--mode", default="quick", choices=sorted(config.TRAINING))
    demo.add_argument("--json", action="store_true", help="同时输出结构化 JSON")
    demo.set_defaults(func=_demo)

    imp = sub.add_parser("import-studio", help="导入 bsense-dataset-studio 的试采记录")
    imp.add_argument("--dataset-root", required=True)
    imp.add_argument("--root", default="var/session")
    imp.set_defaults(func=_import_studio)

    ui = sub.add_parser("ui", help="启动桌面界面（指标面板 / 热力图 / 训练 / 报告 / 趋势）")
    ui.add_argument("--root", default="var/session")
    ui.add_argument("--participant", default="p01")
    ui.set_defaults(func=_ui)

    test = sub.add_parser("self-test", help="运行单元测试与全链路用例")
    test.set_defaults(func=_self_test)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
