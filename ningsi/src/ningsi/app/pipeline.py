"""端到端会话流水线：设备质检 → 量表 → 基线 → 行为任务 → 指标监测 → 联合评估 → 训练 → 报告。

默认使用仿真脑电源，便于无设备联调、演示与自动化测试；接入真实设备时把 eeg 换成 LSL 源即可，
后续各步不感知数据来源。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ningsi import config
from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.adapters import studio
from ningsi.assessment import joint
from ningsi.assessment.report import SessionReport
from ningsi.behavior.pvt import simulate_result as simulate_pvt
from ningsi.behavior.sart import SartResult, build_sequence
from ningsi.models import logistic as features
from ningsi.monitoring import history as history_module
from ningsi.monitoring.alerts import AlertEngine
from ningsi.monitoring.heatmap import render_svg as heatmap_svg
from ningsi.scales.scoring import score_scale
from ningsi.scales.store import save_scale_run as save_scale
from ningsi.signal.baseline import build_baseline
from ningsi.signal.indicators import aggregate, compute_indicators, summarize_indicators
from ningsi.signal.window import analyze_window
from ningsi.training.neurofeedback import NeurofeedbackSession, initial_target


def matrix_np(rows, indices):
    import numpy as np
    return np.array([rows[index] for index in indices], dtype=float)


def labels_np(labels, indices):
    import numpy as np
    return np.array([labels[index] for index in indices], dtype=int)


@dataclass
class SessionConfig:
    participant: str = "p01"
    session: str = "01"
    run: str = "001"
    root: str = "var/session"
    device: str = "sim-bsense"
    srate: float = 250.0
    channels: int = 1
    seed: int = 7
    training_mode: str = "quick"
    baseline_seconds: tuple = (config.BASELINE_PROTOCOL["eyes_open_sec"], config.BASELINE_PROTOCOL["eyes_closed_sec"])
    task_plan: tuple = (("rest", 8), ("focused", 12), ("drowsy", 15))
    sas_answers: tuple = tuple([2] * 20)
    sds_answers: tuple = tuple([2] * 20)
    sad_answers: tuple = ()
    sart_rt: float = 0.32
    sart_commission: int = 0
    pvt_lapse_probability: float = 0.05


@dataclass
class SessionArtifacts:
    config: SessionConfig
    quality: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)
    behavior: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    assessment: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "participant": self.config.participant,
            "session": self.config.session,
            "run": self.config.run,
            "device": self.config.device,
            "quality": self.quality,
            "baseline": self.baseline,
            "scales": {k: v.get("standard_score") for k, v in self.scales.items()},
            "behavior": {k: v.get("rt_mean", v.get("rt_median")) for k, v in self.behavior.items()},
            "indicators": self.indicators,
            "alerts": self.alerts,
            "assessment": self.assessment,
            "training": {k: self.training.get(k) for k in ("mean_focus", "on_target_ratio", "first_to_last_change", "final_target")},
            "model": {
                "spec": self.model.get("spec"),
                "train_accuracy": (self.model.get("train") or {}).get("accuracy"),
                "validation_accuracy": (self.model.get("validation") or {}).get("accuracy"),
                "test_accuracy": (self.model.get("test") or {}).get("accuracy"),
                "test_auc": (self.model.get("test") or {}).get("auc"),
                "subject_split": self.model.get("subject_split"),
                "model_path": self.model.get("model_path"),
            },
            "paths": self.paths,
        }


class SessionRunner:
    def __init__(self, config_in: SessionConfig | None = None, eeg=None) -> None:
        self.config = config_in or SessionConfig()
        self.eeg = eeg or SyntheticEEG(srate=self.config.srate, channels=self.config.channels, seed=self.config.seed)
        self.root = Path(self.config.root)

    # ---------- 各步骤 ----------
    def device_qc(self) -> dict:
        windows = [analyze_window(self.eeg.window("rest"), self.config.srate, t_end=index * config.STEP_SEC)
                   for index in range(4)]
        windows.append(analyze_window(self.eeg.window("rest", artifact=True), self.config.srate, t_end=8.0))
        usable = [w for w in windows if w.usable]
        reasons: dict[str, int] = {}
        for window in windows:
            for reason in window.quality.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "windows": len(windows),
            "usable": len(usable),
            "valid_ratio": round(len(usable) / len(windows), 4),
            "reasons": reasons,
            "passed": len(usable) >= 3,
        }

    def collect_baseline(self, seconds: float | None = None, state: str = "rest"):
        """按秒数采集基线窗；4 秒窗、2 秒步长，故窗数 = 秒数 / 2。"""
        duration = float(seconds if seconds is not None else self.config.baseline_seconds[0])
        count = max(1, int(round(duration / config.STEP_SEC)))
        return [analyze_window(self.eeg.window(state), self.config.srate, t_end=index * config.STEP_SEC)
                for index in range(count)]

    def collect_baselines(self) -> dict:
        """睁眼与闭眼静息基线各自独立质检与建基线（文档 7.2.4、8.4.2）。"""
        open_seconds, closed_seconds = self.config.baseline_seconds
        eyes_open = build_baseline(self.collect_baseline(open_seconds, "rest"),
                                   device=self.config.device, srate=self.config.srate,
                                   min_windows=config.BASELINE_PROTOCOL["min_windows"])
        eyes_closed = build_baseline(self.collect_baseline(closed_seconds, "eyes_closed"),
                                     device=self.config.device, srate=self.config.srate,
                                     min_windows=config.BASELINE_PROTOCOL["min_windows"])
        return {"eyes_open": eyes_open, "eyes_closed": eyes_closed}

    def run_scales(self) -> dict:
        results = {}
        responses = {}
        if self.config.sas_answers:
            results["SAS"] = score_scale("SAS", self.config.sas_answers)
            responses["SAS"] = list(self.config.sas_answers)
        if self.config.sds_answers:
            results["SDS"] = score_scale("SDS", self.config.sds_answers)
            responses["SDS"] = list(self.config.sds_answers)
        for code, result in results.items():
            save_scale(
                self.root, self.config.participant, self.config.session, self.config.run,
                result, responses.get(code), scale_code=code,
            )
        return {code: result.as_dict() for code, result in results.items()}

    def run_behavior(self) -> dict:
        sequence = build_sequence(self.config.participant, self.config.session, self.config.run)
        nogo = set(sequence.nogo_positions[: self.config.sart_commission])
        responded, rts = [], []
        for index in range(sequence.trials):
            hit = index not in sequence.nogo_positions or index in nogo
            responded.append(hit)
            rts.append(self.config.sart_rt if hit else None)
        sart = SartResult(
            self.config.participant, self.config.session, self.config.run,
            sequence, tuple(responded), tuple(rts),
        ).score()
        pvt = simulate_pvt(seed=self.config.seed + 1, lapse_probability=self.config.pvt_lapse_probability).score()
        return {"sart": sart, "pvt": pvt}

    def run_task(self, baseline) -> dict:
        engine = AlertEngine()
        windows, results, series = [], [], []
        t_end = 100.0
        for state, count in self.config.task_plan:
            for _ in range(count):
                t_end += config.STEP_SEC
                window = analyze_window(self.eeg.window(state), self.config.srate, t_end=t_end)
                result = compute_indicators(window, baseline) if window.usable else None
                windows.append(window)
                if result is not None:
                    results.append(result)
                series.append((window.t_end, None if not window.usable or result is None else result.score("focus")))
                engine.feed(window, result)
        quality = aggregate(windows, results)
        return {
            "windows": windows,
            "results": results,
            "quality": quality,
            "indicators": summarize_indicators(results),
            "band_z": results[-1].band_z if results else {},
            "series": series,
            "alerts": [event.as_dict() for event in engine.history if event.state == "triggered"],
        }

    def train_baseline_model(self, subjects: int = 6, windows_per_state: int = 8) -> dict:
        """在仿真被试上训练可解释基线模型，并按被试划分评估（保证被隔离）。"""
        matrix, labels, participants = [], [], []
        for index in range(subjects):
            sim = SyntheticEEG(srate=self.config.srate, channels=self.config.channels, seed=200 + index)
            for state, label in (("focused", 1), ("drowsy", 0)):
                for _ in range(windows_per_state):
                    window = analyze_window(sim.window(state), self.config.srate)
                    row = features.features_from_window(window)
                    matrix.append([row[name] for name in features.FEATURE_NAMES])
                    labels.append(label)
                    participants.append(f"sim{index:02d}")
        split = features.subject_split(participants, seed=self.config.seed)
        train = [i for i, name in enumerate(participants) if name in split["train"]]
        validation = [i for i, name in enumerate(participants) if name in split["validation"]]
        test = [i for i, name in enumerate(participants) if name in split["test"]]
        model = features.fit_logistic(matrix_np(matrix, train), labels_np(labels, train),
                                      trained_subjects=tuple(split["train"]))
        metrics = {
            "spec": model.spec,
            "features": list(features.FEATURE_NAMES),
            "subject_split": split,
            "train": features.evaluate(model, matrix_np(matrix, train), labels_np(labels, train)),
            "validation": features.evaluate(model, matrix_np(matrix, validation), labels_np(labels, validation)) if validation else None,
            "test": features.evaluate(model, matrix_np(matrix, test), labels_np(labels, test)) if test else None,
        }
        model_path = self.root / "models" / "classifier.json"
        model.save(model_path)
        metrics["model_path"] = str(model_path)
        return metrics

    def run_training(self, baseline_before) -> dict:
        target, rationale = initial_target(baseline_before)
        session = NeurofeedbackSession(
            self.config.participant, self.config.session, self.config.run,
            mode=self.config.training_mode, target=target, rationale=rationale,
        )
        plan = session.plan()
        states = ["focused", "drowsy"]
        for index in range(plan["segments"]):
            state = states[index % len(states)]
            wins = [analyze_window(self.eeg.window(state), self.config.srate, t_end=500.0 + index * 120.0 + step * config.STEP_SEC)
                    for step in range(10)]
            results = [compute_indicators(w, baseline_before) for w in wins]
            samples = [(w.t_end, r.score("focus")) for w, r in zip(wins, results)]
            session.add_segment(samples, plan["segment_sec"], excluded_windows=sum(1 for w in wins if not w.usable))
        after_windows = self.collect_baseline(seconds=30.0)
        baseline_after = build_baseline(after_windows, device=self.config.device, srate=self.config.srate)
        return session.summary(baseline_before, baseline_after)

    # ---------- 全流程 ----------
    def run(self) -> SessionArtifacts:
        self.root.mkdir(parents=True, exist_ok=True)
        artifacts = SessionArtifacts(config=self.config)

        artifacts.quality = self.device_qc()
        artifacts.scales = self.run_scales()
        baselines = self.collect_baselines()
        baseline = baselines[config.BASELINE_PROTOCOL["task_reference"]]
        artifacts.baseline = {
            "protocol": dict(config.BASELINE_PROTOCOL),
            "reference": config.BASELINE_PROTOCOL["task_reference"],
            "eyes_open": baselines["eyes_open"].as_dict(),
            "eyes_closed": baselines["eyes_closed"].as_dict(),
        }
        artifacts.behavior = self.run_behavior()

        task = self.run_task(baseline)
        artifacts.indicators = {"summary": task["indicators"], "band_z": task["band_z"], "quality": task["quality"]}
        artifacts.alerts = task["alerts"]

        eeg_summary = {
            "available": baseline.valid,
            "valid_ratio": task["quality"].get("valid_ratio", 0.0),
            "focus": task["indicators"].get("focus", {}).get("mean"),
            "relax": task["indicators"].get("relax", {}).get("mean"),
            "load": task["indicators"].get("load", {}).get("mean"),
            "spec": config.INDICATOR_SPEC,
            "baseline_spec": baseline.spec,
        }
        scale_objects = {"SAS": score_scale("SAS", self.config.sas_answers)} if self.config.sas_answers else {}
        if self.config.sds_answers:
            scale_objects["SDS"] = score_scale("SDS", self.config.sds_answers)
        assessment = joint.assess(eeg_summary, scale_objects, {"sart": artifacts.behavior["sart"], "pvt": artifacts.behavior["pvt"]})
        artifacts.assessment = assessment.as_dict()

        report = SessionReport(
            participant=self.config.participant,
            session=self.config.session,
            run=self.config.run,
            indicators=artifacts.indicators,
            quality=task["quality"],
            scales=artifacts.scales,
            behavior=artifacts.behavior,
            assessment=assessment,
            extras={"device": self.config.device, "srate": self.config.srate, "channels": self.config.channels,
                    "alerts": artifacts.alerts,
                    "baseline_protocol": artifacts.baseline["protocol"],
                    "baseline_reference": artifacts.baseline["reference"],
                    "baseline_eyes_open": artifacts.baseline["eyes_open"],
                    "baseline_eyes_closed": artifacts.baseline["eyes_closed"]},
        )
        reports = self.root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        stem = f"sub-{self.config.participant}_ses-{self.config.session}_run-{self.config.run}"
        md_path = reports / f"{stem}_report.md"
        json_path = reports / f"{stem}_report.json"
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        json_path.write_text(report.to_json(), encoding="utf-8")
        heat_path = reports / f"{stem}_heatmap.svg"
        heat_path.write_text(heatmap_svg(task["series"], title=f"状态热力图 sub-{self.config.participant}"), encoding="utf-8")

        artifacts.training = self.run_training(baseline)
        artifacts.model = self.train_baseline_model()

        history_records = history_module.load_records(self.root)
        _, rejected = history_module.comparable(history_records, self.config.device, self.config.srate, self.config.channels)
        history_path = history_module.append_record(self.root, {
            "participant": self.config.participant,
            "session": self.config.session,
            "run": self.config.run,
            "device": self.config.device,
            "srate": self.config.srate,
            "channels": self.config.channels,
            "indicators": {k: v["mean"] for k, v in task["indicators"].items() if isinstance(v, dict) and "mean" in v},
            "quality": {"valid_ratio": task["quality"].get("valid_ratio"), "passed": artifacts.quality["passed"]},
            "behavior": {"sart_rt_mean": artifacts.behavior["sart"]["rt_mean"], "pvt_rt_median": artifacts.behavior["pvt"]["rt_median"]},
            "assessment": {"conclusion": assessment.conclusion},
        })
        trend_path = reports / "trend.svg"
        records, _ = history_module.comparable(history_module.load_records(history_path), self.config.device, self.config.srate, self.config.channels)
        points = history_module.aggregate(records, "focus", "week")
        trend_path.write_text(history_module.render_svg_trend(points, title="专注度周趋势"), encoding="utf-8")

        artifacts.paths = {
            "report_md": str(md_path),
            "report_json": str(json_path),
            "heatmap_svg": str(heat_path),
            "trend_svg": str(trend_path),
            "history": str(history_path),
            "root": str(self.root),
        }
        return artifacts


def run_session(config_in: SessionConfig | None = None, eeg=None) -> SessionArtifacts:
    return SessionRunner(config_in, eeg).run()


def import_studio_history(dataset_root, root) -> int:
    """把上游试点采集记录并入凝思历史，用于真实数据的趋势与证据回填。"""
    return studio.import_from_studio(dataset_root, lambda record: history_module.append_record(root, record))
