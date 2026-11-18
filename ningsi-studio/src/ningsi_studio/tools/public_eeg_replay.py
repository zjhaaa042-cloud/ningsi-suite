"""Create a native Studio session from the public PhysioNet EEGMMIDB recordings."""

from dataclasses import replace
import json
from pathlib import Path
import re

import numpy as np

from ningsi import config
from ningsi.monitoring.alerts import AlertEngine
from ningsi.signal.baseline import build_baseline
from ningsi_studio import bootstrap
from ningsi_studio.core import phases
from ningsi_studio.core.live_source import ManagedLslSource, SourceInfo, condition_eeg_samples
from ningsi_studio.core.runtime import SessionRuntime
from ningsi_studio.db import repository as repo
from ningsi_studio.db import sqlite_store as store
from ningsi_studio.db.ownership import DataDirectoryLock
from ningsi_studio.domain import indicators
from ningsi_studio.domain.export import sha256_of
from ningsi_studio.settings import Settings


DATASET_URL = "https://physionet.org/content/eegmmidb/1.0.0/"
FILES_URL = "https://physionet.org/files/eegmmidb/1.0.0/"
CHANNELS = ("Fp1", "Fp2")
RATE = 160.0
WAVE_PHASES = (
    ("qc", "R03", "左右手张合任务（起始10秒，用于质检）"),
    ("baseline_open", "R01", "睁眼静息"),
    ("baseline_closed", "R02", "闭眼静息"),
    ("monitor", "R04", "左右手运动想象（含任务内休息）"),
)
SCOPE = "脑电指数描述公开运动想象录波相对该被试睁眼静息基线的变化。单次回放可用于检验报告链路，无法判定受试者的专注、压力或训练效果。"
MISSING = "本次未采集 SAS/SDS、SART、PVT-B、自评或反馈训练。公开录波的任务事件仅用于记录出处。"


def load_recordings(input_dir: Path, subject: str, channels=CHANNELS) -> tuple[dict, dict]:
    """Verify publisher hashes, then decode the selected EEG pair in physical µV."""
    if not re.fullmatch(r"S[0-9]{3}", subject) or not 1 <= int(subject[1:]) <= 109:
        raise ValueError("被试编号应为 PhysioNet S001–S109")
    if len(channels) != 2 or len({name.casefold() for name in channels}) != 2:
        raise ValueError("请选择两个不同的原始脑电导联")
    try:
        from mne.io import read_raw_edf
    except ImportError as exc:
        raise RuntimeError("回放 EDF 需要安装 MNE（python -m pip install mne）") from exc
    hashes = {}
    for line in (input_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition(" ")
        hashes[name.strip()] = digest
    source = {
        "kind": "public_recorded_eeg", "dataset": "EEG Motor Movement/Imagery Dataset v1.0.0",
        "citation": "Gerwin Schalk (2009), PhysioNet, DOI 10.13026/C28G6P",
        "url": DATASET_URL, "license": "Open Data Commons Attribution License v1.0",
        "participant": subject, "sample_rate_hz": RATE,
        "sample_unit": "µV",
        "timing": "按 EDF 原采样点顺序，每窗4秒、步长2秒。各阶段取独立录波，阶段拼接时间轴仅用于报告展示。",
        "scope": SCOPE, "missing": MISSING, "segments": [],
    }
    recordings = {}
    for phase, run, condition in WAVE_PHASES:
        name = f"{subject}{run}.edf"
        relpath = f"{subject}/{name}"
        path = input_dir / relpath
        if not path.is_file():
            path = input_dir / name
        digest = sha256_of(path)
        if digest != hashes.get(relpath):
            raise ValueError(f"EDF 未通过 PhysioNet SHA-256 校验：{relpath}")
        raw = read_raw_edf(path, verbose="ERROR")
        try:
            if raw.info["sfreq"] != RATE or len(raw.ch_names) != 64:
                raise ValueError(f"EDF 与发布的 64 通道、160 Hz 规格不符：{relpath}")
            candidates = {name.rstrip(".").casefold(): name for name in raw.ch_names}
            names = [candidates.get(label.casefold()) for label in channels]
            if None in names:
                raise ValueError(f"EDF 缺少所选导联 {'/'.join(channels)}：{relpath}")
            labels = [name.rstrip(".") for name in names]
            if source.setdefault("channel_labels", labels) != labels:
                raise ValueError(f"EDF 中的导联名称与同次回放其他片段不一致：{relpath}")
            values = raw.get_data(picks=names, units="uV")
            available = values.shape[1] / RATE
        finally:
            raw.close()
        if phase == "qc":
            values = values[:, :round(10 * RATE)]
        width = round(config.WINDOW_SEC * RATE)
        hop = round(config.STEP_SEC * RATE)
        windows = (values.shape[1] - width) // hop + 1
        if windows < 1 or not np.isfinite(values).all():
            raise ValueError(f"EDF 样点不足或包含非有限数值：{relpath}")
        used = width + (windows - 1) * hop
        recordings[phase] = values[:, :used]
        source["segments"].append({
            "phase": phase, "condition": condition, "filename": name,
            "url": FILES_URL + relpath, "sha256": digest, "edf_channels": names,
            "recorded_sec": available, "samples_per_channel": used,
            "first_sample": 0, "end_sample_exclusive": used, "analyzed_sec": used / RATE,
            "windows": windows,
        })
    return recordings, source


class RecordedEEG:
    """Move exactly one hop per analysis call. There is no looping across recordings."""

    def __init__(self, recordings, labels):
        self.recordings = recordings
        self.channel_labels = tuple(labels)
        self.width = round(RATE * config.WINDOW_SEC)
        self.hop = round(RATE * config.STEP_SEC)
        self.context = round(RATE * ManagedLslSource.condition_seconds)
        self.last_condition = {}

    def select(self, phase):
        self.values = self.recordings[phase]
        self.end = self.width - self.hop

    def analysis_window(self, _state, *, artifact=False):
        self.end += self.hop
        if artifact or self.end > self.values.shape[1]:
            raise RuntimeError("回放分析步数超过所选原始录波")
        raw = self.values[:, max(0, self.end - self.context):self.end]
        wave, evidence, self.last_condition = condition_eeg_samples(
            raw, self.width, RATE, labels=self.channel_labels, units=("uV", "uV"),
        )
        return wave, evidence

    def conditioning(self):
        return self.last_condition


class ReplaySession(SessionRuntime):
    def __init__(self, row, subject_row, recordings, provenance, settings):
        super().__init__(row["uuid"], settings=settings)
        self.session = row
        self.subject = subject_row
        self.protocol = phases.PROFILE_SHORT
        self.profile = phases.profile_dict(self.protocol)
        self.profile_phase_keys = phases.REPLAY_PHASES
        self.skipped_phases = ["scales", "sart", "pvt", "training", "model"]
        self.eeg_calibration = None  # EDF decoding already provides physical µV.
        self.provenance = provenance
        labels = provenance["channel_labels"]
        recorded = RecordedEEG(recordings, labels)
        self.source = SourceInfo(row["source"], "replay", RATE, len(labels), row["device"],
                                 f"PhysioNet {provenance['participant']} · {'/'.join(labels)} 公开真人录波", recorded)
        self.engine = indicators.WindowEngine(recorded, RATE, device=row["device"])

    def elapsed_now(self):
        return self.engine.t

    def _affect_observation(self, step, phase):
        # The public motor imagery task does not contain paired check-ins.
        return None

    def _run(self):
        device = self.source.device
        for segment in self.provenance["segments"]:
            phase = segment["phase"]
            self.source.engine.select(phase)
            self.engine.alerts = AlertEngine()
            # Start the 4 s window at the first sample of each recording. Report
            # time joins the analyzed intervals, while provenance retains each EDF clock.
            segment["report_offset_sec"] = self.engine.t
            self.engine.t += config.WINDOW_SEC - config.STEP_SEC
            with self._phase(phase) as context:
                if phase == "qc":
                    qc = self.engine.device_qc(
                        artifact=False, state="unknown", on_step=lambda step: self._publish_window(
                            step, phase="qc", include_signal=False))
                    self._merge_run_payload(context["run_id"], qc)
                    continue
                self.engine.baseline = self.baselines.get("baseline_open") if phase == "monitor" else None
                steps = []
                for _ in range(segment["windows"]):
                    self._check_cancel()
                    step = self.engine.step("unknown")
                    steps.append(step)
                    self._publish_window(step, phase=phase, include_signal=False)
                quality = self.engine.summarize(steps)
                self._persist_summary_metrics(quality, phase=phase)
                if phase == "monitor":
                    monitor = {"quality": quality, "summary": quality.get("indicators", {}), "steps": steps}
                    self._merge_run_payload(context["run_id"], {"quality": quality})
                else:
                    baseline = build_baseline((step.window for step in steps), device=device, srate=RATE,
                                              min_windows=config.BASELINE_PROTOCOL["min_windows"])
                    self.baselines[phase] = baseline
                    self._merge_run_payload(context["run_id"], {
                        "baseline": baseline.as_dict(), "windows": len(steps), "valid_ratio": quality["valid_ratio"],
                    })

        baseline = self.baselines["baseline_open"]
        with self._phase("assessment") as context:
            assessment = self._run_assessment(baseline, {}, {}, monitor["summary"], monitor["quality"])
            assessment = replace(
                assessment, conclusion="公开脑电回放完成，联合评估缺少配对量表和行为数据",
                advice=("如需验证联合评估，应采集同一参与者的凝思协议脑电、量表和行为数据。",),
            )
            self._merge_run_payload(context["run_id"], assessment.as_dict())
        with self._phase("report") as context:
            provenance_path = self.runs_root / "source.json"
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            provenance_path.write_text(json.dumps(self.provenance, ensure_ascii=False, indent=2), encoding="utf-8")
            self.artifacts["source"] = provenance_path
            baselines = {segment["phase"]: segment["analyzed_sec"] for segment in self.provenance["segments"]}
            report_context = {
                "replay": self.provenance, "protocol_label": phases.REPLAY_LABEL, "protocol_note": "",
                "baseline_protocol": {**config.BASELINE_PROTOCOL, "eyes_open_sec": baselines["baseline_open"],
                                      "eyes_closed_sec": baselines["baseline_closed"]},
            }
            paths = self._write_report(baseline, {}, {}, {}, monitor, None, assessment, report_context=report_context)
            self._merge_run_payload(context["run_id"], {kind: str(path) for kind, path in paths.items()})
        self._finish("done", None)


def run(input_dir: Path, subject: str, data_dir: Path, channels=CHANNELS) -> dict:
    """Generate a report in its own Studio data directory and return the session ID."""
    recordings, provenance = load_recordings(input_dir, subject, channels)
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=False)
    settings = Settings(data_dir=data_dir)
    settings.ensure_dirs()
    device = f"replay:PhysioNet-EEGMMIDB-{subject}-{'-'.join(provenance['channel_labels'])}"
    participant = f"physionet-{subject.lower()}"
    with DataDirectoryLock(data_dir):
        store.initialize(settings.db_path)
        with store.connect(settings.db_path) as conn:
            subject_id = repo.create_subject(
                conn, participant, label=f"公开被试 {subject}", consent_version="public-odc-by-1.0",
                note=f"使用发布方公开许可导入的去标识录波。{DATASET_URL}",
            )
            row = repo.create_session(conn, subject_id, label=phases.REPLAY_LABEL, device=device, source=device,
                                      srate=RATE, channels=len(channels), protocol=phases.PROFILE_SHORT,
                                      engine_versions=bootstrap.engine_versions())
            subject_row = repo.get_subject_by_pk(conn, subject_id)
        runtime = ReplaySession(row, subject_row, recordings, provenance, settings)
        runtime.start()
        try:
            runtime.join()
        except KeyboardInterrupt:
            runtime.cancel()
            runtime.join()
            raise
        with store.read_only(settings.db_path) as conn:
            final = repo.get_session(conn, row["uuid"])
            artifacts = {item["kind"]: item["path"] for item in repo.list_artifacts(conn, row["id"])}
        if final["status"] != "done" or not {"report_html", "report_md", "report_json", "heatmap_svg", "trend_svg",
                                            "history", "source"}.issubset(artifacts):
            raise RuntimeError(final["error"] or "Studio 报告或产物登记不完整")
    return {"uuid": row["uuid"], "status": final["status"], "data_dir": str(data_dir), "artifacts": artifacts}
