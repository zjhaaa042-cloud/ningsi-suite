import csv
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from .test_support import FakeClock
from ningsi.signal.artifacts import Verdict
from ningsi_capture.acquisition import simulation_source, synthetic_samples
from ningsi_capture.export import export_session
from ningsi_capture.live_quality import CaptureQuality, QualityMonitor, describe_window
from ningsi_capture.readiness import line_power_ratios, formal_probe_summary
from ningsi_capture.storage import SessionStore, open_readonly


class LiveQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = FakeClock()
        source = {**simulation_source(), "channel_labels": ["Left", "Right"]}
        self.store = SessionStore(self.temp.name, participant="QCTEST", source=source, protocol={}, clock=self.clock)
        self.addCleanup(self.store.close, "aborted")
        self.connection = open_readonly(self.store.path)
        self.addCleanup(self.connection.close)
        self.quality = CaptureQuality()

    def append(self, seconds, *, flat=False, offset=0.):
        indices = np.arange(self.store.sample_count, self.store.sample_count + round(seconds * 250))
        values = synthetic_samples(indices)
        if flat:
            values[:, 1] = 0
        self.clock.advance(seconds)
        self.store.append(values, 1000 + indices / 250, offset=offset)

    def latest(self):
        target = self.store.sample_count
        while True:
            last_index = self.quality.next_index
            result = self.quality.update(self.connection, self.store.metadata)
            if self.quality.next_index == last_index or self.quality.next_index >= target:
                return result

    def test_unaligned_chunk_and_channel_reasons_match_export(self):
        self.append(4.12)
        result = self.latest()
        self.assertIn("等待 2 秒滤波历史", result.reading)
        self.append(2)
        result = self.latest()
        self.assertEqual(result.reading, "通过")
        self.assertAlmostEqual(result.window_end, 1006)
        self.append(8, flat=True)
        result = self.latest()
        self.assertIn("通道近似平线", result.reading)
        self.assertIn("Right 标准差 0 µV", result.reading)
        self.append(8, offset=None)
        self.assertIn("LSL 时钟尚未同步", self.latest().reading)

    def test_disconnect_and_delayed_timestamps_invalidate_green_state(self):
        self.append(6)
        result = self.latest()
        self.assertIn("模拟源 · 4 秒窗质控  通过", result.operator_text(self.clock.now(), self.clock.seconds, True))
        self.clock.advance(3.1)
        self.assertIn("信号超过 3 秒未更新", result.operator_text(self.clock.now(), self.store.last_received, True))
        self.assertIn("等待最新窗口", result.operator_text(self.clock.now(), self.clock.seconds, True))
        self.append(8, offset=-20.)
        delayed = self.latest().operator_text(self.clock.now(), self.clock.seconds, False)
        self.assertIn("窗口时间与当前时钟", delayed)
        self.assertNotIn("通过", delayed)

    def test_rapid_waveform_change_label_describes_only_the_observation(self):
        verdict = Verdict(False, ("motion_transient",))
        self.assertEqual(describe_window(verdict, self.store.metadata["source"]), "未通过 · 波形突变")

    def test_preflight_uses_raw_samples_before_qc_and_expires_on_disconnect(self):
        grid = np.arange(1000) / 250
        data = np.column_stack((700 * np.sin(2 * np.pi * 50 * grid) + 12000,
                                15 * np.sin(2 * np.pi * 10 * grid)))
        self.clock.advance(4)
        self.store.append(data, 1000 + grid, offset=None)
        snapshot = self.latest()
        measure = snapshot.preflight
        self.assertEqual((measure.end_index, measure.count, measure.passed), (999, 1, 0))
        self.assertGreater(measure.ratios[0], .99)
        self.assertLess(measure.ratios[1], .001)
        self.assertIn("工频占多数", snapshot.operator_text(self.clock.now(), self.store.last_received, False))
        self.clock.advance(3.1)
        self.assertIsNone(snapshot.latest_preflight(self.clock.now(), self.store.last_received))
        self.assertNotIn("50 Hz 占比", snapshot.operator_text(self.clock.now(), self.store.last_received, False))
        self.append(36)
        next_snapshot = self.latest()
        self.assertEqual((next_snapshot.preflight.count, next_snapshot.preflight.passed), (14, 14))
        self.assertNotIn("工频占多数", next_snapshot.preflight.text())
        self.store.close("aborted")
        out = Path(self.temp.name) / "export"
        export_session(self.store.path, out)
        with (out / "windows.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["usable"], "False")
        self.assertEqual(float(rows[0]["raw_50hz_ratio_ch1"]), measure.ratios[0])
        self.assertEqual(float(rows[-1]["raw_50hz_ratio_ch2"]), next_snapshot.preflight.ratios[1])

    def test_unmeasurable_channel_and_low_sample_rate_do_not_claim_clean_signal(self):
        data = np.zeros((1000, 3))
        data[:, 1] = np.nan
        data[:, 2] = np.sin(2 * np.pi * 50 * np.arange(1000)/250)
        measures = line_power_ratios(data, 250)
        self.assertEqual(measures[:2], (None, None))
        self.assertGreater(measures[2], .99)
        self.assertEqual(line_power_ratios(data, 100), (None, None, None))

    def test_export_counts_trainable_classes_separately_from_self_report(self):
        def row(label, *, practice=False, usable=True, train=False, test=False):
            return dict(label=label, practice=practice, binary_label_valid=label is not None,
                        feature_usable=usable, train_eligible=train, test_eligible=test)
        counts = formal_probe_summary([row(0, train=True), row(1, usable=False), row(1, test=True),
                                      row(1, practice=True), row(None), row(0)])
        self.assertEqual(counts, dict(formal_probes=5, on_task=2, internal_mind_wandering=2,
                                     usable_eeg=4, paired=3, eligible_on_task=1, eligible_internal_mind_wandering=1))

    def test_baseline_and_block_tally_match_export_without_a_model(self):
        self.store.event("phase_start", {"phase": "baseline_calibration"})
        self.append(32)
        self.assertIn("基线 采集中", self.latest().summary)
        self.store.event("phase_end", {"phase": "baseline_calibration"})
        block = {"phase": "sart_1", "block": 1, "practice": False}
        self.store.event("phase_start", {**block, "spec": {"kind": "sart"}})
        for probe_id, choice, flat in (("a", 1, False), ("b", 2, False), ("c", 5, True)):
            self.append(16, flat=flat)
            self.store.event("probe_onset", {**block, "probe_id": probe_id})
            self.store.event("probe_response", {**block, "probe_id": probe_id, "choice": choice})
        snapshot = self.latest()
        summary = snapshot.summary
        self.assertIn("睁眼基线 通过", summary)
        self.assertIn("已提示 3，脑电可用 2，同时有二分类回答 1", summary)
        self.assertIn("脑电配对专注 1，内部走神 0", snapshot.probe_summary)
        self.store.close("completed")
        output = Path(self.temp.name) / "export"
        stats = export_session(self.store.path, output)["training_readiness"]
        self.assertEqual((stats["on_task"], stats["internal_mind_wandering"], stats["paired"]), (1, 1, 1))
        self.assertEqual((stats["eligible_on_task"], stats["eligible_internal_mind_wandering"]), (0, 0))
        baseline = json.loads((output / "baseline.json").read_text(encoding="utf8"))
        with (output / "features.csv").open(encoding="utf-8-sig", newline="") as file:
            features = list(csv.DictReader(file))
        self.assertIn(f"合格 {baseline['n_windows']}/", summary)
        self.assertEqual(sum(row["feature_usable"] == "True" for row in features), 2)
        self.assertEqual(sum(row["feature_usable"] == row["binary_label_valid"] == "True" for row in features), 1)

    def test_background_monitor_stops_without_taking_over_capture_writer(self):
        self.append(6)
        monitor = QualityMonitor(self.store.path)
        monitor.start()
        try:
            timeout = time.monotonic() + 5
            while not monitor.latest.reading and time.monotonic() < timeout:
                time.sleep(.02)
            self.assertEqual(monitor.latest.reading, "通过")
            self.append(.12)
        finally:
            monitor.stop()
            monitor.join(3)
        self.assertFalse(monitor.is_alive())
        self.assertFalse(self.store.closed)
