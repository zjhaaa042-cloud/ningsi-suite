"""联合评估、报告、预警、热力图、趋势与训练的规则测试。"""

from __future__ import annotations

import json
import unittest

from tests.helpers import temp_root
from ningsi.assessment.joint import BOUNDARY_NOTE, assess
from ningsi.assessment.report import SessionReport
from ningsi.monitoring import history
from ningsi.monitoring.alerts import AlertEngine
from ningsi.monitoring.heatmap import band_of, render_svg, render_text
from ningsi.scales.scoring import score_scale
from ningsi.signal.indicators import compute_indicators
from ningsi.signal.window import analyze_window
from ningsi.training.neurofeedback import NeurofeedbackSession, initial_target
from tests.helpers import elevated_answers, make_baseline, normal_answers

NORMAL_SART = {"go_accuracy": 0.99, "commission_rate": 0.0, "omission_rate": 0.0, "rt_mean": 0.3,
               "rt_variability": 0.08, "sequence_set_id": 1, "seed": 1, "spec": "joint-assessment-v1"}
POOR_SART = {"go_accuracy": 0.8, "commission_rate": 0.2, "omission_rate": 0.15, "rt_mean": 0.45,
             "rt_variability": 0.4, "sequence_set_id": 1, "seed": 1, "spec": "joint-assessment-v1"}
EEG_FINE = {"available": True, "valid_ratio": 0.9, "focus": 0.7, "relax": 0.6, "load": 0.3,
            "spec": "indicator-v1", "baseline_spec": "baseline-v1"}
EEG_STRESSED = {"available": True, "valid_ratio": 0.9, "focus": 0.25, "relax": 0.2, "load": 0.8,
                "spec": "indicator-v1", "baseline_spec": "baseline-v1"}


class JointAssessmentTest(unittest.TestCase):
    def test_consistent_stress_flags(self):
        result = assess(EEG_STRESSED, {"SAS": score_scale("SAS", elevated_answers("SAS"))}, {"sart": NORMAL_SART})
        self.assertIn("压力", result.conclusion)
        self.assertEqual(result.dimension_states["stress"]["state"], "一致提示偏离")

    def test_mismatch_asks_for_retest(self):
        result = assess(EEG_STRESSED, {"SAS": score_scale("SAS", normal_answers("SAS"))}, {"sart": POOR_SART})
        self.assertIn("不一致", result.conclusion)

    def test_low_quality_yields_insufficient_data(self):
        result = assess({"available": True, "valid_ratio": 0.3, "focus": 0.6, "relax": 0.6, "load": 0.4}, {}, {})
        self.assertIn("数据不足", result.conclusion)
        self.assertTrue(any("证据不足" in item for item in result.advice))

    def test_normal_case_and_boundary(self):
        result = assess(EEG_FINE, {"SAS": score_scale("SAS", normal_answers("SAS")), "SDS": score_scale("SDS", normal_answers("SDS"))}, {"sart": NORMAL_SART})
        self.assertIn("常模范围内", result.conclusion)
        self.assertEqual(result.boundary, BOUNDARY_NOTE)
        self.assertTrue(all(item.ref is not None for item in result.evidence))


class ReportTest(unittest.TestCase):
    def test_report_contains_evidence_backfill(self):
        assessment = assess(EEG_STRESSED, {"SAS": score_scale("SAS", elevated_answers("SAS"))}, {"sart": NORMAL_SART})
        report = SessionReport(
            "p01", "01", "001",
            {"summary": {"focus": {"mean": 0.3, "std": 0.1, "n": 5}}, "band_z": {"theta": 1.2}, "quality": {}},
            {"windows_total": 10, "windows_usable": 9, "valid_ratio": 0.9, "unusable_reasons": {"emg": 1}},
            {"SAS": score_scale("SAS", elevated_answers("SAS")).as_dict()},
            {"sart": NORMAL_SART},
            assessment,
        )
        markdown = report.to_markdown()
        for heading in ("一、状态评分", "二、采集质量", "三、量表结果", "四、行为任务", "五、联合评估结论", "六、改善建议", "七、证据回填"):
            self.assertIn(heading, markdown)
        self.assertNotIn("边界说明", markdown)
        self.assertNotIn(assessment.boundary, assessment.advice)
        self.assertNotIn(assessment.boundary, markdown)
        self.assertIn("sas_standard_score", markdown)
        self.assertEqual(json.loads(report.to_json())["participant"], "p01")

    def test_overlapping_windows_note_only_for_real_fast_sessions(self):
        """快速演示模式的真机会话要提醒"窗计数是高重叠样本"；仿真与真实节奏不提。"""
        base = (
            "p01", "01", "001",
            {"summary": {}, "band_z": {}, "quality": {}},
            {"windows_total": 35, "windows_usable": 0, "valid_ratio": 0.0,
             "unusable_reasons": {"eog": 35}},
            {}, {}, None,
        )
        real_fast = SessionReport(*base, extras={"source_kind": "lsl", "time_scale": 0.05}).to_markdown()
        self.assertIn("高重叠样本", real_fast)
        real_rhythm = SessionReport(*base, extras={"source_kind": "lsl", "time_scale": 1.0}).to_markdown()
        self.assertNotIn("高重叠样本", real_rhythm)
        sim_fast = SessionReport(*base, extras={"source_kind": "sim", "time_scale": 0.05}).to_markdown()
        self.assertNotIn("高重叠样本", sim_fast)
        no_extras = SessionReport(*base).to_markdown()
        self.assertNotIn("高重叠样本", no_extras)
        self.assertIn("排除原因", no_extras)


class AlertTest(unittest.TestCase):
    def setUp(self):
        self.sim, self.baseline = make_baseline()
        self.engine = AlertEngine()

    def test_low_focus_triggers_after_sustain(self):
        triggered = []
        for index in range(15):
            window = analyze_window(self.sim.window("drowsy"), self.sim.srate, t_end=100 + index * 2.0)
            result = compute_indicators(window, self.baseline)
            triggered += [event for event in self.engine.feed(window, result) if event.state == "triggered"]
        self.assertIn("low_focus", [event.kind for event in triggered])
        self.assertLessEqual(triggered[0].sustained_sec, 22.0)

    def test_artifact_window_pauses_timer(self):
        for index in range(6):
            window = analyze_window(self.sim.window("drowsy"), self.sim.srate, t_end=100 + index * 2.0)
            result = compute_indicators(window, self.baseline)
            self.engine.feed(window, result)
        artifact = analyze_window(self.sim.window("rest", artifact=True), self.sim.srate, t_end=200.0)
        self.assertEqual(self.engine.feed(artifact, None), [])
        self.assertEqual(self.engine._state["low_focus"].breached, 0.0)

    def test_release_condition(self):
        for index in range(15):
            window = analyze_window(self.sim.window("drowsy"), self.sim.srate, t_end=100 + index * 2.0)
            self.engine.feed(window, compute_indicators(window, self.baseline))
        self.assertIn("low_focus", self.engine.active_kinds())
        released = []
        for index in range(20):
            window = analyze_window(self.sim.window("focused"), self.sim.srate, t_end=300 + index * 2.0)
            released += [event for event in self.engine.feed(window, compute_indicators(window, self.baseline))
                         if event.kind == "low_focus" and event.state == "released"]
        self.assertTrue(released)


class HeatmapHistoryTest(unittest.TestCase):
    def test_band_mapping_includes_missing(self):
        self.assertEqual(band_of(0.05)["label"], "深蓝")
        self.assertEqual(band_of(0.95)["label"], "橙红")
        self.assertEqual(band_of(None)["label"], "低质量缺失")
        self.assertEqual(band_of(1.5)["index"], 4)

    def test_renderers(self):
        series = [(index * 2.0, 0.1 * (index % 10)) for index in range(30)]
        series[3] = (6.0, None)
        self.assertIn("■", render_text(series, columns=10))
        self.assertIn("<svg", render_svg(series))

    def test_history_aggregation_and_comparability(self):
        root = temp_root("history")
        for index, device in enumerate(("sim", "sim", "other")):
            history.append_record(root, {
                "participant": "p01", "device": device, "srate": 250.0, "channels": 1,
                "indicators": {"focus": 0.4 + 0.1 * index},
                "recorded_at": f"2026-09-{10 + index:02d}T09:00:00+00:00",
            })
        records = history.load_records(root)
        usable, rejected = history.comparable(records, "sim", 250.0, 1)
        self.assertEqual(len(usable), 2)
        self.assertEqual(len(rejected), 1)
        points = history.aggregate(usable, "focus", "week")
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["mean"], 0.45, places=4)


class TrainingTest(unittest.TestCase):
    def test_initial_target_comes_from_baseline_median(self):
        _, baseline = make_baseline()
        target, rationale = initial_target(baseline)
        from ningsi.signal.indicators import score_from_z
        expected = score_from_z(baseline.z_index("focus", baseline.index_stat("focus")["median"]))
        self.assertAlmostEqual(target, max(0.25, min(0.85, expected)), places=6)
        self.assertIn("基线中位数", rationale)

    def test_adaptation_raises_and_lowers_target(self):
        session = NeurofeedbackSession("p01", "01", "001", target=0.5)
        session.add_segment([(index * 2.0, 0.9) for index in range(10)], 120.0)
        self.assertGreater(session.target, 0.5)
        session.add_segment([(index * 2.0, 0.1) for index in range(10)], 120.0)
        self.assertLess(session.target, 0.55)

    def test_segment_stats(self):
        session = NeurofeedbackSession("p01", "01", "001", target=0.5)
        segment = session.add_segment([(0.0, 0.6), (2.0, 0.4), (4.0, 0.8)], 120.0)
        stats = segment.stats()
        self.assertAlmostEqual(stats["mean"], 0.6, places=4)
        self.assertAlmostEqual(stats["on_target_ratio"], 2 / 3, places=4)

    def test_continuous_hold_rejects_fragmented_and_missing_windows(self):
        from ningsi.training.neurofeedback import Segment
        fragmented = Segment(1, 120, 0.5, 6, [(i * 2, 0.9 if i % 2 == 0 else 0.1) for i in range(12)])
        self.assertFalse(fragmented.stats()["achieved"])
        uninterrupted = Segment(1, 120, 0.5, 6, [(0, 0.9), (2, 0.9), (4, 0.9), (6, 0.9)])
        self.assertTrue(uninterrupted.stats()["achieved"])
        gap = Segment(1, 120, 0.5, 6, [(0, 0.9), (2, 0.9), (6, 0.9), (8, 0.9)])
        self.assertEqual(gap.stats()["max_hold_sec"], 2)
        self.assertFalse(gap.stats()["achieved"])

    def test_summary_marks_comparability(self):
        _, baseline_before = make_baseline()
        _, baseline_after = make_baseline(seed=21)
        session = NeurofeedbackSession("p01", "01", "001", target=0.5)
        session.add_segment([(0.0, 0.5)], 120.0)
        summary = session.summary(baseline_before, baseline_after)
        self.assertTrue(summary["baseline_comparable"])
        self.assertEqual(summary["baseline_comparability"], "comparable")


if __name__ == "__main__":
    unittest.main()
