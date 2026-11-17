"""Training performance and acquisition quality must remain distinct in exports."""
import unittest
import re
from html import unescape

from ningsi.assessment.report import SessionReport


def report(stats, **patch):
    segment = {"index": 1, "duration_sec": 10, "stats": stats,
               "effective_time": {"valid_seconds": 2, "target_seconds": 50,
                   "elapsed_seconds": 10, "max_seconds": 100,
                   "completion_reason": "wall_time_limit"}}
    metadata = {"segments": [segment], "elapsed_sec": 10,
                "max_acquisition_seconds": 1040, "acquisition_complete": False, **patch}
    return SessionReport("fixture", "01", "001", extras={
        "protocol": "effective_time", "effective_time": metadata})


class EffectiveTimeReportTests(unittest.TestCase):
    def test_good_performance_does_not_hide_low_coverage_or_timeout(self):
        source = report({"timing_spec": "neurofeedback-v3", "valid_seconds": 2,
            "valid_time_ratio": .2, "on_target_valid_ratio": 1., "on_target_ratio": .2})
        text = source.to_markdown()
        self.assertIn("## 有效时长反馈训练", text)
        self.assertIn("有效时间内达标100.0%；有效覆盖20.0%；整段达标20.0%", text)
        self.assertIn("有效反馈2秒／训练时长10秒（协议秒）。实际时长10秒", text)
        self.assertIn("尚未全部完成", text)
        self.assertIn("达到补采上限", text)
        self.assertIn("本协议未采集独立监测", text)
        self.assertNotIn("十分钟分级反馈", text)
        self.assertIsNone(source.as_dict()["extras"]["effective_time"].get("changed"))

    def test_no_data_is_not_zero_performance(self):
        text = report({"timing_spec": "neurofeedback-v3", "valid_seconds": 0,
            "valid_time_ratio": 0, "on_target_valid_ratio": None, "on_target_ratio": None}).to_markdown()
        self.assertIn("无有效脑电反馈", text)
        self.assertIn("不计算达标率", text)
        self.assertNotIn("有效时间内达标0.0%", text)

    def test_missing_or_nonfinite_ratios_do_not_turn_into_scores(self):
        for value in (None, True, float("nan"), float("inf"), -1, 2):
            with self.subTest(value=value):
                text = report({"timing_spec": "neurofeedback-v3", "valid_seconds": 2,
                    "valid_time_ratio": .2, "on_target_valid_ratio": value}).to_markdown()
                self.assertIn("有效时间内达标—", text)

    def test_fixed_ten_minute_record_keeps_its_protocol_identity(self):
        source = report({"timing_spec": "neurofeedback-v3", "valid_seconds": 2,
            "valid_time_ratio": .2, "on_target_valid_ratio": 1., "on_target_ratio": .2})
        source.extras = {"protocol": "ten_min", "ten_min": source.extras["effective_time"]}
        text = source.to_markdown()
        self.assertIn("## 十分钟分级反馈", text)
        self.assertNotIn("## 有效时长反馈训练", text)
        self.assertNotIn("PVT刺激、反应时", text)

    def test_offline_html_distinguishes_performance_quality_and_time(self):
        from ningsi.assessment.diagnostics import render_html
        source = report({"timing_spec": "neurofeedback-v3", "valid_seconds": 2,
            "valid_time_ratio": .2, "on_target_valid_ratio": 1., "on_target_ratio": .2},
            note='<script>alert("bad")</script>')
        text = render_html(source.as_dict())
        self.assertIn("有效时间内达标 100.0%", text)
        self.assertIn("有效覆盖 20.0%", text)
        self.assertIn("实际时长 10秒", text)
        self.assertIn("达到补采上限", text)
        self.assertIn("尚未全部完成", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn("<script>", text)

    def test_sufficient_seconds_do_not_hide_an_invalid_baseline(self):
        source = report({}, phase_results={"baseline_open": {
            "target_seconds": 60, "valid_seconds": 60, "elapsed_seconds": 180,
            "max_seconds": 180, "baseline_valid": False, "completion_reason": "max_duration"}})
        text = source.to_markdown()
        self.assertIn("睁眼基线：有效时间60秒／目标60秒", text)
        self.assertIn("基线质量条件未通过", text)
        self.assertIn("尚未全部完成", text)

    def test_accelerated_simulation_exports_keep_protocol_and_actual_clocks_distinct(self):
        from ningsi.assessment.diagnostics import render_html
        source = report({"timing_spec": "neurofeedback-v3", "valid_seconds": 50,
            "valid_time_ratio": .5, "on_target_valid_ratio": 1., "on_target_ratio": .5},
            time_scale=.02, elapsed_sec=12.5, actual_elapsed_seconds=12.5,
            protocol_elapsed_seconds=625, phase_results={"baseline_open": {
                "valid_seconds": 60, "target_seconds": 60, "elapsed_seconds": 100,
                "max_seconds": 180, "actual_elapsed_seconds": 2,
                "time_scale": .02, "completion_reason": "target_reached", "baseline_valid": True}})
        segment = source.extras["effective_time"]["segments"][0]
        segment["duration_sec"] = 100
        segment["effective_time"].update(valid_seconds=50, elapsed_seconds=100,
            actual_elapsed_seconds=2, time_scale=.02, completion_reason="target_reached")
        outputs = [source.to_markdown(), unescape(re.sub(r"<[^>]*>", "", render_html(source.as_dict())))]
        for text in outputs:
            with self.subTest(renderer="html" if text is outputs[1] else "markdown"):
                actual = float(re.search(r"实际时长\s*([\d.]+)秒", text).group(1))
                protocol = float(re.search(r"训练时长\s*([\d.]+)秒（协议秒）", text).group(1))
                self.assertEqual(actual, 2)
                self.assertEqual(protocol, 100)
                self.assertAlmostEqual(actual/protocol, .02)
                total_actual = float(re.search(r"实际采集用时\s*([\d.]+)秒", text).group(1))
                total_protocol = float(re.search(r"协议采集用时\s*([\d.]+)秒", text).group(1))
                self.assertEqual(total_actual, 12.5)
                self.assertEqual(total_protocol, 625)
                self.assertAlmostEqual(total_actual/total_protocol, .02)
                phase = re.search(r"睁眼基线：.*?协议用时([\d.]+)秒.*?实际用时([\d.]+)秒", text)
                self.assertEqual(tuple(map(float, phase.groups())), (100, 2))
                self.assertIn("时间倍率0.02", text)
        # Export formatting does not rewrite the stored scoring denominator.
        self.assertEqual(segment["duration_sec"], 100)
        self.assertEqual(segment["stats"]["valid_time_ratio"], .5)

    def test_legacy_scaled_record_derives_actual_time_when_explicit_field_missing(self):
        from ningsi.assessment.report import _effective_actual_elapsed
        metadata = {"time_scale": .02}
        self.assertEqual(_effective_actual_elapsed({"elapsed_seconds": 100}, metadata), 2)
        # Prefer the explicit current field rather than recalculating its value.
        self.assertEqual(_effective_actual_elapsed({"elapsed_seconds": 100,
            "actual_elapsed_seconds": 2.01}, metadata), 2.01)

    def test_inconsistent_timing_has_an_explicit_failure_reason_in_both_exports(self):
        from ningsi.assessment.diagnostics import render_html
        source = report({})
        source.extras["effective_time"]["segments"][0]["effective_time"].update(
            completion_reason="timing_issue", completed=False)
        for text in (source.to_markdown(), render_html(source.as_dict())):
            self.assertIn("计时不一致，未记为完成", text)
            self.assertNotIn("结束原因 详见阶段采集记录", text)
