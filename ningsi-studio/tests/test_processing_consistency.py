"""Replay identical windows through the CLI and Studio application adapters."""

import unittest

import numpy as np

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.app.pipeline import SessionConfig, SessionRunner
from ningsi.signal.baseline import build_baseline
from ningsi.signal.indicators import summarize_indicators
from ningsi.signal.window import analyze_window
from ningsi_studio.domain.indicators import WindowEngine


class ReplaySource:
    def __init__(self, rows):
        self.rows = iter(rows)

    def analysis_window(self, state="rest", *, artifact=False):
        return next(self.rows)


class ProcessingConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = SyntheticEEG(seed=42)
        cls.baseline = build_baseline(
            [analyze_window(source.window(), 250, t_end=4 + i * 2) for i in range(20)],
            device="test", srate=250)
        assert cls.baseline.valid

    def replay(self, rows, baseline):
        runner = SessionRunner(SessionConfig(device="test", task_plan=(("rest", len(rows)),)),
                               eeg=ReplaySource(rows))
        cli = runner.run_task(baseline)
        engine = WindowEngine(ReplaySource(rows), 250, device="test", baseline=baseline)
        engine.t = 100.0  # CLI task's existing session time origin.
        steps = engine.run("rest", len(rows))
        self.assertEqual([w.as_dict() for w in cli["windows"]], [s.window.as_dict() for s in steps])
        self.assertEqual(cli["series"], engine.series(steps))
        studio_quality = engine.summarize(steps)
        # Keep each adapter's existing optional empty aggregate field compatible.
        self.assertEqual({**cli["quality"], "indicators": cli["quality"].get("indicators", {})},
                         {**studio_quality, "indicators": studio_quality.get("indicators", {})})
        self.assertEqual(cli["indicators"], summarize_indicators([s.indicators for s in steps]))
        self.assertEqual(cli["alerts"], engine.fired_alerts)
        return cli, steps

    def test_scores_and_unavailable_windows_agree_across_adapters(self):
        source = SyntheticEEG(seed=9)
        rows = [(source.window("focused"), {}), (None, {}),
                (np.zeros((1, 0)), {}), (np.ones((1, 500)), {}),
                (np.full((1, 1000), np.nan), {}),
                (source.window("drowsy"), {"reasons": ["motion_reference_excursion"]}),
                (source.window("focused"), {})]
        cli, steps = self.replay(rows, self.baseline)
        self.assertEqual([s.usable for s in steps], [True, False, False, False, False, False, True])
        self.assertTrue(all(value is None for _, value in cli["series"][1:6]))
        self.assertEqual(steps[2].payload()["quality"]["metrics"]["samples"], 0)

    def test_invalid_baseline_keeps_clean_quality_and_no_attention_scores(self):
        cli, steps = self.replay([(SyntheticEEG(seed=9).window(), {})], None)
        self.assertTrue(steps[0].usable)
        self.assertEqual(steps[0].payload()["indicator_reason"], "baseline_invalid")
        self.assertEqual(cli["series"], [(102.0, None)])

    def test_artifact_windows_cannot_accumulate_low_attention_warning_time(self):
        raw = SyntheticEEG(seed=9).window("drowsy")
        rejected = (raw, {"reasons": ["motion_reference_excursion"]})
        cli, _ = self.replay([rejected] * 20, self.baseline)
        self.assertEqual({event["kind"] for event in cli["alerts"]}, {"poor_signal"})
        self.assertTrue(all(value is None for _, value in cli["series"]))


if __name__ == "__main__":
    unittest.main()
