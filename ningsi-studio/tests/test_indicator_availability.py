import unittest
import numpy as np
from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.signal.baseline import index_values
from ningsi_studio.domain.indicators import WindowEngine


class IndicatorAvailabilityTests(unittest.TestCase):
    def test_good_signal_requires_baseline_and_recovers_after_calibration(self):
        engine = WindowEngine(SyntheticEEG(seed=42), 250, device="sim-bsense")
        first = engine.step("rest").payload()
        self.assertTrue(first["usable"])
        self.assertFalse(first["indicator_available"])
        self.assertEqual(first["indicator_reason"], "baseline_invalid")
        self.assertIsNone(first["baseline_valid"])
        self.assertEqual(first["focus_basis"], {})
        self.assertTrue(first["relative_band_powers"])
        baseline, _ = engine.build_baseline("rest", 30)
        self.assertTrue(baseline.valid)
        step = engine.step("focused")
        result = step.payload()
        self.assertTrue(result["indicator_available"])
        self.assertTrue(result["baseline_valid"])
        self.assertEqual(result["focus_basis"], {
            "window_log_ratio": index_values(step.window)["focus"],
            "baseline_log_ratio": baseline.index_stat("focus")["mean"],
        })
        self.assertEqual(set(result["scores"]), {"focus", "relax", "load"})

    def test_nonfinite_and_missing_samples_do_not_generate_scores(self):
        class Source:
            def window(self, *args, **kwargs):
                return self.data
        source = Source()
        engine = WindowEngine(source, 250, device="test")
        for raw, reason in [(np.full((1, 1000), np.nan), "non_finite_signal"),
                            (np.zeros((1, 0)), "no_data"),
                            (np.zeros((1, 10)), "signal_too_short")]:
            source.data = raw
            result = engine.step().payload()
            self.assertFalse(result["indicator_available"])
            self.assertIn(reason, result["reasons"])
            import json
            json.dumps(engine.step().payload(include_signal=True), allow_nan=False)
