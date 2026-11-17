"""Workbench analysis performs real filtering without asserting human validity."""

import copy
import json
import unittest

import numpy as np

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi_studio.domain.showcase import build_analysis, metadata


class ShowcaseAnalysisTests(unittest.TestCase):
    def test_clean_demo_has_independent_simulated_baseline_and_valid_windows(self):
        result = build_analysis({"mode": "demo", "artifact": "clean"})
        self.assertEqual(result["source"]["kind"], "sim")
        self.assertTrue(result["source"]["simulated"])
        self.assertFalse(result["source"]["real_hardware_verified"])
        self.assertTrue(result["baseline"]["valid"])
        self.assertGreater(result["quality"]["windows_usable"], 0)
        self.assertTrue(result["latest"]["scores"])
        self.assertEqual({row["name"] for row in result["bands"]}, {"theta", "alpha", "beta"})
        self.assertEqual(set(result["latest"]["scores"]), {"focus", "relax", "load"})
        self.assertEqual(result["traces"]["channels"][0]["label"], "Ch1")
        self.assertLess(result["spectrum"]["filtered_line_fraction"]["50"],
                        result["spectrum"]["raw_line_fraction"]["50"] * .01)
        json.dumps(result, allow_nan=False)

    def test_all_injected_transient_presets_reject_the_latest_window(self):
        for artifact in ("blink", "muscle", "movement", "mixed"):
            with self.subTest(artifact=artifact):
                result = build_analysis({"mode": "demo", "artifact": artifact})
                self.assertFalse(result["latest"]["usable"])
                self.assertTrue(result["latest"]["reasons"])
                self.assertEqual(result["latest"]["scores"], {})
                self.assertEqual(result["latest"]["features"], {})
                self.assertTrue(result["traces"]["masked_intervals"])
                self.assertIn(None, result["traces"]["channels"][0]["cleaned_uv"])
                json.dumps(result, allow_nan=False)

    def raw_input(self, seconds=12):
        return {"mode": "raw", "rate": 250, "scale_to_uv": 1.,
                "samples": SyntheticEEG(seed=42, noise_uv=1.5).window("rest", seconds=seconds).tolist()}

    def test_imports_are_not_claimed_as_hardware_and_task_never_becomes_its_baseline(self):
        payload = self.raw_input()
        before = copy.deepcopy(payload)
        result = build_analysis(payload)
        self.assertEqual(payload, before)
        self.assertEqual(result["source"]["kind"], "imported")
        self.assertFalse(result["source"]["clock_verified"])
        self.assertIn("not_device_clock", result["source"]["timestamp_basis"])
        self.assertFalse(result["baseline"]["valid"])
        self.assertGreater(result["quality"]["windows_usable"], 0)
        self.assertTrue(all(not row["scores"] for row in result["windows"]))
        self.assertEqual(result["latest"]["indicator_reason"], "baseline_invalid")

    def test_separate_adequate_baseline_enables_relative_indices_without_changing_input(self):
        payload = self.raw_input()
        payload["baseline_samples"] = SyntheticEEG(seed=23, noise_uv=1.5).window("rest", seconds=36).tolist()
        payload["channel_labels"] = ["Fp1"]
        before = copy.deepcopy(payload)
        result = build_analysis(payload)
        self.assertTrue(result["baseline"]["valid"])
        self.assertEqual(result["baseline"]["source"], "independent_input")
        self.assertTrue(result["latest"]["scores"])
        self.assertEqual(payload, before)

    def test_short_or_polluted_baseline_cannot_enable_scoring(self):
        for baseline in (self.raw_input(6)["samples"], np.zeros((1, 36 * 250)).tolist()):
            with self.subTest(width=len(baseline[0])):
                payload = self.raw_input()
                payload["baseline_samples"] = baseline
                result = build_analysis(payload)
                self.assertFalse(result["baseline"]["valid"])
                self.assertEqual(result["latest"]["scores"], {})

    def test_clock_faults_are_visible_instead_of_repaired(self):
        payload = self.raw_input(6)
        stamps = (np.arange(1500) / 250).tolist()
        stamps[900] = stamps[899]
        payload["timestamps"] = stamps
        before = list(stamps)
        result = build_analysis(payload)
        self.assertIn("nonmonotonic_timestamps", result["latest"]["reasons"])
        self.assertEqual(result["quality"]["windows_usable"], 0)
        self.assertTrue(all(v is None for v in result["traces"]["channels"][0]["filtered_uv"]))
        self.assertEqual(stamps, before)
        json.dumps(result, allow_nan=False)

    def test_units_scale_before_filtering_and_spectra(self):
        payload = self.raw_input(6)
        result = build_analysis(payload)
        payload["samples"] = (np.asarray(payload["samples"]) / 1e6).tolist()
        payload["scale_to_uv"] = 1e6
        volts = build_analysis(payload)
        np.testing.assert_allclose(result["traces"]["channels"][0]["filtered_uv"],
                                   volts["traces"]["channels"][0]["filtered_uv"], atol=1e-6)

    def test_overlap_coverage_uses_the_union_and_display_is_bounded(self):
        result = build_analysis({"mode": "demo", "artifact": "clean", "rate": 1000, "channels": 1})
        self.assertLessEqual(len(result["traces"]["time_sec"]), 1000)
        self.assertEqual(result["quality"]["usable_seconds"], 16.)
        self.assertLess(result["quality"]["usable_seconds"], result["quality"]["windows_usable"] * 4)

    def test_bounded_strict_json_inputs(self):
        valid = self.raw_input(6)
        cases = [{}, {"mode": "raw"}, {"mode": "demo", "artifact": "unknown"},
                 {"mode": "demo", "rate": True}, {"mode": "demo", "channels": 1.5},
                 {"mode": "demo", "rate": 2500}, {**valid, "scale_to_uv": None},
                 {**valid, "samples": [valid["samples"][0], [1.]]},
                 {**valid, "samples": [[float("nan")] * 1500]},
                 {**valid, "samples": [[True] * 1500]},
                 {**valid, "samples": [[1.] * 1499]},
                 {**valid, "channel_labels": []},
                 {**valid, "timestamps": [0.] * 1499}]
        for payload in cases:
            with self.subTest(keys=list(payload)):
                with self.assertRaises(ValueError):
                    build_analysis(payload)

    def test_metadata_does_not_promise_ground_truth_or_hardware_validation(self):
        result = metadata()
        self.assertEqual(result["processing"]["policy"], "causal-eeg-artifact-mask-v1")
        self.assertGreaterEqual(len(result["limitations"]), 5)
        self.assertEqual(result["baseline"]["minimum_usable_sec"], 30)
        json.dumps(result, allow_nan=False)

    def test_huge_legal_json_integers_are_input_errors_instead_of_overflow_errors(self):
        enormous = 10 ** 500
        valid = self.raw_input(6)
        samples = copy.deepcopy(valid["samples"])
        samples[0][0] = enormous
        timestamps = (np.arange(1500) / 250).tolist()
        timestamps[0] = enormous
        for payload in ({"mode": "demo", "rate": enormous},
                        {"mode": "demo", "channels": enormous},
                        {**valid, "scale_to_uv": enormous},
                        {**valid, "samples": samples},
                        {**valid, "timestamps": timestamps}):
            with self.subTest(keys=list(payload)):
                # JSON itself accepts integers larger than an IEEE double.
                decoded = json.loads(json.dumps(payload))
                with self.assertRaises(ValueError):
                    build_analysis(decoded)


if __name__ == "__main__":
    unittest.main()
