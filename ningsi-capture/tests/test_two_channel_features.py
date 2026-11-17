import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from attlapse_two_channel_features import FEATURE_SETS, extract_features
from train_attlapse import extract_probe


def probe_fixture(rate=512):
    start = 50000
    indices = np.arange(start, start + 46 * rate)
    times = indices / rate
    onset = (indices[-1] + 1) / rate
    channels = np.array([12 * np.sin(2 * np.pi * 7 * times + delay)
                         + 9 * np.sin(2 * np.pi * 10 * times - .5 * delay)
                         + 12 * np.sin(2 * np.pi * 16.5 * times + 2 * delay) for delay in (0, 1)])
    return dict(values_uv=channels, timestamps=times, channels=["Fp1", "Fp2"], unit="µV", source="sourcedata",
                source_score=57, source_answer="spontaneous_mw", label=1, bad_channels=[], participant="fixture",
                probe_id="test-1", rate=rate, phase_start_sec=times[0], start_sample=start,
                onset_sample=int(indices[-1] + 1), onset_sec=onset)


class FeatureTests(unittest.TestCase):
    def test_short_strict_spectral_features_reproduce_production(self):
        probe = probe_fixture()
        expected = extract_probe(probe)
        results = extract_features(probe)
        self.assertEqual(len(results), 9)
        result = next(row for row in results if (row["window"], row["quality"]) == ("pre12_2", "strict"))
        self.assertEqual(result["reasons"], "")
        self.assertEqual(result["windows_usable"], expected["windows_usable"])
        self.assertEqual(result["usable_seconds"], expected["usable_seconds"])
        for name in FEATURE_SETS["bipolar6"]:
            self.assertAlmostEqual(result["features"][name], expected[name], places=9)
        self.assertEqual([len(names) for names in FEATURE_SETS.values()], [6, 15, 18])
        self.assertEqual(set(result["features"]), set(FEATURE_SETS["bilateral_variability18"]))
        # All representations share the same accepted window set.
        self.assertTrue(all(end % (2 * probe["rate"]) == 0 for r in results for end in r["accepted_window_end_samples"]))

    def test_scores_and_recent_eeg_cannot_influence_features(self):
        probe = probe_fixture(250)
        probe["onset_sec"] -= .001  # Source probe onset between downsampled sample times.
        before = extract_features(probe)
        probe["values_uv"][:, probe["timestamps"] > probe["onset_sec"] - 2] = 1e7
        probe.update(label=0, source_answer="on_task", source_score=41)
        after = extract_features(probe)
        self.assertEqual([row["features"] for row in before], [row["features"] for row in after])
        self.assertTrue(all(row["window_end_sec"] == probe["onset_sec"] - 2 for row in after))

    def test_quality_sensitivity_retains_both_electrodes_physical_checks(self):
        probe = probe_fixture()
        probe["values_uv"] += 24 * np.sin(2 * np.pi * 2 * probe["timestamps"])
        gate = {row["quality"]: row for row in extract_features(probe) if row["window"] == "pre12_2"}
        self.assertIn("insufficient", gate["strict"]["reasons"])
        self.assertIn("input_eog", gate["strict"]["window_rejections"])
        self.assertEqual(gate["bipolar_strict"]["reasons"], "")
        self.assertEqual(gate["physical"]["reasons"], "")
        probe["values_uv"][0] = 0
        self.assertTrue(all("input_flat_channel" in row["window_rejections"] and not row["features"]
                            for row in extract_features(probe)))
        probe["bad_channels"] = ["Fp2"]
        self.assertTrue(all(row["reasons"] == "source_channel_marked_bad:Fp2" for row in extract_features(probe)))


if __name__ == "__main__":
    unittest.main()
