"""Deployment guards for both frontal EEG research candidates."""
import unittest
from types import SimpleNamespace

from ningsi.models.affect import load_models, predict_window


class PublicAffectTests(unittest.TestCase):
    def setUp(self):
        self.window = SimpleNamespace(channels=2, usable=True, srate=250,
                                      rel={"alpha": .25, "beta": .18, "theta": .2})

    def test_weights_and_declared_people_are_disjoint(self):
        for model in load_models().values():
            self.assertTrue(model["trained_people"])
            self.assertTrue(model["held_out_people"])
            self.assertFalse(set(model["trained_people"]) & set(model["held_out_people"]))
            self.assertGreater(model["holdout"]["trials_scored"], 0)

    def test_both_models_score_and_withhold_personal_classification(self):
        predictions = predict_window(self.window, ["Fp2", "Fp1"], 250)
        self.assertEqual(set(predictions), {"stress", "valence"})
        for row in predictions.values():
            self.assertTrue(row["available"])
            self.assertLessEqual(0, row["score"])
            self.assertLessEqual(row["score"], 1)
            self.assertFalse(row["display_class"])
            self.assertFalse(row["target_device_validated"])

    def test_wrong_labels_rate_and_failed_quality_never_produce_scores(self):
        for labels, rate, reason in ((["Ch1", "Ch2"], 250, "requires_fp1_fp2"),
                                     (["Fp1", "Fp2"], 200, "requires_250hz")):
            for item in predict_window(self.window, labels, rate).values():
                self.assertIsNone(item["score"])
                self.assertEqual(item["reason"], reason)
        self.window.usable = False
        self.assertFalse(predict_window(self.window, ["Fp1", "Fp2"], 250)["stress"]["available"])

    def test_corrupted_or_absent_relative_band_is_rejected(self):
        self.window.rel["theta"] = float("nan")
        self.assertEqual(predict_window(self.window, ["Fp1", "Fp2"], 250)["valence"]["reason"],
                         "invalid_band_features")
