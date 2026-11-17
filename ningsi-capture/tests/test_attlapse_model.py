"""External pretraining must use the same causal grid and montage as blind scoring."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from . import test_support
from ningsi.models import attlapse
from ningsi.models.logistic import FEATURE_NAMES
from ningsi.models.research_features import aggregate_probe

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_attlapse.py"
SPEC = importlib.util.spec_from_file_location("train_attlapse", SCRIPT)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


class AttlapseModelTests(unittest.TestCase):
    def test_probe_excerpt_reproduces_recording_grid(self):
        rate = 512
        t = np.arange(rate * 35) / rate
        pair = np.asarray([11 * np.sin(2 * np.pi * f * t) for f in (9, 11)])
        onset = 29.3
        windows = []
        for end in range(4 * rate, len(t) + 1, 2 * rate):
            start = max(0, end - 6 * rate)
            windows.append(attlapse.bipolar_window(pair[:, start:end], t[start:end], rate, 1., (0, 1)))
        expected = aggregate_probe(windows, onset, baseline_valid=True, phase_start=0.)
        start, stop = int((onset - 16) * rate), int(np.ceil((onset - 2) * rate))
        probe = dict(rate=rate, onset_sec=onset, timestamps=t[start:stop], values_uv=pair[:, start:stop],
                     phase_start_sec=0., source_answer="on_task", participant="example", probe_id="p1", label=0)
        row = trainer.extract_probe(probe)
        self.assertFalse(row["reasons"])
        np.testing.assert_allclose([row[k] for k in FEATURE_NAMES], [expected["features"][k] for k in FEATURE_NAMES])
        probe["phase_start_sec"] = onset - 11
        self.assertIn("lookback_crosses_phase", trainer.extract_probe(probe)["reasons"])

    def test_electrodes_and_bad_input_signal(self):
        self.assertEqual(attlapse.channel_indices(["FP2", "Cz", "fp1"]), (2, 0))
        for labels in (["Ch1", "Ch2"], ["Fp1", "Fp1", "Fp2"]):
            with self.assertRaises(ValueError):
                attlapse.channel_indices(labels)
        t = np.arange(1000) / 250.
        pair = np.stack([10 * np.sin(2 * np.pi * 10 * t), np.zeros_like(t)])
        self.assertIn("flat_channel", attlapse.bipolar_window(pair, t, 250., 1., (0, 1))["window"].quality.reasons)

    def test_source_eye_correction_uses_only_preprobe_samples(self):
        t = np.arange(512 * 20) / 512.
        rng = np.random.default_rng(8)
        aux = rng.normal(size=(4, len(t)))
        probe = dict(aux_channels=list(trainer.EOG_CHANNELS), bad_aux_channels=[], aux_values=aux,
                     aux_units=["µV"] * 4, onset_sec=18., rate=512)
        x = np.stack([aux[0] - aux[1], aux[2] - aux[3]])
        raw = x + 0.1 * rng.normal(size=x.shape)
        cleaned, reason = trainer.remove_source_eog(probe, raw, t)
        self.assertIsNone(reason)
        before = t <= 16.
        self.assertLess(np.std(cleaned[:, before]), np.std(raw[:, before]) / 5)
        raw[:, ~before] = -500
        aux[:, ~before] = np.nan
        repeated, _ = trainer.remove_source_eog(probe, raw, t)
        np.testing.assert_array_equal(repeated[:, before], cleaned[:, before])
        probe["bad_aux_channels"] = ["eog_up"]
        self.assertIsNone(trainer.remove_source_eog(probe, raw, t)[0])

    def test_frozen_model_roundtrip_and_participant_splits(self):
        rng = np.random.default_rng(14)
        rows = [{"participant": f"fixture_{p}", "probe_id": f"{p}_{j}", "label": j % 2, "reasons": "",
                 **dict(zip(FEATURE_NAMES, rng.normal(size=len(FEATURE_NAMES))))}
                for p in range(7) for j in range(6)]
        model, report = trainer.fit_candidate(rows)
        for fold in report["folds"]:
            self.assertTrue(set(fold["validation_participants"]).isdisjoint(fold["train_participants"]))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            self.assertIsNone(attlapse.load_candidate(path))
            path.write_text(json.dumps(model), encoding="utf-8")
            score = attlapse.score_probe(attlapse.load_candidate(path), rows[0])
            self.assertTrue(0 <= score <= 1)
            model["parameters"]["bias"] += 1
            path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaises(ValueError):
                attlapse.load_candidate(path)
