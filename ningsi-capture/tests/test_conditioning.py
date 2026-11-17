"""Signal-level regressions for the acquisition / human feature contract."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .test_support import FakeClock
from ningsi_capture.acquisition import simulation_source, synthetic_samples
from ningsi_capture.export import export_session
from ningsi_capture.storage import SessionStore
from ningsi.signal import conditioning as dsp


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        self.rate = 250
        self.t = np.arange(1500) / self.rate
        self.eeg = synthetic_samples(np.arange(1500)).T

    def run_window(self, values, stamps=None, factor=1.):
        return dsp.condition(values, self.t if stamps is None else stamps, self.rate, factor, dsp.POLICY, traces=True)

    def test_drift_mains_and_passband_at_multiple_rates(self):
        for rate in (100, 128, 250, 512, 1000):
            for hz, low, high in [(0, 0, 1e-9), (.2, 0, .09), (4, .97, 1.02), (8, .97, 1.02),
                                  (13, .97, 1.02), (20, .97, 1.02), (30, .97, 1.02),
                                  (49.8, 0, .01), (50, 0, .01), (60.2, 0, .01)]:
                if hz >= rate / 2:
                    continue
                impulse = dsp._kernel(rate)
                level = abs(impulse @ np.exp(-2j * np.pi * hz * np.arange(len(impulse)) / rate))
                self.assertTrue(low <= level <= high, (rate, hz, level))

    def test_strong_mains_creates_no_edge_or_false_artifact(self):
        clean = self.run_window(self.eeg)
        self.assertTrue(clean["window"].usable)
        for phase in np.linspace(0, 2*np.pi, 9):
            noise = 850*np.sin(2*np.pi*50*self.t + phase) + 400*np.sin(2*np.pi*60*self.t - phase)
            row = self.run_window(self.eeg + 270000 + noise)
            self.assertTrue(row["window"].usable, row["window"].quality.reasons)
            np.testing.assert_allclose(row["filtered_uv"], clean["filtered_uv"], atol=.005)
            self.assertFalse(row["artifact_mask"].any())

    def test_overlap_and_no_future_input_or_gap_bridging(self):
        raw = synthetic_samples(np.arange(2000)).T
        a = self.run_window(raw[:, :1500])
        b = self.run_window(raw[:, 500:], stamps=self.t + 2)
        np.testing.assert_allclose(a["filtered_uv"][:, 500:], b["filtered_uv"][:, :500], atol=1e-12)
        future = raw[:, :1500].copy()
        future[:, -100:] += 1e6
        np.testing.assert_allclose(self.run_window(future)["filtered_uv"][:, :-100], a["filtered_uv"][:, :-100])
        for where, code in ((slice(200, None), "sample_gap_in_filter_history"), (slice(2, None), "timestamp_gap_over_100ms")):
            t = self.t.copy()
            t[where] += .02 if where.start == 200 else .25
            bad = self.run_window(self.eeg, t)
            self.assertIn(code, bad["window"].quality.reasons)
            self.assertIsNone(bad["filtered_uv"])
            self.assertFalse(bad["features"])
        self.assertFalse(self.run_window(self.eeg, factor=None)["features"])

    def test_local_artifacts_and_clipping_do_not_enter_features(self):
        blink = 160*np.exp(-.5*((self.t-3.5)/.105)**2)
        active = (self.t > 3.3) & (self.t < 3.9)
        emg = 45*np.sin(2*np.pi*36*self.t)*active
        high_emg = sum(18*np.sin(2*np.pi*f*self.t) for f in (68, 73, 81, 88, 93))*active
        step = 170*(self.t > 3.7)
        for artifact, flag in ((blink, dsp.EYE), (emg, dsp.MUSCLE), (high_emg, dsp.MUSCLE), (step, dsp.MOVEMENT)):
            raw = self.eeg.copy()
            raw[1] += artifact
            row = self.run_window(raw)
            self.assertTrue(np.any(row["artifact_mask"] & flag))
            self.assertFalse(row["features"])
            self.assertFalse(row["window"].usable)
        raw = self.eeg.copy()
        raw[0, 480:520] = 4095
        row = self.run_window(raw)
        self.assertTrue(np.any(row["artifact_mask"] & dsp.SIGNAL_FAULT))
        self.assertFalse(row["features"])

    def test_mask_and_raw_roundtrip_and_clean_preview(self):
        from ningsi_capture.live_quality import CaptureQuality
        from ningsi_capture.storage import open_readonly
        from contextlib import closing
        with TemporaryDirectory() as temp:
            clock = FakeClock()
            store = SessionStore(temp, participant="MASK_FIXTURE", source=simulation_source(), protocol={}, clock=clock)
            noisy = self.eeg.copy()
            noisy[0] += 160*np.exp(-.5*((self.t-3.5)/.105)**2)
            try:
                store.append(noisy.T, 1000+self.t, offset=0.)
                with closing(open_readonly(store.path)) as db:
                    snapshot = CaptureQuality().update(db, store.metadata)
            finally:
                store.close("aborted")
            out = Path(temp) / "export"
            summary = export_session(store.path, out)
            clean = np.load(out / "cleaned_windows_uv.npy")
            masks = np.load(out / "artifact_mask.npy")
            np.testing.assert_equal(np.load(out / "raw_samples.npy"), noisy.T)
            self.assertTrue(np.all(masks[0] == dsp.UNAVAILABLE))
            self.assertTrue(np.all(np.isnan(clean[0])))
            self.assertTrue(np.any(masks[1] & dsp.EYE))
            self.assertTrue(np.any(masks[1] == 0))
            self.assertEqual(summary["windows_usable"], 0)
            np.testing.assert_equal(np.isnan(clean[1, 0]), masks[1] != 0)
            np.testing.assert_equal(clean[1].T, np.array(snapshot.cleaned_samples))
