from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from .test_support import FakeClock
from ningsi.acquisition.timing import BIOMULTI_REC_TIMING
from ningsi_capture import biomulti
from ningsi_capture.acquisition import synthetic_samples
from ningsi_capture.export import export_session, recent_window_quality
from ningsi_capture.protocol import ProtocolConfig
from ningsi_capture.storage import SessionStore, open_readonly, read_metadata


class NativeRecTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.vendor = self.root / "vendor.db"
        self.source = {"kind": "lsl", "type": "EEG", "data_origin": "real", "name": "BioMulti Lite EEG-fixture",
                       "srate": 250, "channels": 2, "channel_labels": ["Ch1", "Ch2"],
                       "timestamp_policy": BIOMULTI_REC_TIMING, "source_unit": "unknown", "scale_to_uv": None,
                       "reference": "unknown", "native_rec": {"record_id": 7, "created_ms": 1000}}
        # The second channel plus small seeded noise makes every full packet unique.
        n = 12000
        points = np.arange(n)
        self.values = (synthetic_samples(points) + np.random.default_rng(7).normal(0, .03, (n, 2))).astype("<f4")
        batch_delay = (.045 + .035 * np.sin((points // 10) * 1.6) + .18 * ((points // 10) % 23 == 2))
        self.lsl = 1000 + points / 250 + batch_delay
        store = SessionStore(self.root, participant="REC_FIXTURE", source=self.source, protocol={}, clock=FakeClock())
        self.session = store.path
        store.append(self.values, self.lsl, offset=0.)
        store.close("aborted")
        self.frames = [{"pkg_type": 1, "pkgnum": 10000 + start // 10, "time_mark": 5000 + start * 4,
                        "eeg_channel": 2, "eeg_data_num": 10, "isLosspkg": False,
                        "brain_elec_channel": self.values[start:start + 10, ::-1].T.tolist()}
                       for start in range(0, n, 10)]
        with closing(sqlite3.connect(self.vendor)) as db, db:
            db.execute("CREATE TABLE record(id,recoredCreateTime,recoredEndTime,configText)")
            db.execute("CREATE TABLE source(id,recordId,data)")
            db.execute("INSERT INTO record VALUES (7,1000,49000,?)", (json.dumps({"eeg_channel_count": 2, "eeg_sample_rate": 250}),))
        self.save_frames(self.frames)

    def save_frames(self, frames):
        with closing(sqlite3.connect(self.vendor)) as db, db:
            db.execute("DELETE FROM source")
            db.execute("INSERT INTO source VALUES(1,7,?)", (json.dumps(frames),))

    def align(self):
        with closing(open_readonly(self.session)) as db:
            return biomulti.align(db, self.source, self.vendor, 7, 1000)

    def test_real_format_preserves_gaps_flags_and_ambiguities(self):
        self.frames[810]["isLosspkg"] = True
        self.save_frames(self.frames[:600] + self.frames[602:] + [self.frames[850]])
        times, audit, packets = self.align()
        self.assertEqual(len(list(packets)), len(self.frames) - 4)
        self.assertEqual(audit["matched_samples"], len(times) - 40)
        self.assertTrue(np.isnan(times[6000:6020]).all())
        self.assertTrue(np.isnan(times[8100:8110]).all())
        self.assertTrue(np.isnan(times[8500:8510]).all())
        self.assertTrue(np.all(np.diff(times[np.isfinite(times)]) > 0))
        self.assertLess(abs(audit["sample_rate_hz"] - 250), 1.)

    def test_outlet_jitter_fixed_only_in_derived_clock_and_units_must_be_confirmed(self):
        self.assertEqual(recent_window_quality(self.session), ("unit_unconfirmed", "native_clock_pending"))
        output = self.root / "uncalibrated"
        info = export_session(self.session, output, native_db=self.vendor)
        self.assertEqual(info["native_alignment"]["status"], "matched")
        self.assertEqual(info["windows_usable"], 0)
        np.testing.assert_array_equal(np.load(output / "timestamps.npy")[:, 0], self.lsl)
        np.testing.assert_array_equal(np.load(output / "raw_samples.npy"), self.values)
        self.assertTrue((np.diff(np.load(output / "device_timestamps.npy")) > 0).all())
        info = export_session(self.session, self.root / "confirmed", native_db=self.vendor,
                              calibration={"unit": "uV", "evidence": "software fixture in microvolts"})
        self.assertGreater(info["windows_usable"], 0)
        self.assertEqual((info["train_eligible"], info["test_eligible"]), (0, 0))
        meta = json.loads((self.root / "confirmed/manifest.json").read_text(encoding="utf8"))
        self.assertIsNone(meta["source_at_capture"]["scale_to_uv"])
        self.assertEqual(meta["source"]["unit_confirmation_stage"], "at_export")
        with closing(open_readonly(self.session)) as db:
            self.assertIsNone(read_metadata(db)["source"]["scale_to_uv"])

    def test_both_writers_skipping_a_packet_invalidates_the_boundary(self):
        for frame in self.frames[500:]:
            frame["pkgnum"] += 1
            frame["time_mark"] += 40
        self.save_frames(self.frames)
        times, audit, _ = self.align()
        self.assertTrue(np.isnan(times[5000]))
        self.assertEqual(audit["sequence_breaks"], 1)

    def test_wrong_record_and_unmatched_source_export_raw_without_qualifying(self):
        with self.assertRaisesRegex(ValueError, "编号"):
            export_session(self.session, self.root / "wrong-id", native_db=self.vendor, native_record=99)
        for f in self.frames:
            f["brain_elec_channel"].reverse()
        self.save_frames(self.frames)
        info = export_session(self.session, self.root / "wrong-samples", native_db=self.vendor,
                              calibration={"unit": "uV", "evidence": "fixture"})
        self.assertEqual(info["native_alignment"]["status"], "unavailable")
        self.assertEqual(info["windows_usable"], 0)
        self.assertFalse((self.root / "wrong-samples/device_timestamps.npy").exists())

    def test_binding_requires_current_record_and_export_requires_saved_record(self):
        with self.assertRaisesRegex(ValueError, "开始原生录制"):
            biomulti.active_record(self.vendor)
        with closing(sqlite3.connect(self.vendor)) as db, db:
            db.execute("UPDATE record SET recoredEndTime=0")
        self.assertEqual(biomulti.active_record(self.vendor), {"record_id": 7, "created_ms": 1000})
        with self.assertRaisesRegex(ValueError, "保存"):
            self.align()

    def test_clock_reset_and_unsupported_units_are_rejected(self):
        self.frames[760]["time_mark"] = 10
        self.save_frames(self.frames)
        with self.assertRaisesRegex(ValueError, "重置"):
            self.align()
        for unit, factor, evidence in [("scan", None, "fixture"), ("ADC", -1., "fixture"), ("uV", None, "")]:
            with self.assertRaises(ValueError):
                biomulti.confirmed_calibration(self.source, unit, evidence, factor)

    def test_development_probe_requires_both_device_clock_and_physical_unit(self):
        clock = FakeClock()
        store = SessionStore(self.root, participant="SOFTWARE_FIXTURE", cohort="development", source=dict(self.source),
                             protocol=ProtocolConfig.short().as_dict(), clock=clock, notes="Constructed software test.")
        try:
            store.append(self.values, self.lsl, offset=0.)
            store.event("protocol_start")
            store.event("phase_start", {"phase":"baseline_calibration"})
            clock.advance(34.5)  # Include 30 s of usable baseline after filter warm-up and native clock offset.
            store.event("phase_end", {"phase":"baseline_calibration"})
            store.event("phase_start", {"phase":"sart_1", "block":1})
            clock.advance(13.5)
            probe = {"phase":"sart_1", "block":1, "probe_id":"fixture-probe", "practice":False}
            store.event("probe_onset", probe)
            store.event("probe_response", {**probe, "choice":3, "timeout":False})
            store.event("protocol_complete")
        finally:
            store.close("completed")
        summary = export_session(store.path, self.root / "dev-unknown", native_db=self.vendor)
        self.assertEqual(summary["train_eligible"], 0)
        summary = export_session(store.path, self.root / "dev-confirmed", native_db=self.vendor,
                                 calibration={"unit":"uV", "evidence":"generated in microvolts"})
        self.assertTrue(summary["baseline_valid"])
        self.assertEqual((summary["train_eligible"], summary["test_eligible"]), (1, 0))
        self.assertEqual(summary["probe_comparison"]["scored"], 0)  # no retrospective blind predictions
