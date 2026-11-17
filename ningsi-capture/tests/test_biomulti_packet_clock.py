from contextlib import closing
import csv
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .test_support import FakeClock
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, PACKET_TIMING, biomulti_packet_clock
from ningsi_capture import biomulti
from ningsi_capture.acquisition import synthetic_samples
from ningsi_capture.export import analysis_windows, export_session, recent_window_quality
from ningsi_capture.live_quality import CaptureQuality
from ningsi_capture.storage import SessionStore, open_readonly, read_metadata


def releases(count):
    # Two nearly simultaneous packet releases after a transport stall.
    lag = np.resize([0., .002, .003, .004, .085, .052, .020, .001], (count + 9) // 10)
    return 1000. + np.arange(count) / 250. + np.repeat(lag, 10)[:count]


class BioMultiPacketClockTests(unittest.TestCase):
    def test_overlapping_packets_and_an_actual_clock_break(self):
        raw = releases(10000)
        saved = raw.copy()
        times = biomulti_packet_clock(raw)
        self.assertTrue(np.any(np.diff(raw) < 0))
        self.assertTrue(np.isfinite(times).all())
        self.assertTrue(np.all(np.diff(times) > 0))
        np.testing.assert_array_equal(raw, saved)
        # Re-reading an interior interval, including one beginning mid-packet.
        part = biomulti_packet_clock(raw[7:9507], first_index=7)
        np.testing.assert_array_equal(part[1000-7:9000-7], times[1000:9000])
        raw[2420:] += .18
        broken = biomulti_packet_clock(raw)
        self.assertTrue(np.isnan(broken[2420:2430]).any())
        self.assertTrue(np.isfinite(broken[4000:]).all())
        raw[5000:] -= 5
        reset = biomulti_packet_clock(raw)
        self.assertTrue(np.all(np.diff(reset[np.isfinite(reset)]) > 0))
        self.assertTrue(np.isnan(reset[5000:5250]).all())

    def test_unknown_interval_and_nonpacket_source_do_not_gain_a_clock(self):
        raw = releases(4500)
        raw[1002] = np.nan
        raw[2250 + 3] -= .08
        times = biomulti_packet_clock(raw)
        self.assertTrue(np.isnan(times[1000:1250]).all())
        self.assertTrue(np.isnan(times[2250:2500]).all())
        self.assertTrue(np.isfinite(times[3000:]).all())

    def test_live_and_existing_record_export_match_without_rec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "resources/app/package.json"
            package.parent.mkdir(parents=True)
            package.write_text(json.dumps({"name": "BioMultiLite", "version": "1.0.9-E-Release"}), encoding="utf8")
            source = {"kind": "lsl", "type": "EEG", "data_origin": "real", "name": "BioMulti Lite EEG-00abcd",
                      "srate": 250., "channels": 2, "channel_labels": ["Ch1", "Ch2"], "declared_units": ["", ""],
                      "hostname": socket.gethostname(), "scale_to_uv": None, "source_unit": "unknown",
                      "timestamp_policy": PACKET_TIMING}
            with patch.object(biomulti, "default_database", return_value=root / "data.db"):
                unit = biomulti.automatic_unit(source)
                self.assertEqual((unit["source_unit"], unit["scale_to_uv"]), ("uV", 1.))
                self.assertEqual(unit["unit_basis"], "biomulti_software_inference")
                self.assertEqual(biomulti.automatic_unit({**source, "hostname": "another-host"}), {})
                self.assertEqual(biomulti.automatic_unit({**source, "declared_units": ["mV","mV"]})["scale_to_uv"], 1000.)
                self.assertEqual(biomulti.automatic_unit({**source, "declared_units": ["mV","uV"]}), {})
                self.assertEqual(biomulti.automatic_unit({**source, "declared_units": ["unknown","unknown"]}), unit)
                clock = FakeClock()
                store = SessionStore(root, participant="ARCHIVE", cohort="pilot", source=source, protocol={}, clock=clock)
                raw = releases(5000)
                data = synthetic_samples(range(len(raw)))
                try:
                    for start in range(0, len(raw), 100):
                        clock.seconds = max(clock.seconds, float(np.max(raw[start:start+100])) + .01)
                        store.append(data[start:start+100], raw[start:start+100], offset=0.)
                finally:
                    store.close("aborted")
                out = root / "export"
                result = export_session(store.path, out)
                self.assertGreater(result["windows_usable"], 0)
                self.assertEqual(result["train_eligible"], 0)
                np.testing.assert_array_equal(np.load(out / "timestamps.npy")[:, 0], raw)
                np.testing.assert_array_equal(np.load(out / "raw_samples.npy"), data)
                np.testing.assert_array_equal(np.load(out / "packet_timestamps.npy"), biomulti_packet_clock(raw))
                with (out / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
                    csv_times = [float(row["analysis_timestamp"]) for row in csv.DictReader(stream)]
                np.testing.assert_array_equal(csv_times, biomulti_packet_clock(raw))
                meta = json.loads((out / "manifest.json").read_text(encoding="utf8"))
                self.assertEqual(meta["source"]["timestamp_policy"], BIOMULTI_LSL_TIMING)
                self.assertIsNone(meta["source_at_capture"]["scale_to_uv"])
                with closing(open_readonly(store.path)) as db:
                    self.assertIsNone(read_metadata(db)["source"]["scale_to_uv"])
                    rows = analysis_windows(db, meta)
                    one, = analysis_windows(db, meta, first_index=2999, stop_index=3000)
                    expected = next(r for r in rows if r["sample_end_index"] == 2999)
                    self.assertEqual((one["start"], one["end"]), (expected["start"], expected["end"]))
                    late, = analysis_windows(db, meta, first_index=2999, stop_index=3000,
                                             received_by=raw[2500], sample_limit=3000)
                    self.assertFalse(late["window"].usable)
                    monitor = CaptureQuality()
                    for _ in range(3):
                        live = monitor.update(db, meta)
                    self.assertIn("分包时钟估计", live.reading)
                    self.assertIn("µV（软件推定）", live.reading)
                # The operator may explicitly choose raw-only in a new capture.
                store = SessionStore(root, participant="RAW", source={**source, "unit_selection": "raw_only",
                                     "timestamp_policy": BIOMULTI_LSL_TIMING}, protocol={}, clock=clock)
                try:
                    for start in range(0, 1250, 250):
                        clock.seconds = raw[start+249] + .02
                        store.append(data[start:start+250], raw[start:start+250], offset=0.)
                    self.assertIn("unit_unconfirmed", recent_window_quality(store.path))
                finally:
                    store.close("aborted")
                self.assertEqual(export_session(store.path, root / "raw_export")["windows_usable"], 0)
