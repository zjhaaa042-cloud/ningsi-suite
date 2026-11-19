from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from ningsi_capture.bootstrap import ensure_engine

ensure_engine()

from ningsi_capture.native_motion import MOTION_CONFIG, VENDOR_VERSION, load_native_motion


class NativeMotionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.db_path = root / "data.db"
        self.manifest = root / "resources/app/package.json"
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text(json.dumps({"name": "BioMultiLite", "version": VENDOR_VERSION}), encoding="utf-8")
        self.binding = {"record_id": 1, "created_ms": 10000}
        self.eeg = [(10 * i, i, 20000 + 40 * i, 100 + .04 * i, 100 + .04 * i) for i in range(150)]
        self.axis = 100 - .036 + np.arange(1500) * .004
        self.frames = [
            {"pkg_type": 1, "pkgnum": number, "time_mark": tick, "eeg_channel": 2,
             "eeg_data_num": 10, "isLosspkg": False}
            for _, number, tick, _, _ in self.eeg
        ]
        self.frames += [
            {"pkg_type": 3, "pkgnum": i, "time_mark": 20000 + 80 * i, "isLosspkg": False,
             "gyro_x": .6 if i == 17 else .1, "gyro_y": .8 if i == 17 else .1, "gyro_z": 0}
            for i in range(75)
        ]
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.executescript(
                "CREATE TABLE record(id, recoredCreateTime, recoredEndTime, configText);"
                "CREATE TABLE source(id, recordId, data);"
            )
            config = {"eeg_channel_count": 2, "eeg_sample_rate": 250, "motion": MOTION_CONFIG}
            connection.execute("INSERT INTO record VALUES (1, 10000, 20000, ?)", (json.dumps(config),))
            connection.execute("INSERT INTO source VALUES (1, 1, ?)", (json.dumps(self.frames),))

    def load(self):
        return load_native_motion(self.db_path, self.binding, iter(self.eeg), self.axis)

    def change_frames(self):
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("UPDATE source SET data=?", (json.dumps(self.frames),))

    def test_bracketed_measured_peak_on_matched_eeg_clock(self):
        trace, audit = self.load()
        self.assertEqual(audit["status"], "available")
        self.assertEqual(audit["packets"], 75)
        self.assertAlmostEqual(trace.times_lsl[17], 101.36)
        self.assertFalse(trace.times_lsl.flags.writeable)
        motion = trace.assess([101.30, 101.39], threshold=.7)
        self.assertEqual(motion["status"], "motion")
        self.assertAlmostEqual(motion["peak"], 1.)
        self.assertEqual(trace.assess([100.16, 100.21], threshold=.7)["status"], "quiet")
        self.assertEqual(trace.assess([100.16, 100.21])["status"], "observed")
        self.assertEqual(trace.assess([99.999, 100.02], threshold=.7)["status"], "unavailable")
        self.assertEqual(trace.assess([105.9, 105.93], threshold=.7)["status"], "unavailable")

    def test_lost_or_malformed_packet_cannot_prove_quiet_across_gap(self):
        self.frames[150 + 19]["gyro_y"] = None
        self.change_frames()
        trace, audit = self.load()
        self.assertEqual(audit["status"], "available")
        self.assertEqual(trace.assess([101.4, 101.56], threshold=.7)["reason"], "gyro_packet_gap_in_eeg_span")
        self.assertEqual(trace.assess([100.16, 100.21], threshold=.7)["status"], "quiet")

    def test_bad_binding_or_clock_cannot_produce_motion_trace(self):
        self.binding["created_ms"] += 1
        trace, audit = self.load()
        self.assertIsNone(trace)
        self.assertEqual(audit["reason"], "native_record_not_closed_or_binding_mismatch")
        self.binding["created_ms"] -= 1
        self.axis[9] += .01
        trace, audit = self.load()
        self.assertIsNone(trace)
        self.assertEqual(audit["reason"], "eeg_analysis_clock_anchor_mismatch")

    def test_constant_native_gyro_cannot_confirm_quiet(self):
        for frame in self.frames[150:]:
            frame.update(gyro_x=0, gyro_y=0, gyro_z=0)
        self.change_frames()
        trace, audit = self.load()
        self.assertIsNone(trace)
        self.assertEqual(audit["reason"], "constant_native_gyro_trace")

    def test_cannot_use_duplicate_native_eeg_anchor_or_unknown_version(self):
        self.frames.append(dict(self.frames[0]))
        self.change_frames()
        trace, audit = self.load()
        self.assertIsNone(trace)
        self.assertEqual(audit["reason"], "native_eeg_anchor_record_mismatch")
        self.manifest.write_text('{"name":"BioMultiLite","version":"unknown"}', encoding="utf-8")
        trace, audit = self.load()
        self.assertIsNone(trace)
        self.assertEqual(audit["reason"], "unsupported_biomulti_version")
