import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from .test_support import FakeClock
import numpy as np
from ningsi.acquisition.timing import PACKET_TIMING
from ningsi_capture.acquisition import simulation_source
from ningsi_capture.demo import create_demo
from ningsi_capture.export import FEATURES, PROBE_FIELDS, build_dataset, csv_write, export_session, recent_window_quality
from ningsi_capture.storage import SessionStore, engine_fingerprint, write_json


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.session = create_demo(cls.root / "recordings")
        cls.output = cls.root / "export"
        cls.summary = export_session(cls.session, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_raw_and_event_export_preserves_count_and_values(self):
        raw = np.load(self.output / "raw_samples.npy")
        times = np.load(self.output / "timestamps.npy")
        self.assertEqual(raw.shape, (26000, 2))
        self.assertEqual(times.shape, (26000, 4))
        self.assertEqual(times[0, 0], 1000.)
        self.assertEqual(times[-1, 0], 1103.996)
        self.assertEqual(self.summary["raw_samples"], 26000)
        events = [json.loads(line) for line in (self.output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(e["kind"] == "probe_onset" for e in events), 3)
        self.assertTrue((self.output / "SHA256.json").is_file())
        self.assertNotIn("artifact_calibration", self.summary)

    def test_personal_artifact_detection_is_an_independent_hashed_export(self):
        from .test_artifact_calibration import _record
        from ningsi_capture.clock import Stamp

        raw, stamps, events, _, _ = _record({"blink": "blink", "jaw": "muscle"})
        clock = FakeClock(stamps[0])
        clock.domain = "lsl_local_clock"
        source = {**simulation_source(), "kind": "lsl", "type": "EEG", "data_origin": "real"}
        store = SessionStore(self.root / "recordings", participant="ARTIFACT", source=source, cohort="pilot",
                             protocol={"personal_artifact_calibration": True}, clock=clock)
        try:
            store.append(raw.T, stamps, offset=0.)
            for e in events:
                t = e["time_lsl"]
                store.event(e["kind"], e["payload"], Stamp(t, t, "test-time"))
        finally:
            clock.seconds = float(stamps[-1])
            store.close("completed")

        output = self.root / "calibration_export"
        summary = export_session(store.path, output)
        calibration = summary["artifact_calibration"]
        self.assertEqual((calibration["status"], calibration["calibrated_actions"]), ("partial", 2))
        self.assertEqual(summary["train_eligible"], 0)
        report = json.loads((output / calibration["report_file"]).read_text(encoding="utf-8"))
        self.assertEqual(report["session_id"], summary["session_id"])
        self.assertEqual(report["actions"]["blink"]["validation_hits"], 2)
        rows = read_csv(output / calibration["detections_file"])
        self.assertEqual(rows[0]["actions"], "blink")
        self.assertEqual(json.loads(rows[0]["events"])[0]["action"], "blink")
        hashes = json.loads((output / "SHA256.json").read_text(encoding="utf-8"))
        self.assertIn(calibration["report_file"], hashes)
        self.assertIn(calibration["detections_file"], hashes)

    def test_one_row_per_probe_correct_label_mapping_and_no_simulated_training(self):
        rows = read_csv(self.output / "features.csv")
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["label"] for r in rows], ["0", "1", "1"])
        self.assertTrue(self.summary["baseline_valid"])
        self.assertEqual(self.summary["feature_rows_usable"], 3)
        self.assertEqual(self.summary["train_eligible"], 0)
        self.assertTrue(all(r["train_eligible"] == r["test_eligible"] == "False" for r in rows))
        self.assertTrue(all(r["window_end_lsl"] == str(float(r["onset_lsl"]) - 2) for r in rows))

    def test_local_quality_review_keeps_timed_descriptive_probes_in_a_separate_hashed_file(self):
        review = self.summary["local_quality_review"]
        self.assertEqual(review["windows_total"], self.summary["windows_total"])
        self.assertEqual(review["windows_strict"], self.summary["windows_usable"])
        self.assertEqual(review["formal_probes_with_descriptive_spectrum"], 3)
        self.assertEqual(review["formal_probes_with_at_least_6_sec"], 3)
        features = {row["probe_id"]: row for row in read_csv(self.output / "features.csv")}
        periods = read_csv(self.output / review["periods_file"])
        self.assertIn("baseline", {row["kind"] for row in periods})
        for row in (period for period in periods if period["kind"] == "probe"):
            formal = features[row["probe_id"]]
            lower, upper = float(formal["onset_lsl"]) - 12, float(formal["onset_lsl"]) - 2
            cursor = lower
            intervals = json.loads(row["intervals_lsl"])
            for start, stop in intervals:
                self.assertLessEqual(cursor, start)
                self.assertLessEqual(stop, upper + 1e-9)
                cursor = stop
            self.assertAlmostEqual(float(row["support_sec"]), sum(stop - start for start, stop in intervals), places=5)
            self.assertTrue(float(row["rel_alpha"]) > 0)
            self.assertEqual(formal["train_eligible"], "False")
        hashes = json.loads((self.output / "SHA256.json").read_text(encoding="utf-8"))
        for key in ("spans_file", "windows_file", "periods_file"):
            self.assertIn(review[key], hashes)

    def test_local_quality_review_does_not_use_per_window_clock_fit_as_a_continuous_axis(self):
        import shutil
        source = self.root / "per_window_clock"
        shutil.copytree(self.session, source)
        with sqlite3.connect(source / "recording.sqlite3") as connection:
            metadata = json.loads(connection.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0])
            metadata["source"]["timestamp_policy"] = PACKET_TIMING
            connection.execute("UPDATE meta SET value=? WHERE key='session'", (json.dumps(metadata),))
        output = self.root / "per_window_clock_export"
        summary = export_session(source, output)
        review = summary["local_quality_review"]
        self.assertEqual(review["clock_axis"], "corrected_before_per_window_fit")
        self.assertEqual(review["formal_probes_with_descriptive_spectrum"], 0)
        rows = read_csv(output / review["periods_file"])
        self.assertTrue(rows)
        self.assertEqual({row["reason"] for row in rows}, {"continuous_analysis_clock_unavailable"})
        self.assertEqual(summary["feature_rows_usable"], 3)

    def test_formal_export_readiness_counts_both_classes_and_respects_preassignment(self):
        import shutil
        for cohort, expected in (("development", (3, 0)), ("test", (0, 3)), ("pilot", (0, 0))):
            with self.subTest(cohort=cohort):
                # Exercise the real-data branch only inside this constructed software fixture.
                session = self.root / ("fixture_" + cohort)
                shutil.copytree(self.session, session)
                connection = sqlite3.connect(session / "recording.sqlite3")
                try:
                    metadata = json.loads(connection.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0])
                    metadata["cohort"] = cohort
                    metadata["protocol"]["name"] = "attention-validation-short-v1"
                    metadata["source"].update(kind="lsl", type="EEG", data_origin="real")
                    with connection:
                        connection.execute("UPDATE meta SET value=? WHERE key='session'", (json.dumps(metadata),))
                finally:
                    connection.close()
                summary = export_session(session, self.root / ("checked_" + cohort))
                self.assertEqual((summary["train_eligible"], summary["test_eligible"]), expected)
                result = summary["training_readiness"]
                self.assertEqual((result["formal_probes"], result["on_task"], result["internal_mind_wandering"],
                                  result["usable_eeg"], result["paired"]), (3, 1, 2, 3, 3))
                self.assertEqual((result["eligible_on_task"], result["eligible_internal_mind_wandering"]),
                                 (1, 2) if cohort != "pilot" else (0, 0))

    def test_features_do_not_read_future_samples(self):
        import shutil
        destination = self.root / "future_modified"
        shutil.copytree(self.session, destination)
        connection = sqlite3.connect(destination / "recording.sqlite3")
        with connection:
            connection.execute("UPDATE samples SET values_f64=? WHERE sample_index BETWEEN 16000 AND 17000",
                               (np.array([1e7, -1e7], dtype="<f8").tobytes(),))
        connection.close()
        output = self.root / "future_export"
        export_session(destination, output)
        old, new = read_csv(self.output / "features.csv")[0], read_csv(output / "features.csv")[0]
        self.assertEqual({k: old[k] for k in FEATURES}, {k: new[k] for k in FEATURES})

    def test_no_overwrite_of_existing_export(self):
        with self.assertRaises(FileExistsError):
            export_session(self.session, self.output)

    def test_failed_checksum_leaves_original_and_allows_retry_including_empty_record(self):
        store = SessionStore(self.root / "recordings", participant="EMPTY", source=simulation_source(),
                             protocol={}, clock=FakeClock())
        store.close("failed", "no_samples")
        destination = self.root / "retried"
        with patch("hashlib.file_digest", side_effect=OSError("injected write failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                export_session(store.path, destination)
        self.assertFalse(destination.exists())
        result = export_session(store.path, destination)
        self.assertEqual(result["raw_samples"], 0)
        self.assertEqual(np.load(destination / "raw_samples.npy").shape, (0, 2))
        self.assertEqual(np.load(destination / "timestamps.npy").shape, (0, 4))
        hashes = json.loads((destination / "SHA256.json").read_bytes())
        self.assertFalse(any(name.endswith(("-wal", "-shm", ".pending")) for name in hashes))
        self.assertEqual(set(hashes) | {"SHA256.json"}, {p.name for p in destination.iterdir()})

    def test_precheck_uses_original_values_and_corrected_timestamps(self):
        from ningsi_capture.acquisition import synthetic_samples
        store = SessionStore(self.root / "recordings", participant="QUALITY", source=simulation_source(),
                             protocol={}, clock=FakeClock())
        try:
            points = np.arange(1500)
            values = synthetic_samples(points)
            store.append(values, 1000. + points / 250)
            self.assertIn("clock_unsynchronized", recent_window_quality(store.path))
            store.append(values, 1006. + points / 250, offset=.05)
            self.assertEqual(recent_window_quality(store.path), ())
            store.append(np.zeros_like(values), 1012. + points / 250, offset=.05)
            self.assertIn("flat_channel", recent_window_quality(store.path))
        finally:
            store.close("aborted")

    def test_old_conflicting_units_keep_raw_but_cannot_make_features(self):
        import shutil
        destination, output = self.root / "unit_conflict", self.root / "unit_conflict_export"
        shutil.copytree(self.session, destination)
        connection = sqlite3.connect(destination / "recording.sqlite3")
        with connection:
            metadata = json.loads(connection.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0])
            metadata["source"]["declared_units"] = ["mV", "mV"]
            connection.execute("UPDATE meta SET value=? WHERE key='session'", (json.dumps(metadata),))
        connection.close()
        result = export_session(destination, output)
        np.testing.assert_array_equal(np.load(output / "raw_samples.npy"), np.load(self.output / "raw_samples.npy"))
        self.assertEqual(result["feature_rows_usable"], 0)
        self.assertTrue(result["unit_conflict"])
        self.assertTrue(all("unit_conflict" in r["unavailable_reasons"] for r in read_csv(output / "features.csv")))

    def test_merge_checks_original_feature_bytes_before_creating_dataset(self):
        import shutil
        copy = self.root / "edited_export"
        copy.mkdir()
        for name in ("manifest.json", "export.json", "features.csv", "SHA256.json"):
            shutil.copyfile(self.output / name, copy / name)
        with (copy / "features.csv").open("ab") as file:
            file.write(b"\r\n")
        destination = self.root / "edited_dataset"
        with self.assertRaisesRegex(ValueError, "校验失败"):
            build_dataset([copy], destination)
        self.assertFalse(destination.exists())

    def test_active_recording_requires_explicit_recovery_and_is_not_training(self):
        store = SessionStore(self.root / "recordings", participant="ACTIVE", source=simulation_source(), protocol={}, clock=FakeClock())
        try:
            store.append([[1, 2], [3, 4]], [1000., 1000.004], offset=0.)
            with self.assertRaises(ValueError):
                export_session(store.path, self.root / "active_export")
            result = export_session(store.path, self.root / "recovery_export", recover=True)
            self.assertEqual(result["raw_samples"], 2)
            self.assertEqual(result["source_state"], "recording")
            self.assertEqual(result["train_eligible"], 0)
        finally:
            store.close("aborted")

    def test_unknown_unit_retains_raw_but_disables_features(self):
        import shutil
        destination = self.root / "unknown_unit"
        shutil.copytree(self.session, destination)
        connection = sqlite3.connect(destination / "recording.sqlite3")
        metadata = json.loads(connection.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0])
        metadata["source"]["scale_to_uv"] = None
        with connection:
            connection.execute("UPDATE meta SET value=? WHERE key='session'", (json.dumps(metadata),))
        connection.close()
        output = self.root / "unknown_export"
        result = export_session(destination, output)
        self.assertEqual(result["raw_samples"], 26000)
        self.assertEqual(result["feature_rows_usable"], 0)
        self.assertFalse(result["baseline_valid"])
        self.assertTrue(all("unit_unconfirmed" in r["reasons"] for r in read_csv(output / "windows.csv")))

    def test_engine_drift_is_audited_and_blocks_training(self):
        import shutil
        destination = self.root / "engine_drift"
        shutil.copytree(self.session, destination)
        connection = sqlite3.connect(destination / "recording.sqlite3")
        metadata = json.loads(connection.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0])
        metadata.update(cohort="development")
        metadata["source"]["data_origin"] = "real"  # Deliberate test-only metadata fixture.
        metadata["protocol"]["name"] = "attention-validation-v1"
        metadata["engine_fingerprint_at_capture"] = {}
        with connection:
            connection.execute("UPDATE meta SET value=? WHERE key='session'", (json.dumps(metadata),))
        connection.close()
        output = self.root / "drift_export"
        result = export_session(destination, output)
        self.assertTrue(result["engine_changed_since_capture"])
        self.assertEqual(result["train_eligible"], 0)
        self.assertTrue(all("engine_changed_since_capture" in r["unavailable_reasons"] for r in read_csv(output / "features.csv")))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def fixture(self, folder, participant, cohort, session, origin="real", fingerprint=None):
        path = self.root / folder
        path.mkdir()
        fingerprint = fingerprint or engine_fingerprint()
        metadata = {"participant": participant, "visit": "01", "cohort": cohort, "engine_fingerprint_at_capture": fingerprint,
                    "session_id": session, "source": {"data_origin": origin}, "state": "completed",
                    "protocol": {"name": "attention-validation-v1"}}
        write_json(path / "manifest.json", metadata)
        write_json(path / "export.json", {"schema": "ningsi-capture-export-v1", "engine_fingerprint_at_export": fingerprint,
                                           "engine_changed_since_capture": False, "protocol_complete": True})
        row = {name: "" for name in PROBE_FIELDS}
        row.update({key: metadata[key] for key in ("participant", "visit", "cohort", "session_id")})
        row.update(data_origin=origin, probe_id="p1", label="1", label_name="internal_mind_wandering", choice=3,
                   practice=False, timeout=False, cancelled=False, baseline_valid=True, feature_usable=True,
                   binary_label_valid=True, focus=.5, windows_usable=4, windows_total=4, usable_seconds=10,
                   train_eligible=cohort == "development",
                   test_eligible=cohort == "test", **{name: 0.25 for name in FEATURES})
        csv_write(path / "features.csv", PROBE_FIELDS, [row])
        return path

    def test_development_and_test_are_separate_and_simulation_excluded(self):
        dev = self.fixture("dev", "P1", "development", "s1")
        test = self.fixture("test", "P2", "test", "s2")
        sim = self.fixture("sim", "P3", "development", "s3", "simulation")
        result = build_dataset([dev, test, sim], self.root / "dataset")
        self.assertEqual(result["development_rows"], 1)
        self.assertEqual(result["test_rows"], 1)
        self.assertEqual(result["excluded_rows"], 1)

    def test_same_participant_across_cohorts_is_rejected(self):
        dev = self.fixture("dev", "P1", "development", "s1")
        test = self.fixture("test", "P1", "test", "s2")
        with self.assertRaisesRegex(ValueError, "跨分组"):
            build_dataset([dev, test], self.root / "dataset")

    def test_duplicate_session_probe_is_rejected(self):
        dev = self.fixture("dev", "P1", "development", "s1")
        with self.assertRaisesRegex(ValueError, "重复"):
            build_dataset([dev, dev], self.root / "dataset")

    def test_mixed_engine_versions_are_rejected(self):
        first = self.fixture("dev", "P1", "development", "s1")
        second = self.fixture("test", "P2", "test", "s2", fingerprint={**engine_fingerprint(), "config.py": "b"*64})
        with self.assertRaisesRegex(ValueError, "不同版本"):
            build_dataset([first, second], self.root / "dataset")


if __name__ == "__main__":
    unittest.main()
