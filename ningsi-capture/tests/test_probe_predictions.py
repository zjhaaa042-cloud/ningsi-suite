import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from .test_support import FakeClock
from ningsi.models import attlapse
from ningsi_capture.acquisition import simulation_source, synthetic_samples
from ningsi_capture.export import export_session
from ningsi_capture.protocol import ProtocolConfig
from ningsi_capture.storage import SessionStore


class ProbePredictionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.clock = FakeClock()
        self.model = {"model_id": "software-fixture", "threshold": .5,
                      "parameters": {"weights": [0] * 6, "scale": [1] * 6, "mean": [0] * 6, "bias": 1.}}
        self.loader = patch.object(attlapse, "load_candidate", return_value=self.model).start()
        self.addCleanup(patch.stopall)

    def store(self, source=None):
        # The hardware declaration exercises the real-source branch using explicitly synthetic test signals.
        source = source or {**simulation_source(), "kind": "lsl", "data_origin": "real",
                             "name": "EEG test fixture", "channel_labels": ["Fp2", "Fp1"]}
        self.session = SessionStore(self.root, participant="FIXTURE", source=source,
                                    cohort="pilot", protocol=ProtocolConfig.short().as_dict(), clock=self.clock)
        self.addCleanup(self.session.close)
        return self.session

    def append(self, left, right, received=None):
        points = np.arange(left * 250, right * 250)
        values = synthetic_samples(points)
        values[:, 1] *= -.85  # Distinct, usable differential derivation.
        self.clock.seconds = 1000 + (right if received is None else received)
        self.session.append(values, 1000 + points / 250, offset=0.)

    def probe(self, name, choice=None, practice=False, status="answered"):
        payload = {"probe_id": name, "phase": "sart_1", "practice": practice}
        self.session.event("probe_onset", payload)
        if status == "cancelled":
            self.session.event("probe_cancelled", payload)
        elif status != "unanswered":
            self.session.event("probe_response", {**payload, "choice": choice, "timeout": status == "timeout"})

    def test_delayed_worker_is_causal_and_answers_do_not_enter_the_prediction(self):
        store = self.store()
        store.event("phase_start", {"phase": "baseline_calibration"})
        self.append(0, 40)
        store.event("phase_end", {"phase": "baseline_calibration"})
        store.event("phase_start", {"phase": "sart_1"})
        self.append(40, 60)
        # Deliberately retain all background jobs until after every answer and late chunk has arrived.
        release = threading.Event()
        store.predictions.executor.submit(release.wait, 20)
        try:
            for name, choice, status in (("ot", 1, "answered"), ("mw", 3, "answered"), ("other", 6, "answered"),
                                         ("timeout", None, "timeout"), ("cancel", None, "cancelled"),
                                         ("missing", None, "unanswered")):
                self.probe(name, choice, status=status)
            self.clock.seconds = 1080
            self.probe("late", 5)  # No EEG around this probe has arrived.
            self.append(60, 80)
        finally:
            release.set()
            store.close("aborted")
        out = self.root / "export"
        summary = export_session(store.path, out)
        with (out / "probe_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(summary["probe_predictions"]["scored"], 6, rows)
        self.assertEqual(summary["probe_predictions"]["comparable"], 2)
        self.assertEqual(summary["probe_predictions"]["agreed"], 1)
        self.assertEqual(summary["probe_predictions"]["model_id"], self.model["model_id"])
        self.assertEqual([r["agreement"] for r in rows], ["False", "True", "", "", "", "", ""])
        self.assertEqual(len({r["mw_score"] for r in rows[:-1]}), 1)
        self.assertEqual([r["response_status"] for r in rows[2:6]], ["nonbinary", "timeout", "cancelled", "unanswered"])
        self.assertEqual(rows[-1]["reason"], "insufficient_usable_eeg")
        self.assertEqual(rows[-1]["mw_score"], "")
        with (out / "features.csv").open(encoding="utf-8-sig", newline="") as file:
            training = list(csv.DictReader(file))
        self.assertNotIn("mw_score", training[0])
        self.assertTrue(all(row["train_eligible"] == "False" for row in training))

    def test_simulated_and_old_records_never_acquire_a_replayed_score(self):
        store = self.store(simulation_source())
        self.probe("s1", 1)
        self.probe("practice", 3, practice=True)
        store.close("aborted")
        with sqlite3.connect(store.path / "recording.sqlite3") as writer:
            scored = [json.loads(row[0]) for row in writer.execute("SELECT result FROM probe_predictions ORDER BY onset_event_id")]
            self.assertEqual([r["reason"] for r in scored], ["simulation_unverified", "practice"])
            writer.execute("DROP TABLE probe_predictions")
            meta = json.loads(writer.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0])
            del meta["probe_prediction"]
            writer.execute("UPDATE meta SET value=? WHERE key='session'", (json.dumps(meta),))
        writer.close()
        summary = export_session(store.path, self.root / "old-export")
        self.assertEqual(summary["probe_predictions"]["not_recorded"], 2)
        self.assertEqual(summary["probe_predictions"]["scored"], 0)

    def test_missing_or_incompatible_inputs_leave_actions_available(self):
        source = {**simulation_source(), "kind": "lsl", "data_origin": "real", "channel_labels": ["Ch1", "Ch2"]}
        for change, reason in (({}, "electrode_mismatch"), ({"scale_to_uv": None}, "unit_unconfirmed"),
                               ({"declared_units": ["mV", "mV"]}, "unit_conflict"),
                               ({"type": "quality"}, "source_type_mismatch")):
            with self.subTest(reason=reason):
                store = self.store({**source, **change})
                self.probe("p", 1)
                store.close()
                with sqlite3.connect(store.path / "recording.sqlite3") as db:
                    result = json.loads(db.execute("SELECT result FROM probe_predictions").fetchone()[0])
                db.close()
                self.assertEqual(result["reason"], reason)
        self.loader.return_value = None
        store = self.store()
        self.probe("p", 1)
        store.close()
        with sqlite3.connect(store.path / "recording.sqlite3") as db:
            self.assertEqual(json.loads(db.execute("SELECT result FROM probe_predictions").fetchone()[0])["reason"],
                             "model_unavailable")
        db.close()
