import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ningsi_studio.tools.field_validation import evaluate


class FieldValidationTests(unittest.TestCase):
    def test_old_database_and_prospective_pairing(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "studio.sqlite3"
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("CREATE TABLE sessions(id INTEGER, subject_id INTEGER, status TEXT, time_scale REAL)")
            self.assertFalse(evaluate(path)["database_schema_has_checkins"])
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("INSERT INTO sessions VALUES (1,1,'done',1)")
                conn.execute("CREATE TABLE state_checkins(id INTEGER PRIMARY KEY, session_id INTEGER, "
                             "source_kind TEXT, phase TEXT, t_sec REAL, stress INTEGER, "
                             "valence INTEGER, eeg_context TEXT)")
                for t, age, rating, score, source in [(10, 1, 2, .2, "lsl"), (11, 2, 8, .2, "lsl"),
                        (20, 1, 8, .8, "lsl"), (30, 5, 9, .8, "lsl"), (40, 1, 8, .1, "sim")]:
                    prediction = {"stress": {"available": True, "score": score, "model_id": "v1"}}
                    context = json.dumps({"elapsed_sec": t - age, "phase": "monitor", "quality_ok": True,
                                          "prediction": prediction})
                    conn.execute("INSERT INTO state_checkins VALUES(NULL,1,?,'monitor',?,?,5,?)",
                                 (source, t, rating, context))
            result = evaluate(path)["dimensions"]
            stress = result["stress"]
            self.assertEqual((stress["rated"], stress["repeated_window"], stress["with_prediction"]),
                             (4, 1, 2))
            self.assertAlmostEqual(stress["coverage"], 2 / 3)
            metrics = stress["by_frozen_model"]["v1"]
            self.assertEqual(metrics["within_person_macro_auc"], 1.0)
            self.assertIsNone(metrics["participant_bootstrap_ci95"])
            self.assertIsNone(result["valence"]["coverage"])
