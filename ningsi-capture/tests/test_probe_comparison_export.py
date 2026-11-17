import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest

from . import test_support
from ningsi_capture.comparison_export import comparison_table
from ningsi_capture.export import csv_write, recorded_predictions
from ningsi_capture.protocol import PROBE_CHOICES
from ningsi.models.capture_protocols import SHORT_PROTOCOL


def event(number, kind, probe, **info):
    return {"event_id": number, "time_lsl": 5000 + number, "time_perf": 1000 + number,
            "kind": kind, "payload": {"probe_id": probe, **info}}


def onset(number, name, ordinal, practice=False):
    return event(number, "probe_onset", name, phase="probe_practice" if practice else "sart_2",
                 block=0 if practice else 2, practice=practice, after_trial=24,
                 choices=list(PROBE_CHOICES), probe_ordinal=ordinal, onset_basis="tk_idle_estimate",
                 task_elapsed_sec=None if practice else 201.25, protocol_elapsed_sec=570.5)


class ProbeComparisonExport(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.meta = {"session_id": "s", "participant": "p", "visit": "02", "cohort": "development",
                     "source": {"data_origin": "real"}, "protocol": {"name": SHORT_PROTOCOL}}

    def export(self, events, predictions):
        result = comparison_table(self.meta, events, predictions, self.path, csv_write)
        with (self.path / result["file"]).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        return result, rows

    def test_freezes_both_scores_and_distinguishes_time_eligible_rows(self):
        events = [onset(1, "a", 3), event(2, "probe_response", "a", choice=5, response_rt_sec=.54),
                  onset(3, "b", 4), event(4, "probe_response", "b", choice=1, response_rt_sec=1.03)]
        scored = {"status": "scored", "logistic_score": .72, "rbf_score": .47, "model_bundle_id": "b250",
                  "model_ids": {"logistic": "lrhash", "rbf": "rbfhash"}, "baseline_valid": False,
                  "windows_total": 7, "windows_usable": 5, "usable_seconds": 14,
                  "physical_qc_status": "passed", "strict_qc_status": "failed", "strict_windows_usable": 2,
                  "strict_usable_seconds": 6, "strict_qc_reasons": ["insufficient_usable_eeg"],
                  "strict_window_rejections": {"input_eog": 5}, "live_input_window_status": "passed",
                  "live_input_window_end_lsl": 5000.8, "physical_qc_reasons": []}
        unavailable = {"status": "unavailable", "reason": "timestamp_gap_over_100ms"}
        records = {1: {"threshold": .2, "mw_score": .95, "window_start_lsl": 4989,
                       "sample_count_at_onset": 123, "shadow_250": scored},
                   3: {"shadow_250": unavailable}}
        summary, (a, b) = self.export(events, records)
        self.assertEqual((summary["scored"], summary["comparable"], summary["time_comparable"]), (1, 1, 2))
        self.assertEqual((a["logistic_mw_score"], a["rbf_mw_score"], a["model_bundle_id"],
                          a["logistic_model_id"], a["rbf_model_id"]), ("0.72", "0.47", "b250", "lrhash", "rbfhash"))
        self.assertEqual((a["window_start_lsl"], a["window_end_lsl"], a["sample_count_at_onset"]),
                         ("4983", "4999", "123"))
        self.assertEqual((a["participant"], a["visit"], a["probe_ordinal"], a["task_elapsed_sec"],
                          a["protocol_elapsed_sec"]), ("p", "02", "3", "201.25", "570.5"))
        self.assertEqual((a["choice"], a["choice_text"], a["label"], a["response_rt_sec"],
                          a["response_event_id"], a["response_lsl"]), ("5", PROBE_CHOICES[4], "1", "0.54", "2", "5002"))
        self.assertEqual((a["strict_qc_status"], a["physical_qc_status"], a["live_input_window_status"],
                          a["baseline_valid"], a["strict_window_rejections"], a["strict_qc_reasons"]),
                         ("failed", "passed", "passed", "False", '{"input_eog":5}', "insufficient_usable_eeg"))
        self.assertEqual((b["label"], b["time_comparable"], b["comparable"], b["reason"],
                          b["logistic_mw_score"], b["rbf_mw_score"]), ("0", "True", "False",
                          "timestamp_gap_over_100ms", "", ""))

    def test_missing_historical_model_timeout_cancellation_and_incomplete_onsets(self):
        events = [onset(1, "one", None, practice=True), event(2, "probe_response", "one", choice=2),
                  onset(3, "two", 1), event(4, "probe_response", "two", choice=None, timeout=True),
                  onset(5, "three", 2), event(6, "probe_cancelled", "three", reason="stopped"),
                  onset(7, "repeat", 3), onset(8, "repeat", 4), event(9, "probe_response", "repeat", choice=3)]
        with sqlite3.connect(":memory:") as db:
            no_predictions = recorded_predictions(db)
        summary, rows = self.export(events, no_predictions)
        self.assertEqual([r["response_status"] for r in rows], ["nonbinary", "timeout", "cancelled", "unanswered", "binary"])
        self.assertEqual([r["label"] for r in rows], ["", "", "", "", "1"])
        self.assertEqual([r["comparable"] for r in rows], ["False"] * 5)
        self.assertTrue(all(r["logistic_mw_score"] == r["rbf_mw_score"] == "" for r in rows))
        self.assertEqual((rows[0]["practice"], rows[0]["choice_text"], rows[0]["probe_ordinal"]),
                         ("True", PROBE_CHOICES[1], ""))
        self.assertEqual(summary["not_recorded"], 5)
        self.assertEqual(summary["probes_total"], 5)
        self.assertEqual(summary["time_comparable"], 1)


if __name__ == "__main__":
    unittest.main()
