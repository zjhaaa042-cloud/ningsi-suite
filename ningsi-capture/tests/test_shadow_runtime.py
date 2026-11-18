import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from .test_support import FakeClock
from ningsi_capture.acquisition import simulation_source, synthetic_samples
from ningsi_capture.probe_prediction import RECOVERY_INPUT_POLICY, _tcn_score
from ningsi_capture.protocol import ProtocolConfig
from ningsi_capture.storage import SessionStore
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING
from ningsi.models import attlapse_shadow, focus_tcn


class ShadowRuntimeTest(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.clock = FakeClock()
        self.source = {**simulation_source(), "kind": "lsl", "data_origin": "real",
                       "channel_labels": ["Fp2", "Fp1"], "name": "two-channel replay fixture"}
        self.root = Path(root.name)

    def session(self):
        session = SessionStore(self.root, participant="FIXTURE", source=self.source,
                               protocol=ProtocolConfig.short().as_dict(), clock=self.clock)
        self.addCleanup(session.close)
        return session

    def read(self, store):
        with sqlite3.connect(store.path / "recording.sqlite3") as db:
            rows = db.execute("SELECT events.payload, probe_predictions.result FROM events LEFT JOIN "
                              "probe_predictions ON onset_event_id=event_id "
                              "WHERE kind='probe_onset' ORDER BY event_id").fetchall()
        db.close()
        return [(json.loads(ev), json.loads(result)) for ev, result in rows]

    def test_task_time_and_ordinals_across_practice_and_breaks(self):
        store = self.session()
        self.clock.advance(17)
        store.event("protocol_start")
        self.clock.advance(2)
        store.event("phase_start", {"phase": "probe_practice", "practice": True})
        store.event("probe_onset", {"phase": "probe_practice", "practice": True})
        self.clock.advance(23)
        store.event("phase_start", {"phase": "sart_1"})
        self.clock.advance(22)
        store.event("probe_onset", {"phase": "sart_1", "probe_id": "first"})
        store.event("phase_end", {"phase": "sart_1"})
        self.clock.advance(62)
        store.event("phase_start", {"phase": "sart_2"})
        self.clock.advance(20)
        store.event("probe_onset", {"phase": "sart_2", "probe_id": "last"})
        store.event("probe_cancelled", {"phase": "sart_2", "probe_id": "last"})
        store.close("aborted")
        events, predictions = zip(*self.read(store))
        self.assertEqual([e["probe_ordinal"] for e in events], [None, 1, 2])
        self.assertEqual([e["task_elapsed_sec"] for e in events], [None, 22, 104])
        self.assertEqual([e["protocol_elapsed_sec"] for e in events], [2, 47, 129])
        self.assertEqual(predictions[0]["shadow_250"]["reason"], "practice")
        self.assertTrue(all(p["shadow_250"]["status"] == "unavailable" for p in predictions))
        self.assertEqual(predictions[0]["tcn_250"]["reason"], "practice")
        self.assertTrue(all(p["tcn_250"]["status"] == "unavailable" for p in predictions))

    def test_result_is_frozen_before_answer_even_when_worker_runs_afterwards(self):
        store = self.session()
        store.event("phase_start", {"phase": "sart_1"})
        points = np.arange(20 * 250)
        values = synthetic_samples(points)
        values[:, 1] *= -.8
        # Poison only the prompt-adjacent 2 s. Model windows end before it.
        values[18 * 250:] = 100000 * np.sin(points[18 * 250:, None])
        self.clock.advance(20)
        store.append(values, 1000 + points / 250, offset=0.)
        release = threading.Event()
        store.predictions.executor.submit(release.wait, 10)
        try:
            for probe, choice in (("p-1", 1), ("p-2", 5)):
                payload = {"phase": "sart_1", "probe_id": probe}
                store.event("probe_onset", payload)
                store.event("probe_response", {**payload, "choice": choice})
            self.clock.advance(30)
            store.event("probe_onset", {"phase": "sart_1", "probe_id": "late-data"})
            late = np.arange(20 * 250, 50 * 250)
            store.append(synthetic_samples(late), 1000 + late / 250, offset=0.)
        finally:
            release.set()
            store.close("aborted")
        recs = [item[1] for item in self.read(store)]
        first, second, late = [r["shadow_250"] for r in recs]
        self.assertEqual(recs[0]["reason"], "baseline_invalid")
        self.assertEqual(first["status"], "scored", first)
        self.assertEqual(first, second)
        self.assertEqual((first["window_start_lsl"], first["window_end_lsl"]), (1002., 1018.))
        self.assertIsNone(first["baseline_valid"])
        self.assertEqual(first["live_input_window_status"], "failed")
        self.assertEqual(first["live_input_window_end_lsl"], 1020.)
        self.assertEqual(first["physical_qc_status"], "passed")
        self.assertEqual(first["windows_total"], 7)
        self.assertEqual(first["windows_usable"], 7)
        # Independently reconstruct the intended aligned contexts from the original array.
        windows = []
        for end in range(6 * 250, 19 * 250, 2 * 250):
            sl = slice(end - 6 * 250, end)
            windows.append(attlapse_shadow.compute_window(values[sl].T, 1000 + points[sl] / 250, 250, 1, (1, 0)))
        expected = attlapse_shadow.aggregate_and_score(attlapse_shadow.load_candidates(), windows, 1020., 1000.)
        self.assertAlmostEqual(first["logistic_score"], expected["logistic_score"], places=12)
        self.assertAlmostEqual(first["rbf_score"], expected["rbf_score"], places=12)
        tcn_first, tcn_second, tcn_late = [r["tcn_250"] for r in recs]
        model = focus_tcn.load_candidate()
        frames = [[row["features"][name] for name in attlapse_shadow.FEATURE_NAMES] for row in windows[-6:]]
        expected_tcn = focus_tcn.score_sequence(model, frames, np.ones(6, dtype=bool))
        self.assertEqual(tcn_first["status"], "scored", tcn_first)
        self.assertEqual(tcn_first["frame_mask"], "111111")
        self.assertEqual(tcn_first["frame_end_sample_indices"], [end * 250 - 1 for end in range(8, 19, 2)])
        self.assertEqual(tcn_first["model_id"], store.metadata["tcn_250"]["model_id"])
        self.assertEqual(tcn_first, tcn_second)
        self.assertAlmostEqual(tcn_first["mw_score"], expected_tcn, places=12)
        self.assertEqual(tcn_late["status"], "unavailable")
        self.assertIsNone(tcn_late["mw_score"])
        self.assertEqual(tcn_late["frame_mask"], "000000")
        self.assertEqual(late["status"], "unavailable")
        self.assertIsNone(late["rbf_score"])
        self.assertEqual(late["windows_usable"], 0)
        self.assertEqual(late["live_input_window_status"], "failed")
        self.assertTrue({"signal_stale", "quality_window_stale", "window_clock_difference_over_3s"}.issubset(
            late["live_input_window_reasons"]))
        self.assertEqual(late["live_input_last_packet_age_sec"], 30)

    def test_tcn_masks_rejected_and_absent_slots_without_moving_remaining_frames(self):
        def window(end_sec, rejected=False, strict_only=False):
            return {"start": 1000 + end_sec - 4, "end": 1000 + end_sec,
                    "sample_end_index": end_sec * 250 - 1,
                    "features": dict.fromkeys(attlapse_shadow.FEATURE_NAMES, end_sec),
                    "physical_reasons": ["input_amplitude"] if rejected else [],
                    "strict_reasons": ["input_amplitude"] if rejected else (["input_eog"] if strict_only else [])}

        windows = [window(6), window(8), window(10, rejected=True), window(12, strict_only=True),
                   window(14), window(16)]
        with patch.object(focus_tcn, "score_sequence", return_value=.42) as score:
            result = _tcn_score(object(), windows, 18 * 250 - 1, 1020, 1000)
        self.assertEqual(result["mw_score"], .42)
        self.assertEqual(result["frame_mask"], "101110")
        self.assertEqual(result["physical_window_rejections"], {"input_amplitude": 1, "missing_window": 1})
        self.assertEqual(result["strict_qc_status"], "failed")
        _, features, mask = score.call_args.args
        np.testing.assert_array_equal(features[:, 0], [8, 0, 12, 14, 16, 0])
        np.testing.assert_array_equal(mask, [True, False, True, True, True, False])

    def test_recovery_mode_scores_current_probe_and_keeps_late_samples_excluded(self):
        self.source.update(timestamp_policy=BIOMULTI_LSL_TIMING, research_input_policy=RECOVERY_INPUT_POLICY)
        store = self.session()
        store.event("phase_start", {"phase": "sart_1"})
        for end in range(50, 5501, 50):
            points = np.arange(end - 50, end)
            values = synthetic_samples(points)
            values[:, 1] *= -.8
            self.clock.advance(.2)
            store.append(values, 1000 + points / 250, offset=0.)
        preflight = store.predictions.recovery_preflight(self.clock.now(), store.sample_count)
        self.assertGreaterEqual(preflight["windows_usable"], 4, preflight)
        self.assertNotIn("mw_score", preflight)
        store.event("probe_onset", {"phase": "sart_1", "probe_id": "fresh"})
        self.clock.advance(20.)
        store.event("probe_onset", {"phase": "sart_1", "probe_id": "before_late_arrival"})
        late = np.arange(5500, 10500)
        store.append(synthetic_samples(late), 1000 + late / 250, offset=0.)
        store.close("aborted")
        fresh, late = [record for _, record in self.read(store)]
        for key in ("tcn_250", "shadow_250"):
            self.assertEqual(fresh[key]["input_policy"], RECOVERY_INPUT_POLICY)
            self.assertEqual(fresh[key]["status"], "scored", fresh[key])
            self.assertEqual(late[key]["status"], "unavailable", late[key])

    def test_tcn_failure_does_not_change_existing_scores(self):
        for missing, reason in ((True, "model_unavailable"), (False, "prediction_error")):
            with self.subTest(missing=missing):
                self.clock = FakeClock()
                with patch.object(focus_tcn, "load_candidate", side_effect=FileNotFoundError if missing else None,
                                  wraps=focus_tcn.load_candidate):
                    store = self.session()
                store.event("phase_start", {"phase": "sart_1"})
                points = np.arange(20 * 250)
                values = synthetic_samples(points)
                values[:, 1] *= -.8
                self.clock.advance(20)
                store.append(values, 1000 + points / 250, offset=0.)
                with patch.object(focus_tcn, "score_sequence", side_effect=ValueError):
                    store.event("probe_onset", {"phase": "sart_1", "probe_id": "tcn-failure"})
                    store.close("aborted")
                _, prediction = self.read(store)[0]
                self.assertEqual(prediction["shadow_250"]["status"], "scored")
                self.assertEqual(prediction["tcn_250"]["reason"], reason)
                self.assertIsNone(prediction["tcn_250"]["mw_score"])

    def test_mismatched_rate_and_shadow_failure_preserve_legacy_record(self):
        for corrupt, expected in ((False, "rate_mismatch"), (True, "model_invalid")):
            with self.subTest(corrupt=corrupt):
                self.source["srate"] = 256
                with patch.object(attlapse_shadow, "load_candidates",
                                  side_effect=ValueError if corrupt else None,
                                  wraps=attlapse_shadow.load_candidates):
                    store = self.session()
                store.event("probe_onset", {"phase": "sart_1", "probe_id": "x"})
                store.close()
                _, prediction = self.read(store)[0]
                self.assertEqual(prediction["shadow_250"]["reason"], expected)
                self.assertNotEqual(prediction["reason"], expected)
