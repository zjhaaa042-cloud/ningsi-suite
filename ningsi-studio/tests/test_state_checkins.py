import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ningsi_studio.api.routes import build_router
from ningsi_studio.core.runtime import SessionRuntime
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.domain import state_checkins, eeg_units
from ningsi_studio.http.router import Conflict, Request, ValidationError
from ningsi_studio.settings import Settings


class StateCheckinTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.settings = Settings(data_dir=Path(temp.name))
        self.path = store.initialize(self.settings.db_path)
        self.router = build_router(self.settings)
        self.addCleanup(self.router.human_models.close)
        with store.connect(self.path) as conn:
            self.person = repo.create_subject(conn, "p01")
        self.uuid = self.make_session()
        self.runtime = Mock(alive=True, scale=1, source=SimpleNamespace(is_real=True))
        self.runtime.elapsed_now.return_value = 5.2
        self.runtime.checkin_context.return_value = None

    def make_session(self):
        with store.connect(self.path) as conn:
            row = repo.create_session(conn, self.person)
            repo.update_session(conn, row["uuid"], phase="monitor")
            return row["uuid"]

    def dispatch(self, verb, path, body=None, query=None):
        req = Request(verb, path, query or {}, {}, json.dumps(body or {}).encode())
        with patch.object(self.router.manager, "get", return_value=self.runtime):
            return json.loads(self.router.dispatch(req).body)

    def submit(self, request_id, stress, valence, uuid=None):
        return self.dispatch("POST", f"/api/sessions/{uuid or self.uuid}/state-checkins",
                             {"request_id": f"{request_id:032x}", "stress": stress, "valence": valence})

    def finish(self, uuid):
        with store.connect(self.path) as conn:
            repo.update_session(conn, uuid, phase="done", status="done")

    def test_ratings_series_alerts_retries_and_export(self):
        result = self.submit(1, 8, 0)
        self.assertTrue(result["high_stress"])
        self.assertEqual(result["record"]["valence"], 0)
        self.assertEqual(self.submit(1, 8, 0), result)
        with self.assertRaises(Conflict):
            self.submit(1, 4, 0)
        self.assertFalse(self.submit(2, None, 5)["high_stress"])
        self.assertEqual(self.runtime.bus.publish.call_count, 1)
        self.finish(self.uuid)
        self.assertEqual(self.submit(1, 8, 0), result)
        with self.assertRaises(Conflict):
            self.submit(3, 7, 3)
        with store.read_only(self.path) as conn:
            session = repo.get_session(conn, self.uuid)
            payload = state_checkins.report_data(conn, session["id"], [])
            self.assertEqual([r["stress"] for r in payload["records"]], [8, None])
            self.assertEqual(payload["stress_mean"], 8)
            self.assertEqual(len(repo.list_alerts(conn, session["id"])), 1)
            self.assertTrue(any("愉悦程度 0/10" in s for s in payload["summary_lines"]))

    def test_week_month_exclude_sim_and_missing_and_weight_visits_equally(self):
        self.submit(1, 8, 7)
        self.submit(2, 10, None)
        self.finish(self.uuid)
        second = self.make_session()
        self.submit(3, 1, None, second)
        self.finish(second)
        self.runtime.source.is_real = False
        demo = self.make_session()
        self.submit(4, 10, 3, demo)
        self.finish(demo)
        self.finish(self.make_session())  # legacy session with no check-ins
        for period in ("week", "month"):
            for path in ("/api/reports/trend", f"/api/sessions/{self.uuid}/trend"):
                result = self.dispatch("GET", path, query={"period": period, "field": "stress"})
                self.assertEqual(result["sessions"], 2)
                self.assertEqual(result["points"][0]["mean"], 5)
                self.assertEqual(result["points"][0]["n"], 2)

    def test_invalid_answers_and_timed_tasks_are_rejected(self):
        for stress, valence in ((None, None), (-1, 0), (11, None), (False, 3), (6.9, 1)):
            with self.assertRaises(ValidationError):
                self.submit(1, stress, valence)
        with store.connect(self.path) as conn:
            repo.update_session(conn, self.uuid, phase="sart")
        with self.assertRaises(Conflict):
            self.submit(1, 7, None)


class AffectRuntimeTests(unittest.TestCase):
    def test_prediction_requires_known_units_accepted_window_and_fresh_context(self):
        runtime = SessionRuntime("a" * 32)
        runtime.session = {"time_scale": 1}
        descriptor = {"channel_labels": ["Fp1", "VEOG", "Fp2"], "channel_count": 3,
                      "declared_units": ["uV", "uV", ""]}
        runtime.source = SimpleNamespace(is_real=True, engine=Mock(), srate=250)
        runtime.source.engine.descriptor_dict.return_value = descriptor
        step = SimpleNamespace(window=SimpleNamespace(channels=2, srate=250, rel={"theta": .2}),
                               index=1, usable=True)
        with patch("ningsi.models.affect.predict_window") as predict:
            result = runtime._affect_observation(step, "monitor")
            self.assertEqual(result["stress"]["reason"], "unconfirmed_input_units")
            predict.assert_not_called()
            descriptor["declared_units"][-1] = "mV"
            runtime._affect_observation(step, "monitor")
            predict.assert_called_once_with(step.window, ["Fp1", "Fp2"], 250)
            self.assertIsNotNone(runtime.checkin_context("monitor"))
            self.assertIsNone(runtime.checkin_context("scales"))
            runtime.affect_snapshot["elapsed_sec"] -= 5
            self.assertIsNone(runtime.current_affect())
            self.assertIsNone(runtime.checkin_context("monitor"))
            step.usable = False
            self.assertEqual(runtime._affect_observation(step, "monitor")["stress"]["reason"], "quality_rejected")

    def test_calibrated_eeg_does_not_inherit_auxiliary_units(self):
        descriptor = {"channel_count": 3, "channel_labels": ["Fp1", "Fp2", "ACCX"],
                      "declared_units": ["ADC", "", "g"]}
        calibration = eeg_units.confirmation({"source_unit": "ADC", "scale_to_uv": .32})
        eeg_units.verify_descriptor(calibration, descriptor)
        descriptor["declared_units"][1] = "uV"
        with self.assertRaises(ValueError):
            eeg_units.verify_descriptor(calibration, descriptor)
        for invalid in ({"source_unit": "ADC"}, {"source_unit": "ADC", "scale_to_uv": -1},
                        {"source_unit": "uV", "scale_to_uv": 1000}):
            with self.assertRaises(ValueError):
                eeg_units.confirmation(invalid)

    def test_session_unit_override_reaches_runtime_and_conflicting_source_is_rejected(self):
        # No physical amplifier: run the normal Studio startup route with a declared fixture.
        with tempfile.TemporaryDirectory() as root:
            settings = Settings(data_dir=Path(root))
            db = store.initialize(settings.db_path)
            calibration = eeg_units.confirmation({"source_unit": "mV"})
            with store.connect(db) as conn:
                person = repo.create_subject(conn, "p01")
                row = repo.create_session(conn, person, device="lsl:fixture", eeg_calibration=calibration)
            source = SimpleNamespace(kind="lsl", engine=Mock())
            descriptor = {"channel_labels":["Fp1", "Fp2"], "channel_count":2, "declared_units":["",""]}
            source.engine.descriptor_dict.return_value = descriptor
            with patch("ningsi_studio.core.runtime.build_source", return_value=source):
                runtime = SessionRuntime(row["uuid"], settings=settings)
                runtime._load()
                self.assertEqual(runtime.eeg_calibration, calibration)
                self.assertEqual(source.engine.operator_eeg_scale_to_uv, 1000)
                descriptor["declared_units"][1] = "uV"
                with self.assertRaises(ValueError):
                    SessionRuntime(row["uuid"], settings=settings)._load()
