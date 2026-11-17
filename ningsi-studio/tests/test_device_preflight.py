"""Device isolation, exact live-model policy reuse and read-only preflight contracts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ningsi.acquisition.timing import BIOMULTI_REC_TIMING
from ningsi.models import human, human_data
from ningsi_studio.api.routes import build_router
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.domain import device_profiles
from ningsi_studio.domain.preflight import evaluate_preflight
from ningsi_studio.http.router import Request, ValidationError
from ningsi_studio.settings import Settings

from .helpers import StudioTestCase
from .human_fixtures import fixture_files


DEVICE = "lsl:FixtureEEG"
DESCRIPTOR = {"name": "FixtureEEG", "kind": "eeg", "stream_type": "EEG", "supported": True,
              "source_id": "qa-software-fixture", "channel_count": 1, "nominal_srate": 100.,
              "channel_labels": ["Fp1"], "declared_units": ["uV"]}


class DeviceProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_profiles_are_isolated_and_never_claim_certification(self):
        first = device_profiles.save_device_profile(self.root, DEVICE, {"model": "QA only", "source_unit": "uV", "scale_to_uv": 1})
        self.assertEqual(first, device_profiles.load_device_profile(self.root, DEVICE))
        self.assertFalse(device_profiles.load_device_profile(self.root, "lsl:Other EEG")["saved"])
        self.assertEqual(first["compliance_status"], "not_verified")
        self.assertEqual(first["declaration_status"], "operator_declaration")
        self.assertTrue(first["missing_fields"])

    def test_caller_paths_are_text_only_and_source_names_are_hashed(self):
        key = "lsl:../../not-a-path"
        result = device_profiles.save_device_profile(self.root, key, {"document_reference": "C:/private/not-read.pdf"})
        self.assertEqual(result["profile"]["document_reference"], "C:/private/not-read.pdf")
        self.assertEqual(len(list((self.root / "device_profiles").glob("*.json"))), 1)
        self.assertFalse((self.root / "not-a-path").exists())
        self.assertFalse(device_profiles.load_device_profile(self.root, DEVICE)["saved"])

    def test_invalid_save_and_failed_replace_leave_previous_record_untouched(self):
        original = device_profiles.save_device_profile(self.root, DEVICE, {"model": "Existing"})
        for change in ({"srate": True}, {"scale_to_uv": float("nan")}, {"srate": 10**1000},
                       {"source_unit": "mV", "scale_to_uv": 1}, {"model": "x" * 161}, {"source_path": "secret"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                device_profiles.save_device_profile(self.root, DEVICE, change)
        with patch("ningsi_studio.domain.device_profiles.os.replace", side_effect=OSError("fixture failure")):
            with self.assertRaises(OSError):
                device_profiles.save_device_profile(self.root, DEVICE, {"model": "New"})
        self.assertEqual(device_profiles.load_device_profile(self.root, DEVICE), original)
        self.assertEqual(list((self.root / "device_profiles").glob("*.tmp")), [])

    def test_corruption_and_legacy_unscoped_record_are_not_applied(self):
        legacy = self.root / "showcase/device_profile.json"
        legacy.parent.mkdir()
        legacy.write_text('{"profile":{"model":"Legacy"}}')
        self.assertFalse(device_profiles.load_device_profile(self.root, DEVICE)["saved"])
        device_profiles.save_device_profile(self.root, DEVICE, {"model": "Current"})
        path = next((self.root / "device_profiles").glob("*.json"))
        record = json.loads(path.read_text())
        record["device"] = "lsl:Another EEG"
        path.write_text(json.dumps(record))
        result = device_profiles.load_device_profile(self.root, DEVICE)
        self.assertFalse(result["saved"])
        self.assertIn("read_error", result)


class PreflightPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = human.train_development(human_data.import_files(fixture_files()))

    def check(self, descriptor=None, *, model=None, protocol="short", **payload):
        return evaluate_preflight({"device": DEVICE, "protocol": protocol, **payload},
                                  descriptors=[descriptor or DESCRIPTOR], active_model=model)

    def test_index_only_configuration_pass_still_leaves_quality_and_baseline_pending(self):
        result = self.check()
        self.assertTrue(result["ready"])
        self.assertEqual(result["scoring"]["attention_model"]["status"], "none")
        self.assertEqual(result["scoring"]["relative_indices"]["feedback_target"], "relative_focus_index")
        self.assertEqual(next(row for row in result["checks"] if row["key"] == "quality")["status"], "pending")

    def test_unknown_units_need_explicit_confirmation_and_adc_requires_positive_factor(self):
        descriptor = {**DESCRIPTOR, "declared_units": [""]}
        self.assertFalse(self.check(descriptor)["ready"])
        self.assertFalse(self.check(descriptor, eeg_calibration={"source_unit": "ADC"})["ready"])
        self.assertFalse(self.check(descriptor, eeg_calibration={"source_unit": "ADC", "scale_to_uv": -1})["ready"])
        result = self.check(descriptor, eeg_calibration={"source_unit": "ADC", "scale_to_uv": .01})
        self.assertTrue(result["ready"])
        self.assertEqual(result["calibration"]["scale_to_uv"], .01)
        # A saved declaration is a suggestion, never a silent override of a live stream.
        saved = {"saved": True, "profile": {"source_unit": "uV", "scale_to_uv": 1}, "missing_fields": []}
        result = evaluate_preflight({"device": DEVICE}, descriptors=[descriptor], device_profile=saved)
        self.assertFalse(result["ready"])

    def test_proven_unit_conflict_blocks_even_without_a_model(self):
        result = self.check(eeg_calibration={"source_unit": "mV"})
        self.assertFalse(result["ready"])
        self.assertIsNone(result["calibration"])

    def test_low_sampling_rates_cannot_pass_the_fixed_45hz_processing_chain(self):
        for rate in (50, 90):
            with self.subTest(rate=rate):
                result = self.check({**DESCRIPTOR, "nominal_srate": rate})
                self.assertFalse(result["ready"])
                device = next(row for row in result["checks"] if row["key"] == "device")
                self.assertEqual(device["status"], "fail")
                self.assertIn("45 Hz", device["message"])
        self.assertTrue(self.check({**DESCRIPTOR, "nominal_srate": 100})["ready"])

    def test_simulation_absent_non_eeg_and_ambiguous_sources_do_not_pass(self):
        rows = [([], DEVICE), ([{**DESCRIPTOR, "kind": "metric"}], DEVICE),
                ([DESCRIPTOR, DESCRIPTOR], DEVICE), ([], "sim-bsense"),
                ([{**DESCRIPTOR, "source_id": "synthetic-eeg"}], DEVICE)]
        for descriptors, key in rows:
            with self.subTest(key=key, descriptors=descriptors):
                result = evaluate_preflight({"device": key}, descriptors=descriptors, active_model=self.model)
                self.assertFalse(result["ready"])

    def test_model_compatibility_uses_the_live_predictor_and_retains_original_feedback(self):
        result = self.check(model=self.model)
        model = result["scoring"]["attention_model"]
        self.assertTrue(result["ready"])
        self.assertEqual(model["status"], "compatible")
        self.assertEqual(model["model_id"], self.model["model_id"])
        self.assertIn("基线", model["message"])
        self.assertEqual(result["scoring"]["relative_indices"]["feedback_target"], "relative_focus_index")
        for change in ({"channel_count": 2}, {"channel_labels": ["Fp2"]}, {"nominal_srate": 250}, {"source_id": "different"}):
            result = self.check({**DESCRIPTOR, **change}, model=self.model)
            self.assertEqual(result["scoring"]["attention_model"]["status"], "incompatible")

    def test_model_failure_is_visible_but_does_not_silently_replace_or_block_index_assessment(self):
        mismatched = self.check({**DESCRIPTOR, "channel_labels": ["Fp2"]}, model=self.model)
        self.assertTrue(mismatched["ready"])
        self.assertEqual(mismatched["scoring"]["attention_model"]["reason"], "device_channels_units_mismatch")
        bad = copy.deepcopy(self.model)
        bad["engine_fingerprint"] = {"fixture": "mismatched"}
        bad["model_id"] = human.digest({k: v for k, v in bad.items() if k != "model_id"})
        self.assertEqual(self.check(model=bad)["scoring"]["attention_model"]["reason"], "feature_engine_mismatch")

    def test_rec_only_models_and_ten_minute_scope_are_reported_before_start(self):
        model = copy.deepcopy(self.model)
        for profile in model["source_profiles"]:
            profile["timestamp_policy"] = BIOMULTI_REC_TIMING
        model["model_id"] = human.digest({k: v for k, v in model.items() if k != "model_id"})
        result = self.check(model=model)
        self.assertEqual(result["scoring"]["attention_model"]["reason"], "native_rec_clock_required")
        ten = self.check(model=self.model, protocol="ten_min")
        self.assertEqual(ten["scoring"]["attention_model"]["scope_status"], "outside_validated_task")
        self.assertTrue(any(row["key"] == "model_scope" for row in ten["checks"]))


class PreflightApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Settings(data_dir=Path(temporary.name))
        store.initialize(self.settings.db_path)
        self.router = build_router(self.settings)
        self.addCleanup(self.router.human_models.close)

    def call(self, method, path, payload=None, query=None):
        response = self.router.dispatch(Request(method, path, query or {}, {},
            json.dumps(payload).encode() if payload is not None else b""))
        return json.loads(response.body)

    def test_product_profile_routes_roundtrip_and_reject_invalid_shapes(self):
        saved = self.call("POST", "/api/device-profile", {"device": DEVICE, "profile": {"model": "QA"}})
        self.assertEqual(saved, self.call("GET", "/api/device-profile", query={"device": DEVICE}))
        for body in ([], {}, {"device": "../../private", "profile": {}}, {"device": DEVICE, "profile": {}, "path": "x"}):
            with self.subTest(body=body), self.assertRaises(ValidationError):
                self.call("POST", "/api/device-profile", body)

    def test_api_probe_uses_server_metadata_and_never_creates_a_session(self):
        with patch("ningsi_studio.acquisition.lsl.probe_streams", return_value=[DESCRIPTOR]) as probe:
            result = self.call("POST", "/api/sessions/preflight", {"device": DEVICE})
        self.assertTrue(result["ready"])
        probe.assert_called_once_with(timeout=.4)
        with store.read_only(self.settings.db_path) as conn:
            self.assertEqual(repo.overview(conn)["sessions"], 0)
        with patch("ningsi_studio.acquisition.lsl.probe_streams", side_effect=RuntimeError("fixture scan failure")):
            result = self.call("POST", "/api/sessions/preflight", {"device": DEVICE})
        self.assertFalse(result["ready"])
        for body in ([], {"device": DEVICE, "descriptor": DESCRIPTOR}, {"protocol": ["bad"]}):
            with self.subTest(body=body), self.assertRaises(ValidationError):
                self.call("POST", "/api/sessions/preflight", body)

    def test_corrupt_active_model_is_reported_without_hiding_index_assessment(self):
        with patch("ningsi_studio.acquisition.lsl.probe_streams", return_value=[DESCRIPTOR]), \
             patch("ningsi_studio.api.routes.load_active", side_effect=ValueError("bad frozen model")):
            result = self.call("POST", "/api/sessions/preflight", {"device": DEVICE})
        self.assertTrue(result["ready"])
        self.assertEqual(result["scoring"]["attention_model"]["status"], "unavailable")


class PreflightAuthTests(StudioTestCase):
    port_base = 0

    @classmethod
    def make_settings(cls, data_dir):
        return Settings(data_dir=Path(data_dir), port=0, api_token="software-fixture-token")

    def setUp(self):
        pass

    def test_new_api_routes_obey_existing_authentication_before_processing(self):
        self.assert_error(self.get("/api/device-profile?device=lsl:FixtureEEG"), 401, "unauthorized")
        self.assert_error(self.post("/api/device-profile", {"device": DEVICE, "profile": {}}), 401, "unauthorized")
        self.assert_error(self.post("/api/sessions/preflight", {"device": DEVICE}), 401, "unauthorized")
        response = self.post("/api/sessions/preflight", {"device": "sim-bsense"},
                             headers={"X-API-Token": "software-fixture-token"})
        self.assertFalse(self.json_body(response)["ready"])
