"""Manual reference data must never become verified live acquisition settings."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ningsi_studio.api.routes import build_router
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.domain import device_guides, device_profiles
from ningsi_studio.domain.preflight import evaluate_preflight
from ningsi_studio.http.router import Request
from ningsi_studio.settings import Settings


class DeviceGuideTests(unittest.TestCase):
    def test_manual_catalogue_has_detached_provenance_and_no_private_paths(self):
        guide = device_guides.device_guides()[0]
        self.assertEqual(guide["schema"], "ningsi-device-guide-v1")
        self.assertEqual(guide["id"], "bsense-r")
        source = guide["source"]
        self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(source["page_count"], 25)
        self.assertEqual(source["revision"], "未标明")
        for section in ("specs", "steps", "quality_notes"):
            for row in guide[section]:
                self.assertTrue(row["pages"])
                self.assertTrue(all(type(page) is int and 1 <= page <= source["page_count"]
                                    for page in row["pages"]))
        serialized = json.dumps(guide, ensure_ascii=False)
        for private in ("WeChat", "xwechat_files", "wxid_", "D:/", "D:\\"):
            self.assertNotIn(private, serialized)
        guide["source"]["sha256"] = "modified"
        guide["specs"][0]["pages"].clear()
        guide["declaration_seed"]["source_unit"] = "uV"
        again = device_guides.device_guides()[0]
        self.assertNotEqual(again["source"]["sha256"], "modified")
        self.assertTrue(again["specs"][0]["pages"])
        self.assertNotIn("source_unit", again["declaration_seed"])

    def test_explicitly_saving_manual_seed_leaves_unknown_settings_and_qc_pending(self):
        seed = device_guides.device_guides()[0]["declaration_seed"]
        self.assertEqual(set(seed), {"model", "document_reference", "notes"})
        with tempfile.TemporaryDirectory() as folder:
            device = "lsl:FixtureEEG"
            saved = device_profiles.save_device_profile(folder, device, seed)
            profile = saved["profile"]
            for key in ("manufacturer", "electrodes", "reference", "source_unit", "unit_basis"):
                self.assertEqual(profile[key], "")
            for key in ("srate", "scale_to_uv"):
                self.assertIsNone(profile[key])
            self.assertEqual(saved["status"], "pending_verification")
            self.assertEqual(saved["compliance_status"], "not_verified")
            self.assertTrue(saved["missing_fields"])
            self.assertFalse(device_profiles.load_device_profile(folder, "lsl:OtherEEG")["saved"])
            descriptor = {"name": "FixtureEEG", "kind": "eeg", "supported": True,
                          "channel_count": 2, "nominal_srate": 250,
                          "channel_labels": ["Ch1", "Ch2"], "declared_units": ["", ""]}
            checked = evaluate_preflight({"device": device}, descriptors=[descriptor], device_profile=saved)
            self.assertFalse(checked["ready"], "A manual's CSV units must not confirm this live LSL stream")
            self.assertEqual(next(row for row in checked["checks"] if row["key"] == "units")["status"], "fail")
            self.assertEqual(next(row for row in checked["checks"] if row["key"] == "quality")["status"], "pending")
            confirmed = evaluate_preflight({"device": device, "eeg_calibration": {"source_unit": "uV"}},
                                           descriptors=[descriptor], device_profile=saved)
            self.assertTrue(confirmed["ready"], "Existing explicit per-session unit confirmation remains available")
            self.assertEqual(next(row for row in confirmed["checks"] if row["key"] == "quality")["status"], "pending")


class DeviceGuideConfigTests(unittest.TestCase):
    def test_config_publishes_reference_without_scanning_or_writing_profiles(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(data_dir=Path(folder))
            store.initialize(settings.db_path)
            router = build_router(settings)
            self.addCleanup(router.human_models.close)
            before = {str(path.relative_to(folder)) for path in Path(folder).rglob("*")}
            with patch("ningsi_studio.acquisition.lsl.probe_streams", side_effect=AssertionError("unexpected scan")), \
                 patch.object(device_profiles, "save_device_profile", side_effect=AssertionError("unexpected save")):
                response = router.dispatch(Request("GET", "/api/config", {}, {}, b""))
            result = json.loads(response.body)
            self.assertEqual(result["device_guides"], device_guides.device_guides())
            after = {str(path.relative_to(folder)) for path in Path(folder).rglob("*")}
            self.assertEqual(before, after)
            self.assertFalse((settings.data_dir / "device_profiles").exists())
            with store.read_only(settings.db_path) as conn:
                self.assertEqual(repo.overview(conn)["sessions"], 0)


if __name__ == "__main__":
    unittest.main()
