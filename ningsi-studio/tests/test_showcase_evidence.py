"""Field evidence is bounded and anonymous; declarations never certify a device."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ningsi_studio.domain.showcase_evidence import (
    MAX_EVIDENCE_BYTES, PROFILE_FILE, PROFILE_SCHEMA,
    build_evidence, load_device_profile, save_device_profile,
)


class ShowcaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def complete(self, **overrides):
        return {"windows_total": 573, "windows_usable": 3, "baseline_valid": False,
                "training_readiness": {"formal_probes": 15, "paired": 0},
                "train_eligible": 0, "test_eligible": 0, "session_id": "private-session",
                "participant": "private-person", "samples": [[987654321]], **overrides}

    def test_missing_evidence_never_becomes_a_pass(self):
        result = build_evidence(self.root)
        self.assertEqual(result["overall_status"], "pending_field_validation")
        self.assertEqual(result["summary"]["target_device_accepted"], 0)
        self.assertTrue(all(not r["available"] for r in result["evidence"]))
        self.assertTrue(all(r["software_status"] == "implemented" for r in result["requirements"]))
        self.assertEqual(result["evidence_origin"], "workspace_summaries")

    def test_frozen_application_uses_a_clearly_labelled_sanitized_snapshot(self):
        with patch("ningsi_studio.domain.showcase_evidence.sys.frozen", True, create=True):
            frozen = build_evidence()
        self.assertEqual(frozen["evidence_origin"], "bundled_development_snapshot")
        self.assertTrue(frozen["snapshot_at"])
        self.assertIn("不代表当前设备", frozen["snapshot_note"])
        self.assertEqual(frozen["overall_status"], "pending_field_validation")
        self.assertTrue(any(row["available"] for row in frozen["evidence"]))
        text = json.dumps(frozen, ensure_ascii=False)
        self.assertNotIn("P001", text)
        self.assertNotIn("session_id", text)
        self.assertNotIn("samples", text)

    def test_explicit_workspace_never_falls_back_even_inside_a_frozen_application(self):
        with patch("ningsi_studio.domain.showcase_evidence.sys.frozen", True, create=True):
            result = build_evidence(self.root)
        self.assertTrue(all(row["availability"] == "missing" for row in result["evidence"]))
        self.assertIsNone(result["snapshot_at"])
        self.assertEqual(result["evidence_origin"], "workspace_summaries")

    def test_invalid_source_evidence_is_not_replaced_with_a_good_old_snapshot(self):
        path = self.root / "var/preprocessing-0.4.0/P001-export-check.json"
        path.parent.mkdir(parents=True)
        path.write_text("{broken", encoding="utf-8")
        from ningsi_studio.domain import showcase_evidence
        fake_module = self.root / "ningsi-studio/src/ningsi_studio/domain/showcase_evidence.py"
        fake_module.parent.mkdir(parents=True)
        with patch.object(showcase_evidence, "__file__", str(fake_module)), patch.object(
                showcase_evidence, "_bundled_snapshot") as bundled:
            result = build_evidence()
        bundled.assert_not_called()
        self.assertEqual(result["evidence_origin"], "workspace_summaries")
        complete = next(row for row in result["evidence"] if row["id"] == "complete_capture")
        self.assertEqual(complete["availability"], "invalid")

    def test_real_summary_counts_and_failure_are_preserved_without_private_content(self):
        self.write("var/preprocessing-0.4.0/P001-export-check.json", self.complete())
        self.write("var/biomulti-qc-0.3.4/live-device-check.json",
                   {"windows_total": 13, "all_quality_pass": 9, "timing_pass": 13, "seconds_acquired": 29.72})
        self.write("var/preprocessing-0.4.0/live-review.json",
                   {"windows_total": 15, "windows_usable": 0, "channels": 2,
                    "duration_seconds_by_sample_count": 32.16, "srate": 250,
                    "windows_with_valid_history_and_clock": 11,
                    "windows": [{"private_samples": [987654321]}]})
        result = build_evidence(self.root)
        records = {row["id"]: row for row in result["evidence"]}
        self.assertEqual(records["short_capture"]["result"], "partial")
        self.assertEqual(records["short_capture"]["metrics"]["windows_usable"], 9)
        self.assertEqual(records["complete_capture"]["result"], "not_accepted")
        self.assertEqual(records["complete_capture"]["metrics"]["windows_total"], 573)
        self.assertEqual(records["complete_capture"]["metrics"]["paired"], 0)
        self.assertFalse(records["complete_capture"]["metrics"]["baseline_valid"])
        self.assertEqual(records["recent_capture"]["metrics"]["windows_usable"], 0)
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        for private in ("private-session", "private-person", "987654321", "P001", str(self.root)):
            self.assertNotIn(private, encoded)

    def test_good_future_record_does_not_certify_hardware_or_general_model_effectiveness(self):
        self.write("var/preprocessing-0.4.0/P001-export-check.json",
                   self.complete(windows_usable=573, baseline_valid=True,
                                 training_readiness={"formal_probes": 15, "paired": 15}))
        result = build_evidence(self.root)
        complete = next(r for r in result["evidence"] if r["id"] == "complete_capture")
        self.assertEqual(complete["result"], "partial")
        self.assertEqual(result["summary"]["target_device_accepted"], 0)
        self.assertFalse(result["research"]["target_device_validated"])

    def test_malformed_oversized_and_nonfinite_summaries_are_unavailable(self):
        path = self.write("var/preprocessing-0.4.0/P001-export-check.json", self.complete())
        for body in ("[]", "{broken", '{"windows_total": NaN}', " " * (MAX_EVIDENCE_BYTES + 1)):
            with self.subTest(body=body[:20]):
                path.write_text(body, encoding="utf-8")
                record = next(r for r in build_evidence(self.root)["evidence"] if r["id"] == "complete_capture")
                self.assertFalse(record["available"])
                self.assertEqual(record["availability"], "invalid")

    def test_invalid_count_types_are_not_published(self):
        for overrides in ({"windows_total": True}, {"windows_usable": 574}, {"baseline_valid": "true"},
                          {"training_readiness": {"formal_probes": 2, "paired": 3}}):
            with self.subTest(overrides=overrides):
                self.write("var/preprocessing-0.4.0/P001-export-check.json", self.complete(**overrides))
                record = next(r for r in build_evidence(self.root)["evidence"] if r["id"] == "complete_capture")
                self.assertFalse(record["available"])

    def test_recorded_software_results_are_marked_as_snapshot(self):
        path = self.root / "docs/VALIDATION_PLAN.md"
        path.parent.mkdir(parents=True)
        path.write_text("共 470 项软件测试通过。", encoding="utf-8")
        validation = build_evidence(self.root)["software_validation"]
        self.assertEqual(validation["tests_total"], 470)
        self.assertEqual(validation["kind"], "recorded_development_snapshot")
        self.assertIn("不重新执行", validation["note"])

    def test_research_report_is_summarized_without_returning_its_text(self):
        path = self.root / "var/calibration-dose-20261116/RESULTS.md"
        path.parent.mkdir(parents=True)
        path.write_text("固定后测 SAM40 FACED Monash private-person 987654321", encoding="utf-8")
        result = build_evidence(self.root)
        self.assertTrue(result["research"]["available"])
        self.assertEqual(len(result["research"]["sha256"]), 64)
        self.assertFalse(result["research"]["target_device_validated"])
        text = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("private-person", text)
        self.assertNotIn("987654321", text)


class ShowcaseDeviceDeclarationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def profile(self, **changes):
        return {"manufacturer": "设备厂家", "model": "Research EEG",
                "electrodes": "Fp1, Fp2", "reference": "右耳参考",
                "srate": 250, "source_unit": "uV", "scale_to_uv": 1,
                "unit_basis": "厂商文档第 8 页", "document_reference": "设备说明书与赛事允许材料",
                "notes": "现场记录", **changes}

    def test_complete_declaration_roundtrips_but_never_claims_compliance(self):
        result = save_device_profile(self.root, self.profile())
        self.assertEqual(result, load_device_profile(self.root))
        self.assertEqual(result["missing_fields"], [])
        self.assertEqual(result["status"], "pending_verification")
        self.assertEqual(result["compliance_status"], "not_verified")
        self.assertEqual(result["declaration_status"], "operator_declaration")
        self.assertEqual(list((self.root / "showcase").glob("*.tmp")), [])

    def test_incomplete_form_can_be_saved_and_returns_specific_missing_fields(self):
        result = save_device_profile(self.root, {"model": "型号待核对"})
        missing = {item["key"] for item in result["missing_fields"]}
        self.assertIn("reference", missing)
        self.assertIn("unit_basis", missing)
        self.assertIn("document_reference", missing)
        self.assertNotIn("model", missing)
        self.assertTrue(result["saved"])

    def test_material_alias_is_canonicalized_without_opening_document_paths(self):
        result = save_device_profile(self.root, {"materials": "C:\\private\\device.pdf"})
        self.assertEqual(result["profile"]["document_reference"], "C:\\private\\device.pdf")
        self.assertNotIn("materials", result["profile"])

    def test_invalid_values_are_rejected_without_overwriting_previous_declaration(self):
        original = save_device_profile(self.root, self.profile())
        cases = [None, [], {"compliance_status": "passed"}, {"srate": True},
                 {"srate": float("inf")}, {"srate": -1}, {"srate": 20000},
                 {"srate": 10**1000}, {"scale_to_uv": 10**1000},
                 {"source_unit": "unknown"}, {"scale_to_uv": float("nan")},
                 {"source_unit": "mV", "scale_to_uv": 1}, {"model": "x" * 161},
                 {"reference": "bad\x00reference"}, {"materials": "one", "document_reference": "two"}]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    save_device_profile(self.root, payload)
                self.assertEqual(load_device_profile(self.root), original)

    def test_corrupt_file_returns_an_explicit_empty_state(self):
        path = self.root / PROFILE_FILE
        path.parent.mkdir(parents=True)
        for value in ("{broken", json.dumps({"schema": PROFILE_SCHEMA, "profile": {"srate": "250"}})):
            path.write_text(value, encoding="utf-8")
            result = load_device_profile(self.root)
            self.assertFalse(result["saved"])
            self.assertIn("read_error", result)
            self.assertEqual(result["compliance_status"], "not_verified")

    def test_atomic_failure_preserves_the_previous_file_and_cleans_temporary(self):
        original = save_device_profile(self.root, self.profile())
        with patch("ningsi_studio.domain.showcase_evidence.os.replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                save_device_profile(self.root, self.profile(model="New model"))
        self.assertEqual(load_device_profile(self.root), original)
        self.assertEqual(list((self.root / "showcase").glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
