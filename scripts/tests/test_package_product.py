"""Packaging guards use only disposable synthetic files, never real releases."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package_product import build_members, output_paths, parse_suite, read_verification, validate_label
import package_product as product


class ProductPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ningsi-package-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, content="fixture"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_release_label_rejects_traversal_and_windows_reserved_names(self):
        for name in ("", ".", "..", "../old", "release/file", "release\\file", "CON", "LPT1.txt", "product.", "a b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_label(name)
        self.assertEqual(validate_label("ningsi-product-1.2.3-20261117"), "ningsi-product-1.2.3-20261117")

    def test_existing_release_or_archive_is_never_overwritten(self):
        for ending in ("", ".zip", ".zip.sha256"):
            with self.subTest(ending=ending):
                name = "release" + str(len(ending))
                self.write(name + ending)
                with self.assertRaises(FileExistsError):
                    output_paths(name, self.root)

    def test_safe_assets_and_dependency_license_are_kept_runtime_files_excluded(self):
        allowed = ("_internal/web/index.html", "_internal/base_library.zip", "_internal/pylsl/liblsl.dll",
                   "_internal/numpy.dist-info/licenses/LICENSE.txt")
        for name in allowed:
            self.write("build/" + name)
        for name in ("var/studio/private.sqlite3", "QA/private.txt", "exports/raw_samples.npy", ".chart-data-temp/candidate.pptx"):
            self.write("build/" + name)
        names = {path.relative_to(self.root / "build").as_posix() for path in build_members(self.root / "build")}
        self.assertEqual(names, set(allowed))

    def test_raw_data_credentials_and_personal_model_assets_fail_closed(self):
        for index, name in enumerate(("raw_samples.npy", "features.csv", "recording.sqlite3", "recording.sqlite3-wal", "key.pfx", ".env", ".env.local", "attention_model.json")):
            with self.subTest(name=name):
                folder = self.root / f"build{index}"
                self.write(f"build{index}/{name}")
                with self.assertRaises(ValueError):
                    build_members(folder)

    def test_public_snapshot_rejects_raw_or_identifying_fields(self):
        name = "build/_internal/ningsi_studio/domain/showcase_evidence_snapshot.json"
        path = self.write(name, json.dumps({"evidence_origin": "bundled_development_snapshot", "summary": {"windows": 4}}))
        self.assertEqual(build_members(self.root / "build"), [path])
        path.write_text(json.dumps({"evidence_origin": "bundled_development_snapshot", "participant": "private-person"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_members(self.root / "build")

    def test_actual_suite_counts_are_parsed_without_fixed_totals(self):
        path = self.write("suite.log", "Ran 10 tests in 0.1s\n\nOK\nTesting ningsi ...\nRan 7 tests in 0.2s\n\nOK\n"
                          "Testing ningsi-studio ...\nRan 11 tests in 0.3s\n\nOK\nTesting ningsi-capture ...\n"
                          "Ran 5 tests in 0.4s\n\nOK\nSuite verification passed.\n")
        result = parse_suite(path)
        self.assertEqual(result["total"], 33)
        self.assertEqual([group["tests"] for group in result["groups"]], [10, 7, 11, 5])
        for text in ("Ran 10 tests in 0.1s\n\nOK\n", "FAILED (errors=1)\nSuite verification passed.\n",
                     "Ran 10 tests in 0.1s\n\nOK\nSuite verification passed.\n"):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_suite(path)

    def test_frozen_receipt_binds_both_executables_and_excludes_human_claims(self):
        hashes = {"studio": "a" * 64, "capture": "b" * 64}
        versions = {"studio": "1.2.3", "capture": "4.5.6"}
        receipt = {"passed": True, "check_count": 2, "checks": ["starts", "exports"], "exe_sha256": hashes,
                   "versions": versions, "hardware_or_human_validity_tested": False, "raw_fixture_payload": {"private": "must not be copied"}}
        path = self.write("receipt.json", json.dumps(receipt))
        result = read_verification(path, hashes, versions)
        self.assertEqual(result["exe_sha256"], hashes)
        self.assertNotIn("raw_fixture_payload", result)
        for changed in ({"hardware_or_human_validity_tested": True}, {"exe_sha256": {"studio": "c" * 64, "capture": "b" * 64}},
                        {"check_count": 3}, {"versions": {"studio": "0.0.0", "capture": "4.5.6"}}):
            path.write_text(json.dumps({**receipt, **changed}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_verification(path, hashes, versions)

    def test_studio_string_receipt_requires_matching_capture_build_binding(self):
        hashes = {"studio": "a" * 64, "capture": "b" * 64}
        versions = {"studio": "1.2.3", "capture": "4.5.6"}
        path = self.write("receipt.json", json.dumps({"passed": True, "check_count": 1, "exe_sha256": hashes["studio"],
                                                    "hardware_or_human_validity_tested": False}))
        with self.assertRaises(ValueError):
            read_verification(path, hashes, versions)
        self.assertTrue(read_verification(path, hashes, versions, {"sha256": hashes["capture"]})["passed"])

    def test_capture_tcn_build_receipt_binds_the_current_source_model(self):
        exe = self.write("capture/NingsiCapture.exe", "MZ fixture")
        model = self.write(product.FOCUS_TCN_MODEL, '{"model":"fixture"}')
        version = "0.4.2"
        legacy = {"app": "ningsi-capture", "version": version, "sha256": product.sha(exe), "mode": "onefile"}
        with patch.object(product, "ROOT", self.root):
            self.assertIsNone(product._capture_build_receipt(exe, version))
            receipt = self.write("capture/BUILD.json", json.dumps(legacy))
            self.assertNotIn("focus_tcn_sha256", product._capture_build_receipt(exe, version))
            digest = product.sha(model)
            receipt.write_text(json.dumps({**legacy, "focus_tcn_sha256": digest.upper()}), encoding="utf-8")
            self.assertEqual(product._capture_build_receipt(exe, version)["focus_tcn_sha256"], digest)
            for invalid in (None, "bad", "0" * 64):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "TCN model"):
                    receipt.write_text(json.dumps({**legacy, "focus_tcn_sha256": invalid}), encoding="utf-8")
                    product._capture_build_receipt(exe, version)

    def test_package_adds_opt_in_tcn_launcher_and_retains_default_capture(self):
        studio = self.write("studio/NingsiStudio.exe", "MZ studio")
        capture = self.write("capture/NingsiCapture.exe", "MZ capture")
        for capture_build in (None, {"focus_tcn_sha256": "a" * 64}):
            with self.subTest(capture_build=capture_build):
                label = "with-tcn" if capture_build else "default"
                release, archive, checksum = output_paths(label, self.root)
                inputs = dict(release=release, archive=archive, checksum=checksum, studio_build=studio.parent,
                              studio_exe=studio, capture_exe=capture, members=[studio], documents={},
                              versions={"studio": "0.1.15", "capture": "0.4.2"},
                              executables={"studio": product.sha(studio), "capture": product.sha(capture)},
                              capture_build=capture_build, receipt={"versions": True, "check_count": 1},
                              suite={"total": 1}, web_files_verified=1)
                with patch.object(product, "validate_inputs", return_value=inputs), \
                        patch.object(product, "_source_hashes", return_value={}), \
                        patch.object(product, "_git", return_value=""), redirect_stdout(StringIO()):
                    self.assertEqual(product.main(["--verification", "fixture", "--suite-log", "fixture", "--label", label]), 0)
                default = (release / "02-Start-Capture.bat").read_text(encoding="ascii")
                self.assertIn('\n"NingsiCapture.exe"\n', default)
                self.assertNotIn("--tcn-recovery", default)
                instructions = (release / "使用说明.txt").read_text(encoding="utf-8-sig")
                self.assertIn("断网运行", instructions)
                self.assertIn("快速模拟联调", instructions)
                launcher = "03-Start-Capture-TCN.bat"
                self.assertEqual((release / launcher).exists(), bool(capture_build))
                self.assertEqual(launcher in instructions, bool(capture_build))
                with ZipFile(archive) as zipped:
                    self.assertEqual(f"{label}/{launcher}" in zipped.namelist(), bool(capture_build))
                    build = json.loads(zipped.read(f"{label}/BUILD.json"))
                    self.assertEqual(build["capture_build"], capture_build)
                    if capture_build:
                        self.assertIn(b'"NingsiCapture.exe" gui --tcn-recovery\r\n', zipped.read(f"{label}/{launcher}"))


if __name__ == "__main__":
    unittest.main()
