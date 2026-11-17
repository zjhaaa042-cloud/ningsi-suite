"""Independent QC must not manufacture clean EEG, units, timestamps or labels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_field_signal import audit_export, checked_file, clock_summary, spectral_reference, write_report


class FieldSignalAuditTests(unittest.TestCase):
    def test_clock_disorder_and_missing_values_are_not_repaired(self):
        result = clock_summary([0, .004, .004, .003, np.nan, .5, .7])
        self.assertEqual(result["nonincreasing_adjacent_pairs"], 2)
        self.assertEqual(result["finite_samples"], 6)
        self.assertEqual(result["gaps_over_100ms"], 1)

    def test_checksum_mismatch_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "raw.npy").write_bytes(b"unmodified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                checked_file(root, "raw.npy", {"raw.npy": "0" * 64})
            self.assertEqual(checked_file(root, "raw.npy", {"raw.npy": hashlib.sha256(b"unmodified").hexdigest()}), root / "raw.npy")
            with self.assertRaises(ValueError):
                checked_file(root, str(Path(__file__).resolve()), {})

    def test_report_escapes_text_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            result = {"checks": [{"id": "units", "status": "pending", "message": "<script>private</script>"}]}
            write_report(result, output)
            self.assertNotIn("<script>", (output / "audit.html").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((output / "audit.json").read_text(encoding="utf-8")), result)
            with self.assertRaises(FileExistsError):
                write_report(result, output)

    @unittest.skipUnless(importlib.util.find_spec("mne"), "Install optional field-audit requirements")
    def test_mne_detects_line_dominance_without_rejecting_or_filling_samples(self):
        times = np.arange(250 * 8) / 250
        signal = np.stack([10 * np.sin(2 * np.pi * 10 * times) + 100 * np.sin(2 * np.pi * 50 * times),
                           10 * np.sin(2 * np.pi * 10 * times)], axis=1)
        original = signal.copy()
        result = spectral_reference(signal, 250)
        self.assertGreater(result["line_fraction_median_by_channel"]["50"][0], .95)
        self.assertLess(result["line_fraction_median_by_channel"]["50"][1], .01)
        np.testing.assert_array_equal(signal, original)
        signal[100, 0] = np.nan
        result = spectral_reference(signal, 250)
        self.assertEqual(result["nonfinite_blocks"], 1)
        self.assertEqual(result["spectrally_computable_blocks"], 1)
        self.assertTrue(spectral_reference(original, 250, max_windows=1)["truncated"])

    @unittest.skipUnless(importlib.util.find_spec("mne"), "Install optional field-audit requirements")
    def test_export_audit_preserves_input_and_records_every_clock_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = np.arange(2000) / 250
            np.save(root / "raw_samples.npy", np.sin(2 * np.pi * 10 * times)[:, None])
            np.save(root / "timestamps.npy", np.stack([times] * 4, axis=1))
            np.save(root / "packet_timestamps.npy", times)
            np.save(root / "device_timestamps.npy", times)
            (root / "manifest.json").write_text(json.dumps({"source": {"channels": 1, "srate": 250,
                "data_origin": "simulation", "source_unit": "unknown"}}), encoding="utf-8")
            export = {"raw_samples": 2000, "baseline_valid": False, "training_readiness": {"paired": True},
                      "timestamp_columns": ["source_timestamp", "corrected_timestamp", "received_lsl", "received_perf"]}

            def refresh():
                (root / "export.json").write_text(json.dumps(export), encoding="utf-8")
                hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()
                          if path.name != "SHA256.json"}
                (root / "SHA256.json").write_text(json.dumps(hashes), encoding="utf-8")
                return hashes

            before = refresh()
            result = audit_export(root)
            self.assertEqual(result["input_files_sha256"], before)
            self.assertEqual(result["checks"][2]["status"], "fail")  # bool is not a probe count
            self.assertFalse(result["hardware_or_human_validity_established"])
            self.assertFalse(result["training_data_validation"]["import_valid"])  # no fake feature dataset
            for name, digest in before.items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
            export["timestamp_columns"][0] = "unexpected"
            refresh()
            with self.assertRaisesRegex(ValueError, "timestamp column"):
                audit_export(root)


if __name__ == "__main__":
    unittest.main()
