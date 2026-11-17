"""Synthetic segment evidence must never manufacture continuity or eligibility."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_artifact_segments import (MASK_BITS, POLICY, ROOT, TIMESTAMP_COLUMNS, audit_export,
                                     coverage, protect, runs, screen_segments, union, write_report)


def signal(total=1400, rate=100):
    times = np.arange(total) / rate
    return np.stack([10 * np.sin(2 * np.pi * 10 * times) + 4 * np.sin(2 * np.pi * 20 * times),
                     9 * np.sin(2 * np.pi * 10 * times + .2) + 4 * np.sin(2 * np.pi * 20 * times)], axis=0)


class ArtifactSegmentTests(unittest.TestCase):
    def fixture(self, root):
        rate, total = 100, 1400
        wave = signal(total, rate)
        times = 1000 + np.arange(total) / rate
        np.save(root / "raw_samples.npy", wave.T)
        np.save(root / "timestamps.npy", np.stack([times] * 4, axis=1))
        count = 1 + (total - 400) // 200
        cleaned = np.stack([wave[:, i * 200:i * 200 + 400] for i in range(count)])
        masks = np.zeros((count, 400), dtype=np.uint8)
        masks[0] = 16
        cleaned[0] = np.nan
        np.save(root / "cleaned_windows_uv.npy", cleaned)
        np.save(root / "artifact_mask.npy", masks)
        fingerprint = {name: hashlib.sha256((ROOT / "ningsi/src/ningsi" / name).read_bytes()).hexdigest()
                       for name in ("config.py", "signal/artifacts.py", "signal/spectrum.py", "signal/bands.py", "signal/conditioning.py")}
        source = {"channels": 2, "srate": rate, "source_unit": "uV", "scale_to_uv": 1.,
                  "preprocessing_policy": POLICY, "timestamp_policy": "source-timestamps-v1"}
        (root / "manifest.json").write_text(json.dumps({"source": source, "sample_count": total}), encoding="utf-8")
        (root / "export.json").write_text(json.dumps({"raw_samples": total, "windows_total": count,
            "windows_usable": count - 1, "timestamp_columns": TIMESTAMP_COLUMNS,
            "timestamp_policy": "source-timestamps-v1", "preprocessing": {"policy": POLICY, "units": "uV",
            "causal": True, "history_sec": 2, "mask_bits": MASK_BITS},
            "engine_fingerprint_at_export": fingerprint, "implementation_fingerprint_at_export": fingerprint}), encoding="utf-8")
        with (root / "windows.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["start_lsl", "end_lsl", "sample_end_index", "usable", "reasons"])
            writer.writeheader()
            writer.writerows({"start_lsl": times[i * 200], "end_lsl": times[i * 200 + 399] + .01,
                             "sample_end_index": i * 200 + 399, "usable": i > 0,
                             "reasons": "preprocessing_warmup" if i == 0 else ""} for i in range(count))
        (root / "events.csv").write_text("event_id,kind,time_perf\n", encoding="utf-8")
        self.refresh(root)

    def refresh(self, root):
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()
                  if path.name != "SHA256.json"}
        (root / "SHA256.json").write_text(json.dumps(hashes), encoding="utf-8")
        return hashes

    def test_interval_union_never_double_counts_overlapping_windows(self):
        self.assertEqual(union([(0, 400), (200, 600), (800, 900)]), [[0, 600], [800, 900]])
        result = coverage([(0, 400), (200, 600)], 100, 1000, [(100, 500)])
        self.assertEqual(result["seconds_at_nominal_rate"], 6)
        self.assertEqual(result["additional_seconds_outside_strict"], 2)
        self.assertEqual(runs([True, True, False, True]), [(0, 2), (3, 4)])

    def test_short_fragments_are_not_joined_or_padded_into_two_seconds(self):
        wave = signal(500)
        available = np.zeros_like(wave, dtype=bool)
        available[:, :190] = True
        available[:, 200:390] = True
        result = screen_segments(wave, available, 100, 2)
        self.assertEqual(result["all_channels"]["examined_blocks"], 0)
        self.assertEqual(result["all_channels"]["samples"], 0)

    def test_real_two_seconds_stay_descriptive_and_four_seconds_require_contiguity(self):
        wave = signal(500)
        available = np.zeros_like(wave, dtype=bool)
        available[:, 100:350] = True
        two = screen_segments(wave, available, 100, 2)
        four = screen_segments(wave, available, 100, 4)
        self.assertEqual(two["interpretation"], "descriptive_only")
        self.assertEqual(two["all_channels"]["samples"], 250)
        self.assertEqual(four["all_channels"]["samples"], 0)
        self.assertNotIn("focus", two)

    def test_filter_support_guard_rejects_neighbouring_samples_without_closing_gaps(self):
        mask = np.zeros(1000, dtype=bool)
        mask[400:425] = True
        guarded = protect(mask, 200)
        self.assertEqual(runs(guarded), [(200, 625)])
        self.assertEqual(runs(protect(np.array([True, False, False]), 2)), [(0, 3)])
        available = np.broadcast_to(~guarded, (2, 1000))
        two = screen_segments(signal(1000), available, 100, 2)
        for a, b in two["all_channels"]["sample_intervals_half_open"]:
            self.assertTrue(b <= 200 or a >= 625)

    def test_per_channel_and_joint_screening_preserve_bad_channel(self):
        wave = signal(500)
        wave[1] = 500 * np.sin(2 * np.pi * 10 * np.arange(500) / 100)
        result = screen_segments(wave, np.ones_like(wave, dtype=bool), 100, 2)
        self.assertGreater(result["channels"][0]["passed_blocks"], 0)
        self.assertEqual(result["channels"][1]["passed_blocks"], 0)
        self.assertEqual(result["all_channels"]["passed_blocks"], 0)
        self.assertIn("amplitude", result["channels"][1]["rejected_block_reasons"])

    def test_read_only_roundtrip_keeps_eligibility_and_union_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            before = self.refresh(root)
            result = audit_export(root)
            self.assertEqual(result["mask_scope"], "joint_time_mask_from_any_channel; triggering_channel_unknown")
            self.assertEqual(result["recorded_strict"]["seconds_at_nominal_rate"], 12)
            self.assertFalse(result["state_scoring_eligibility_changed"])
            self.assertEqual(result["new_train_eligible"], 0)
            self.assertEqual(result["new_test_eligible"], 0)
            self.assertFalse(result["hardware_or_clean_signal_truth_established"])
            self.assertEqual(result["input_files_sha256"], before)
            for name, expected in before.items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)

    def test_overlap_mask_union_keeps_every_detected_sample_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            masks = np.load(root / "artifact_mask.npy")
            masks[2, 100] = 2  # Original sample 500 also occurs in an unmasked neighbour.
            np.save(root / "artifact_mask.npy", masks)
            data = np.load(root / "cleaned_windows_uv.npy")
            data[2, :, 100] = np.nan
            np.save(root / "cleaned_windows_uv.npy", data)
            with (root / "windows.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[2].update(usable="False", reasons="muscle_burst")
            with (root / "windows.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            item = json.loads((root / "export.json").read_text())
            item["windows_usable"] -= 1
            (root / "export.json").write_text(json.dumps(item))
            self.refresh(root)
            result = audit_export(root)
            self.assertEqual(result["overlapping_mask_disagreement_samples"], 1)
            self.assertEqual(result["local_mask_union_inside_recorded_strict_samples"], 1)
            for segment in result["scenarios"][0]["segments"]:
                for a, b in segment["all_channels"]["sample_intervals_half_open"]:
                    self.assertFalse(a <= 500 < b)

    def test_hard_unit_timestamp_nan_or_gap_is_not_locally_salvaged(self):
        for failure in ("unit", "unit_scale", "source_type", "clock", "nonfinite", "source_gap", "unknown_reason"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                if failure in ("unit", "unit_scale", "source_type"):
                    item = json.loads((root / "manifest.json").read_text())
                    if failure == "unit":
                        item["source"]["source_unit"] = "unknown"
                    elif failure == "unit_scale":
                        item["source"]["scale_to_uv"] = 1000
                    else:
                        item["source"]["type"] = "IMPEDANCE"
                    (root / "manifest.json").write_text(json.dumps(item))
                elif failure == "clock":
                    stamps = np.load(root / "timestamps.npy")
                    stamps[:, 1] = np.nan
                    np.save(root / "timestamps.npy", stamps)
                elif failure == "nonfinite":
                    raw = np.load(root / "raw_samples.npy")
                    raw[:] = np.nan
                    np.save(root / "raw_samples.npy", raw)
                elif failure == "source_gap":
                    (root / "events.csv").write_text("event_id,kind,time_perf\n1,source_gap_start,1000\n")
                else:
                    text = (root / "windows.csv").read_text()
                    (root / "windows.csv").write_text(text.replace("True,", "False,new_unknown_fault"))
                    item = json.loads((root / "export.json").read_text())
                    item["windows_usable"] = 0
                    (root / "export.json").write_text(json.dumps(item))
                self.refresh(root)
                result = audit_export(root)
                self.assertTrue(result["hard_exclusion_window_counts"])
                for scenario in result["scenarios"]:
                    for segment in scenario["segments"]:
                        self.assertEqual(segment["all_channels"]["samples"], 0)

    def test_checksum_shape_unknown_mask_and_overlap_inconsistency_fail_closed(self):
        for failure in ("checksum", "shape", "mask_bit", "overlap", "fingerprint"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                if failure == "checksum":
                    (root / "windows.csv").write_text("changed")
                elif failure == "shape":
                    np.save(root / "artifact_mask.npy", np.zeros((6, 2, 400), dtype=np.uint8))
                elif failure == "mask_bit":
                    masks = np.load(root / "artifact_mask.npy")
                    masks[1, 10] = 32
                    np.save(root / "artifact_mask.npy", masks)
                elif failure == "overlap":
                    data = np.load(root / "cleaned_windows_uv.npy")
                    data[2, :, 10] += 1
                    np.save(root / "cleaned_windows_uv.npy", data)
                else:
                    item = json.loads((root / "export.json").read_text())
                    item["preprocessing"]["history_sec"] = 0
                    (root / "export.json").write_text(json.dumps(item))
                if failure != "checksum":
                    self.refresh(root)
                with self.assertRaises(ValueError):
                    audit_export(root)

    def test_path_escape_and_report_escaping_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            hashes = json.loads((root / "SHA256.json").read_text())
            from audit_field_signal import checked_file
            with self.assertRaises(ValueError):
                checked_file(root, str(Path(__file__).resolve()), hashes)
            item = audit_export(root)
            item["input_directory"] = "<script>bad</script>"
            output = root / "report"
            write_report({"exports": [item]}, output)
            self.assertNotIn("<script>", (output / "audit.html").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_report({"exports": [item]}, output)


if __name__ == "__main__":
    unittest.main()
