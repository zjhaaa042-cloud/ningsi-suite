import json
from pathlib import Path
import tempfile
import unittest

from .test_support import FakeClock
import numpy as np
from ningsi_capture.acquisition import simulation_source
from ningsi_capture.storage import SessionStore, open_readonly, read_metadata


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SessionStore(self.temp.name, participant="T001", source=simulation_source(), protocol={}, clock=FakeClock())
        self.addCleanup(lambda: self.store.close("aborted") if not self.store.closed else None)

    def test_raw_values_and_source_timestamps_survive_exactly(self):
        values = np.array([[2.125, -0.0], [np.nan, np.inf], [-25.125, .000123]], dtype="<f8")
        stamps = np.array([999.75, 999.75, 999.7], dtype="<f8")
        correction = self.store.correction(.25)
        self.store.append(values[:2], stamps[:2], offset=.25, correction_id=correction)
        self.store.append(values[2:], stamps[2:], offset=.25, correction_id=correction)
        connection = open_readonly(self.store.path)
        self.addCleanup(connection.close)
        rows = list(connection.execute("SELECT * FROM samples ORDER BY sample_index"))
        self.assertEqual(len(rows), 3)  # Duplicate or backwards timestamps are not deduplicated away.
        self.assertEqual(b"".join(row["values_f64"] for row in rows), values.tobytes())
        self.assertEqual(b"".join(row["source_timestamp_bytes"] for row in rows), stamps.tobytes())
        self.assertEqual(rows[0]["corrected_timestamp"], 1000.)

    def test_each_append_is_visible_to_an_independent_reader(self):
        self.store.append([[1, 2]], [1000.], offset=0)
        reader = open_readonly(self.store.path)
        self.addCleanup(reader.close)
        self.assertEqual(reader.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 1)
        self.store.append([[3, 4]], [1000.004], offset=0)
        self.assertEqual(reader.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 2)

    def test_bad_chunk_does_not_silently_drop_or_partially_insert(self):
        for values, times in (([[1, 2], [3, 4]], [1]), ([[1]], [1])):
            with self.assertRaises(ValueError):
                self.store.append(values, times)
        self.assertEqual(self.store.sample_count, 0)

    def test_closed_recording_retains_explicit_abort_reason(self):
        self.store.event("probe_response", {"probe_id": "p1", "choice": 3})
        self.store.close("aborted", "operator_stop")
        reader = open_readonly(self.store.path)
        self.addCleanup(reader.close)
        metadata = read_metadata(reader)
        self.assertEqual(metadata["state"], "aborted")
        self.assertEqual(metadata["reason"], "operator_stop")
        with self.assertRaises(RuntimeError):
            self.store.append([[1, 2]], [1])

    def test_unknown_units_allowed_for_raw_not_invented(self):
        self.assertEqual(self.store.metadata["source"]["source_unit"], "uV")
        self.assertEqual(self.store.metadata["source"]["data_origin"], "simulation")

    def test_invalid_metadata_does_not_create_recording(self):
        for participant in ("../bad", "名字", "person@example.com"):
            with self.assertRaises(ValueError):
                SessionStore(self.temp.name, participant=participant, source=simulation_source(), protocol={})


if __name__ == "__main__":
    unittest.main()
