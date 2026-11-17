"""Bounded lifecycle tests use stub sources only; they never discover hardware."""

from contextlib import ExitStack
import importlib.util
import itertools
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("capture_field_signal_tool", ROOT / "scripts/capture_field_signal.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CaptureFieldSignalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.created = []
        self.summary = {"raw_samples": 1250, "windows_total": 2, "windows_usable": 0,
                        "baseline_valid": False, "protocol_complete": False,
                        "train_eligible": 0, "test_eligible": 0}
        self.source = {"name": "stub-simulation-eeg", "kind": "lsl", "data_origin": "simulation_lsl",
                       "channels": 2, "srate": 250., "channel_labels": ["Ch1", "Ch2"],
                       "reference": "unknown", "declared_units": ["", ""],
                       "source_unit": "unknown", "scale_to_uv": None}

    def install(self, *, recorder_type=None, sources=None, sample_count=1250):
        stack = ExitStack()
        self.addCleanup(stack.close)
        times = itertools.count(0., .1)
        stack.enter_context(patch.object(tool.time, "monotonic", side_effect=lambda: next(times)))
        stack.enter_context(patch.object(tool.time, "sleep", return_value=None))
        stack.enter_context(patch.object(tool, "automatic_unit", return_value={}))
        stack.enter_context(patch.object(tool, "supported", return_value=False))
        stack.enter_context(patch.object(tool, "discover", return_value=[(object(), self.source)]
                                         if sources is None else sources))
        created = self.created

        class Store:
            def __init__(self, root, **kwargs):
                self.metadata = dict(kwargs)
                self.metadata["source"] = dict(kwargs["source"])
                self.path = Path(root) / "stub-recording"
                self.path.mkdir(parents=True)
                self.sample_count = sample_count
                self.clock = SimpleNamespace(now=lambda: SimpleNamespace(perf=tool.time.monotonic()))
                self.lock = threading.RLock()
                self.events = []
                self.closed = False
                created.append(self)

            @property
            def last_received(self):
                return tool.time.monotonic()

            def event(self, kind, payload):
                self.events.append((kind, payload))

            def close(self, state, reason):
                self.closed, self.state, self.reason = True, state, reason

        self.store_factory = stack.enter_context(patch.object(tool, "SessionStore", side_effect=Store))

        class Recorder:
            def __init__(self, store, info):
                self.ready, self.stop_event = threading.Event(), threading.Event()
                self.error = None
                self.ident = None
                self.started = False

            def start(self):
                self.ident = 1
                self.started = True
                self.ready.set()

            def is_alive(self):
                return self.started and not self.stop_event.is_set()

            def stop(self):
                self.stop_event.set()

            def join(self, timeout):
                pass

        chosen = recorder_type(Recorder) if recorder_type else Recorder
        stack.enter_context(patch.object(tool, "LslRecorder", chosen))

        def export(path, output):
            Path(output).mkdir(parents=True)
            for name in ("raw_samples.npy", "timestamps.npy", "events.csv", "windows.csv",
                         "manifest.json", "export.json", "SHA256.json"):
                (Path(output) / name).write_bytes(b"stub-only")
            return self.summary

        self.exporter = stack.enter_context(patch.object(tool, "export_session", side_effect=export))

    def run_capture(self, seconds=5.):
        return tool.capture_signal(device=self.source["name"], seconds=seconds,
                                   data=self.root / "new-recordings", export=self.root / "new-export")

    def test_signal_only_pilot_has_no_labels_completion_or_invented_electrodes(self):
        self.install()
        result = self.run_capture()
        store = self.created[0]
        self.assertTrue(store.closed)
        self.assertEqual(store.state, "completed")
        self.assertEqual(store.metadata["cohort"], "pilot")
        self.assertRegex(store.metadata["participant"], r"^QC[0-9]{8}$")
        self.assertEqual(store.metadata["protocol"]["name"], "signal-check-only-v1")
        self.assertFalse(store.metadata["lsl_markers"])
        self.assertEqual([kind for kind, _ in store.events], ["signal_check_start", "signal_check_stop"])
        self.assertEqual(result["source"]["channel_labels"], ["Ch1", "Ch2"])
        self.assertEqual(result["source"]["reference"], "unknown")
        self.assertEqual(result["source"]["unit_selection"], "raw_only")
        self.assertIsNone(result["source"]["scale_to_uv"])
        self.assertEqual(result["source"]["data_origin"], "simulation_lsl")
        self.assertFalse(result["physical_wear_and_latency_validated"])
        self.assertFalse(result["protocol_complete"])
        self.assertEqual(result["train_eligible"], 0)

    def test_no_source_and_same_name_conflict_fail_without_creating_a_record(self):
        self.install(sources=[])
        with self.assertRaisesRegex(RuntimeError, "精确匹配"):
            self.run_capture()
        self.store_factory.assert_not_called()
        with patch.object(tool, "discover", return_value=[(object(), self.source), (object(), self.source)]):
            with self.assertRaisesRegex(RuntimeError, "多个同名"):
                self.run_capture()
        self.store_factory.assert_not_called()

    def test_invalid_duration_and_existing_directories_fail_before_discovery(self):
        with patch.object(tool, "discover") as discover:
            for seconds in (4.99, 60.01, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self.run_capture(seconds)
            (self.root / "new-recordings").mkdir()
            with self.assertRaises(FileExistsError):
                self.run_capture()
            discover.assert_not_called()

    def test_insufficient_raw_history_is_closed_failed_and_never_exported(self):
        self.install(sample_count=1)
        with self.assertRaisesRegex(RuntimeError, "原始样本不足"):
            self.run_capture()
        self.assertTrue(self.created[0].closed)
        self.assertEqual(self.created[0].state, "failed")
        self.exporter.assert_not_called()

    def test_stop_exception_fails_even_when_fallback_stop_event_can_close(self):
        def recorder(base):
            class StopFailure(base):
                def stop(self):
                    raise RuntimeError("stub stop failure")
            return StopFailure
        self.install(recorder_type=recorder)
        with self.assertRaisesRegex(RuntimeError, "停止采集失败"):
            self.run_capture()
        self.assertTrue(self.created[0].closed)
        self.assertEqual(self.created[0].state, "failed")
        self.exporter.assert_not_called()

    def test_export_without_window_history_or_with_training_rows_is_rejected(self):
        self.install()
        self.summary["windows_total"] = 0
        with self.assertRaisesRegex(RuntimeError, "窗口历史"):
            self.run_capture()
        self.assertTrue(self.created[0].closed)

    def test_automatic_unit_is_used_only_with_its_recorded_basis(self):
        self.install()
        with patch.object(tool, "automatic_unit", return_value={"source_unit": "mV", "scale_to_uv": 1000.,
                                                               "unit_basis": "stream_metadata"}):
            result = self.run_capture()
        self.assertEqual(result["source"]["scale_to_uv"], 1000.)
        self.assertEqual(result["source"]["unit_basis"], "stream_metadata")


if __name__ == "__main__":
    unittest.main()
