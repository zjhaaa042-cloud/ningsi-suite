from contextlib import closing
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from .test_support import FakeClock, destroy_test_root
from ningsi_capture.acquisition import simulation_source
from ningsi_capture.clock import Stamp
from ningsi_capture.markers import MarkerPublisher, marker_sample
from ningsi_capture.storage import SessionStore, open_readonly, read_metadata


class LslClock(FakeClock):
    domain = "lsl_local_clock"


class Outlet:
    def __init__(self):
        self.sent = []
        self.threads = []

    def push_sample(self, sample, timestamp):
        self.sent.append((json.loads(sample[0]), timestamp))
        self.threads.append(threading.get_ident())


class MarkerTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("NINGSI_LSL_MARKER_LOOPBACK") == "1", "explicit optional LSL loopback")
    def test_optional_real_lsl_string_marker_loopback(self):
        from pylsl import StreamInlet, resolve_byprop
        from ningsi_capture.clock import Clock
        clock = Clock()
        self.assertEqual(clock.domain, "lsl_local_clock")
        publisher = MarkerPublisher(enabled=True, clock_domain=clock.domain, origin="simulation")
        inlet = None
        try:
            infos = resolve_byprop("source_id", publisher.source_id, timeout=5.)
            self.assertTrue(infos, publisher.snapshot())
            inlet = StreamInlet(infos[0], processing_flags=0)
            inlet.open_stream(timeout=2.)
            stamp = clock.now()
            publisher.publish(41, "stimulus_onset", {"phase": "sart_1", "index": 8, "choice": 3}, stamp.lsl)
            sample, timestamp = inlet.pull_sample(timeout=3.)
            self.assertIsNotNone(sample)
            marker = json.loads(sample[0])
            self.assertEqual(marker["event_id"], 41)
            self.assertNotIn("choice", marker)
            self.assertEqual(timestamp, stamp.lsl)
        finally:
            if inlet is not None:
                inlet.close_stream()
            publisher.close()

    def test_disabled_and_fallback_clock_never_create_an_outlet(self):
        with patch("ningsi_capture.markers._create_outlet") as factory:
            disabled = MarkerPublisher(clock_domain="lsl_local_clock")
            self.assertFalse(disabled.publish(1, "recording_start", {}, 10.))
            self.assertEqual(disabled.close()["state"], "disabled")
            fallback = MarkerPublisher(enabled=True, clock_domain="perf_counter_fallback")
            self.assertFalse(fallback.publish(1, "recording_start", {}, 10.))
            status = fallback.close()
            self.assertEqual(status["error_code"], "clock_domain_unavailable")
            self.assertEqual(status["state"], "unavailable")
            factory.assert_not_called()

    def test_sample_allowlist_excludes_answers_identity_and_arbitrary_text(self):
        payload = {"phase": "sart_2", "index": 17, "block": 2, "practice": False, "task": "sart",
                   "participant": "PRIVATE_PERSON", "notes": "PRIVATE_NOTE", "choice": 3,
                   "choice_text": "PRIVATE_ANSWER", "label": 1, "score": .9,
                   "trial_id": "PRIVATE_ID", "digit": 4, "rt_sec": .2,
                   "reason": "PRIVATE_REASON", "config": {"secret": "PRIVATE_CONFIG"}}
        sample, stamp = marker_sample(88, "stimulus_onset", payload, 1234.56789)
        item = json.loads(sample)
        self.assertEqual(item, {"schema": "ningsi-capture-task-marker-v1", "event_id": 88,
                                "kind": "stimulus_onset", "phase": "sart_2", "trial_index": 17,
                                "block": 2, "practice": False, "task": "sart"})
        self.assertEqual(stamp, 1234.56789)
        self.assertNotIn("PRIVATE", sample)
        unknown_phase = marker_sample(89, "probe_onset", {"phase": "PRIVATE_PERSON", "after_trial": 19}, 12.)
        self.assertNotIn("phase", json.loads(unknown_phase[0]))
        self.assertEqual(json.loads(unknown_phase[0])["trial_index"], 19)
        self.assertIsNone(marker_sample(90, "probe_response", payload, 12.))
        self.assertIsNone(marker_sample(90, "PRIVATE_EVENT", payload, 12.))
        self.assertIsNone(marker_sample(90, "probe_onset", payload, float("nan")))

    def test_store_publishes_saved_event_ids_and_original_lsl_stamp_from_worker(self):
        outlet = Outlet()
        with tempfile.TemporaryDirectory() as directory, patch(
                "ningsi_capture.markers._create_outlet", return_value=outlet) as factory:
            store = SessionStore(directory, participant="PRIVATE_PERSON", notes="PRIVATE_NOTE",
                                 source=simulation_source(), protocol={}, clock=LslClock(), lsl_markers=True)
            original = Stamp(perf=900., lsl=901.123456789, utc="test-time")
            event_id = store.event("stimulus_onset", {"phase": "sart_1", "index": 6,
                                                      "choice": 3, "score": .8}, original)
            store.close("aborted")
            with closing(open_readonly(store.path)) as reader:
                row = reader.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
                metadata = read_metadata(reader)
            matches = [(item, stamp) for item, stamp in outlet.sent if item["event_id"] == event_id]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0][1], row["time_lsl"])
            self.assertEqual(matches[0][1], original.lsl)
            self.assertEqual(json.loads(row["payload"])["choice"], 3)
            self.assertEqual(json.loads(row["payload"])["score"], .8)
            self.assertNotIn("choice", matches[0][0])
            self.assertTrue(all(ident != threading.get_ident() for ident in outlet.threads))
            self.assertNotIn("PRIVATE", json.dumps(outlet.sent))
            self.assertNotIn("PRIVATE", json.dumps(factory.call_args.args))
            status = metadata["lsl_markers"]
            self.assertTrue(status["enabled"])
            self.assertEqual(status["published"], 3)  # recording_start, stimulus, recording_stop
            self.assertEqual(status["state"], "closed")
            self.assertEqual(status["declared_data_origin"], "simulation")
            self.assertIn("not validated", status["scope"])

    def test_outlet_failure_does_not_fail_database_event_or_store_close(self):
        outlet = Outlet()
        outlet.push_sample = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("PRIVATE transport error"))
        with tempfile.TemporaryDirectory() as directory, patch(
                "ningsi_capture.markers._create_outlet", return_value=outlet):
            store = SessionStore(directory, participant="T001", source=simulation_source(),
                                 protocol={}, clock=LslClock(), lsl_markers=True)
            event_id = store.event("probe_onset", {"phase": "sart_1", "choice": 5})
            store.append([[1., 2.]], [1000.], offset=0.)
            store.close("aborted")
            with closing(open_readonly(store.path)) as reader:
                self.assertIsNotNone(reader.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone())
                self.assertEqual(reader.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 1)
                status = read_metadata(reader)["lsl_markers"]
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["failures"], 1)
            self.assertEqual(status["error_code"], "outlet_publish_failed")
            self.assertGreaterEqual(status["dropped"], 1)
            self.assertNotIn("PRIVATE", json.dumps(status))

    def test_slow_network_setup_has_bounded_queue_and_bounded_close(self):
        release = threading.Event()
        outlet = Outlet()

        def create(*args):
            release.wait(2.)
            return outlet

        with patch("ningsi_capture.markers._create_outlet", side_effect=create):
            publisher = MarkerPublisher(enabled=True, clock_domain="lsl_local_clock", capacity=1)
            try:
                before = time.monotonic()
                for number in range(1, 101):
                    publisher.publish(number, "phase_start", {"phase": "sart_1"}, 10.)
                self.assertLess(time.monotonic() - before, .5)
                before = time.monotonic()
                snapshot = publisher.close(timeout=.01)
                self.assertLess(time.monotonic() - before, .5)
                self.assertTrue(snapshot["shutdown_pending"])
                self.assertEqual(snapshot["pending"], 1)
                self.assertEqual(snapshot["dropped"], 99)
            finally:
                release.set()
                publisher._thread.join(2.)

    def test_creation_and_worker_start_failures_are_explicit(self):
        with patch("ningsi_capture.markers._create_outlet", side_effect=RuntimeError("PRIVATE")):
            publisher = MarkerPublisher(enabled=True, clock_domain="lsl_local_clock")
            status = publisher.close(timeout=1.)
            self.assertEqual(status["error_code"], "outlet_creation_failed")
            self.assertNotIn("PRIVATE", json.dumps(status))
        with patch("ningsi_capture.markers.threading.Thread.start", side_effect=RuntimeError("PRIVATE")):
            publisher = MarkerPublisher(enabled=True, clock_domain="lsl_local_clock")
            self.assertEqual(publisher.close()["error_code"], "worker_start_failed")

    def test_marker_adapter_exception_never_rolls_back_database_event(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory, participant="T001", source=simulation_source(), protocol={}, clock=FakeClock())
            with patch.object(store.markers, "publish", side_effect=RuntimeError("marker adapter failed")):
                event_id = store.event("phase_start", {"phase": "sart_1"})
            store.close("aborted")
            with closing(open_readonly(store.path)) as reader:
                self.assertIsNotNone(reader.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone())
                self.assertEqual(read_metadata(reader)["lsl_markers"]["adapter_failures"], 1)

    def test_cli_marker_opt_in_is_explicit_and_off_by_default(self):
        from ningsi_capture.__main__ import main
        with patch("ningsi_capture.gui.run") as gui:
            self.assertEqual(main(["gui"]), 0)
            gui.assert_called_with(None, lsl_markers=False)
            self.assertEqual(main(["gui", "--lsl-markers"]), 0)
            gui.assert_called_with(None, lsl_markers=True)


class MarkerGuiTests(unittest.TestCase):
    def test_checkbox_is_visible_and_explicit_selection_reaches_new_recording(self):
        import gc
        import tkinter as tk
        from ningsi_capture.gui import CaptureApp
        root = tk.Tk()
        root.attributes("-alpha", 0.)
        app = None
        with tempfile.TemporaryDirectory() as directory, patch(
                "ningsi_capture.markers._create_outlet", return_value=Outlet()), patch(
                "ningsi_capture.gui.SimRecorder") as recorder, patch("ningsi_capture.gui.QualityMonitor"):
            try:
                app = CaptureApp(root, directory)
                self.assertFalse(app.lsl_markers.get())
                root.geometry("1050x690+10+10")
                app.marker_checkbox.master.master.select(app.marker_checkbox.master)
                root.deiconify()
                for _ in range(3):
                    root.update()
                self.assertTrue(app.marker_checkbox.winfo_ismapped())
                self.assertLessEqual(app.marker_checkbox.winfo_rooty() + app.marker_checkbox.winfo_height(),
                                     root.winfo_rooty() + root.winfo_height())
                self.assertLessEqual(app.marker_checkbox.winfo_rooty() + app.marker_checkbox.winfo_height(),
                                     app.marker_checkbox.master.winfo_rooty() + app.marker_checkbox.master.winfo_height())
                app.clock = LslClock()
                app.source_mode.set("模拟演示")
                app.protocol_mode.set("快速模拟联调")
                app.lsl_markers.set(True)
                recorder.return_value.error = None
                app.start_recording()
                self.assertIsNotNone(app.store)
                self.assertTrue(app.store.markers.requested)
                self.assertTrue(app.store.markers.enabled)
            finally:
                if app and app.store:
                    app.store.close("aborted")
                destroy_test_root(root)
                gc.collect()


if __name__ == "__main__":
    unittest.main()
