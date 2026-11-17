from contextlib import closing
import tempfile
import gc
import json
import socket
import time
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from .test_support import FakeClock
from ningsi_capture.acquisition import simulation_source, synthetic_samples
from ningsi_capture.gui import AUTO_REC, AUTO_UNIT, CaptureApp
from ningsi_capture.live_quality import CaptureQuality, QualityMonitor
from ningsi.models.capture_protocols import FULL_PROTOCOL, RESEARCH_PROTOCOLS, SHORT_PROTOCOL
from ningsi_capture.storage import SessionStore, open_readonly, read_metadata
from ningsi_capture.protocol import TaskRunner, ProtocolConfig
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, BIOMULTI_REC_TIMING


class GuiLayoutTests(unittest.TestCase):
    def tearDown(self):
        # Dispose Tk cycles on their owning thread before recorder/quality
        # workers in later tests can trigger Python's cyclic collector.
        gc.collect()

    def test_long_operator_diagnostics_keep_phase_clock_and_stop_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = CaptureApp(root, directory)
                app.quality_text.set("设备 · 4 秒窗质控 未通过 · 幅度过大、低频伪迹偏高 · 分包时钟估计\n"
                                     "原始 50 Hz 占比 Ch1 96.9%、Ch2 97.0% · 工频占多数，检查电极接触与线缆\n"
                                     "最近 14 窗合格 2/14（14%）\n睁眼基线 未通过 · 合格 14/58 窗"
                                     "（24.1%，要求 ≥60%），覆盖 50.04/30 秒 · 正式特征不可用，建议调整后新建采集\n"
                                     "SART 区块 3 · 已提示 5，脑电可用 0，同时有二分类回答 0")
                app.eligibility_text.set("开发组 · 合格探针用于训练\n本场正式提示 15 个 · 自报专注 12，内部走神 0"
                                         " · 脑电配对专注 0，内部走神 0（导出核定）")
                app.phase_text.set("阶段 9/9 · 数字任务 第 3 区块 · 已用 01:10 · 预计剩余 03:20")
                root.attributes("-alpha", 0.)
                root.deiconify()
                for width, height in ((1050, 690), (1200, 780)):
                    root.geometry(f"{width}x{height}+10+10")
                    for _ in range(4):
                        root.update()
                    phase = app.canvas.master.grid_slaves(row=4, column=0)[0]
                    for widget in (phase, app.canvas, app.stop_button, app.export_button, app.log):
                        self.assertTrue(widget.winfo_ismapped(), str(widget))
                        self.assertGreater(widget.winfo_height(), 10)
                        self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                             root.winfo_rooty() + height)
                    self.assertLessEqual(phase.winfo_rooty() + phase.winfo_height(), app.export_button.winfo_rooty())
            finally:
                root.destroy()

    def test_short_selection_reaches_recording_and_full_is_the_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            app = CaptureApp(root, directory)
            try:
                self.assertEqual(app.protocol_mode.get(), RESEARCH_PROTOCOLS[FULL_PROTOCOL])
                app.source_mode.set("模拟演示")
                app.protocol_mode.set(RESEARCH_PROTOCOLS[SHORT_PROTOCOL])
                app.start_recording()
                self.assertEqual(app.config.name, SHORT_PROTOCOL)
                self.assertEqual(app.store.metadata["protocol"], app.config.as_dict())
                self.assertEqual(app.store.metadata["source"]["data_origin"], "simulation")
            finally:
                if app.quality_monitor:
                    app.quality_monitor.stop()
                    app.quality_monitor.join(timeout=3)
                if app.recorder:
                    app.recorder.stop()
                    app.recorder.join(timeout=3)
                if app.store:
                    app.store.close("aborted")
                root.update_idletasks()
                root.destroy()

    def test_operator_preview_reports_live_quality_then_clears_on_disconnect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            app = CaptureApp(root, directory)
            app.clock = FakeClock()
            app.started_perf = app.clock.seconds
            app.store = SessionStore(directory, participant="PREVIEW", source=simulation_source(), protocol={}, clock=app.clock)
            root.attributes("-alpha", 0.)
            root.geometry("1200x780+10+10")
            root.deiconify()
            app.recorder = Mock(error=None)
            app.quality_monitor = QualityMonitor(app.store.path)
            try:
                root.update()
                app.last_preview = -float("inf")
                points = range(1500)
                timestamps = [app.clock.seconds - 6 + i / 250 for i in points]
                app.store.append(synthetic_samples(points), timestamps, offset=0.)
                app.quality_monitor.start()
                deadline = time.monotonic() + 4
                while not app.quality_monitor.latest.cleaned_samples and time.monotonic() < deadline:
                    time.sleep(.02)
                app._poll_once()
                self.assertIn("模拟源 · 4 秒窗质控  通过", app.quality_text.get())
                self.assertTrue(app.canvas.find_all())
                self.assertEqual(app.wave_mode.get(), "处理后 · µV")
                app.wave_mode.set("原始 · 源数值")
                app.clock.advance(.3)
                app._poll_once()
                self.assertTrue(any("源数值" in app.canvas.itemcget(tag, "text")
                                    for tag in app.canvas.find_all() if app.canvas.type(tag) == "text"))
                self.assertIsNone(app.task_window)
                app.clock.advance(3.1)
                app._poll_once()
                self.assertIn("信号超过 3 秒未更新", app.quality_text.get())
            finally:
                app.quality_monitor.stop()
                app.quality_monitor.join(timeout=3)
                app.store.close("aborted")
                root.update_idletasks()
                root.destroy()

    def test_selected_biomulti_uses_inferred_unit_and_packet_clock_without_native_rec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            app = CaptureApp(root, directory)
            meta = {**simulation_source(), "kind": "lsl", "type": "EEG", "data_origin": "real",
                    "name": "BioMulti Lite EEG-00abcd", "hostname": socket.gethostname(),
                    "declared_units": ["", ""], "source_unit": "unknown", "scale_to_uv": None}
            app.streams = [(None, meta)]
            app.stream_combo["values"] = [meta["name"]]
            app.stream_combo.current(0)
            try:
                with patch("ningsi_capture.biomulti.local_publisher", return_value=True), \
                        patch("ningsi_capture.biomulti.active_record", side_effect=ValueError("no active REC")), \
                        patch("ningsi_capture.gui.LslRecorder"), patch("ningsi_capture.gui.QualityMonitor"):
                    app.on_select_stream()
                    self.assertEqual((app.units.get(), app.timing_mode.get()), (AUTO_UNIT, AUTO_REC))
                    self.assertIn("预试", app.eligibility_text.get())
                    self.assertIn("Fp1、Fp2", app.eligibility_text.get())
                    app.cohort.set("开发组")
                    app.labels.set("Fp2, Fp1")
                    app.reference.set("verified-common-reference")
                    self.assertIn("合格探针用于训练", app.eligibility_text.get())
                    self.assertNotIn("待核对", app.eligibility_text.get())
                    self.assertNotIn("需按实际顺序", app.eligibility_text.get())
                    app.start_recording()
                selected = app.store.metadata["source"]
                self.assertEqual((selected["source_unit"], selected["scale_to_uv"]), ("uV", 1.))
                self.assertEqual(selected["unit_basis"], "biomulti_software_inference")
                self.assertEqual(selected["timestamp_policy"], BIOMULTI_LSL_TIMING)
                self.assertNotIn("native_rec", selected)
            finally:
                if app.store:
                    app.store.close("aborted")
                root.destroy()

    def test_operator_can_recheck_a_flat_lsl_signal_before_task_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            app = CaptureApp(root, directory)
            source = {**simulation_source(), "kind": "lsl", "scale_to_uv": None}
            app.clock = FakeClock()
            app.store = SessionStore(directory, participant="CHECK", source=source, protocol={}, clock=app.clock)
            try:
                times = [1000. + index / 250 for index in range(1000)]
                app.store.append([[0., 0.]] * len(times), times, offset=0.)
                with patch("ningsi_capture.gui.messagebox.showwarning") as prompt:
                    app.start_tasks()
                self.assertIn("单位", prompt.call_args.args[0])
                self.assertIsNone(app.task_window)
                self.assertIsNone(app.runner)
                self.assertFalse(app.store.closed)
                self.assertEqual(app.store.sample_count, len(times))
            finally:
                app.store.close("aborted")
                root.destroy()

    def test_external_abort_clears_stale_stage_without_another_runner_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            app = CaptureApp(root, directory)
            app.clock = FakeClock()
            app.store = SessionStore(directory, participant="ABORT", source=simulation_source(), protocol={}, clock=app.clock)
            app.recorder = Mock(error=None)
            app.started_perf = app.clock.seconds
            app.runner = TaskRunner(app.store, lambda view: app.clock.now(), ProtocolConfig.short(), app.clock)
            try:
                app.runner.start()
                app.update_task_status()
                self.assertIn("剩余 02:00", app.phase_text.get())
                app.runner.abort("participant_window_closed")
                app._poll_once()
                self.assertIn("任务已中止", app.phase_text.get())
                self.assertIn("窗口已关闭", app.phase_text.get())
                self.assertEqual(app.participant_time.get(), "")
                self.assertIn("先停止保存", app.task_button.cget("text"))
                self.assertIn("重新采集", app.stop_button.cget("text"))
                self.assertFalse(app.store.closed)
            finally:
                app.store.close("aborted")
                root.destroy()

    def test_both_windows_time_every_phase_including_untimed_practice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            app = CaptureApp(root, directory)
            app.clock = FakeClock()
            app.config = ProtocolConfig.demo()
            app.store = SessionStore(directory, participant="TIMER", source=simulation_source(),
                                     protocol=app.config.as_dict(), clock=app.clock)
            try:
                app.store.append(synthetic_samples(range(1000)), [1000 + i / 250 for i in range(1000)], offset=0.)
                app.start_tasks()
                runner = app.runner
                clock_label = next(child for child in app.task_window.winfo_children()
                                   if isinstance(child, tk.Label) and str(child.cget("textvariable")) == str(app.participant_time))
                app.task_window.geometry("850x650")
                seen = set()
                while runner.running:
                    phase = runner.phase
                    seen.add(phase["name"])
                    stage = f"阶段 {runner.phase_index + 1}/{len(runner.plan)}"
                    self.assertIn(phase["title"], app.phase_text.get())
                    self.assertIn(stage, clock_label.cget("text"))
                    elapsed_and_remaining = clock_label.cget("text").split(" · ", 1)[1]
                    self.assertIn(elapsed_and_remaining, app.phase_text.get())
                    self.assertIn("已用 ", elapsed_and_remaining)
                    if phase["kind"] == "practice_probe":
                        app.clock.advance(65)
                        runner.tick()
                        app.update_task_status()
                        self.assertIn("已用 01:05 · 不限时，作答后继续", clock_label.cget("text"))
                        app.task_window.update_idletasks()
                        choices = app.choice_frame.winfo_children()
                        self.assertEqual(len(choices), 7)
                        self.assertGreaterEqual(choices[0].winfo_rooty(), clock_label.winfo_rooty() + clock_label.winfo_height())
                        self.assertLessEqual(choices[-1].winfo_rooty() + choices[-1].winfo_height(), app.task_footer.winfo_rooty())
                        runner.key("3")
                    else:
                        self.assertIn("预计剩余" if phase["kind"] == "sart" else "剩余 ", elapsed_and_remaining)
                        advance_to = runner.deadline
                        if phase["kind"] == "pvt":
                            advance_to = min(advance_to, runner.phase_deadline)
                        app.clock.seconds = advance_to
                        runner.tick()
                    app.update_task_status()
                    if runner.running and runner.phase is not phase:
                        self.assertIn("已用 00:00", clock_label.cget("text"))
                self.assertEqual(seen, {phase["name"] for phase in runner.plan})
                self.assertTrue(runner.finished)
                self.assertEqual(clock_label.cget("text"), "")
                self.assertIn("任务已完成", app.phase_text.get())
            finally:
                app.close_tasks(force=True)
                app.store.close("aborted")
                root.destroy()

    def test_cancel_close_keeps_task_running_and_confirmed_close_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            app = CaptureApp(root, directory)
            app.clock = FakeClock()
            app.store = SessionStore(directory, participant="CLOSE", source=simulation_source(), protocol={}, clock=app.clock)
            app.task_window = Mock()
            app.runner = TaskRunner(app.store, lambda view: app.clock.now(), ProtocolConfig.short(), app.clock)
            try:
                app.runner.start()
                with patch("ningsi_capture.gui.messagebox.askyesno", return_value=False):
                    app.close_tasks()
                self.assertTrue(app.runner.running)
                self.assertFalse(app.task_close_pending)
                app.task_window.destroy.assert_not_called()
                with patch("ningsi_capture.gui.messagebox.askyesno", return_value=True):
                    app.close_tasks()
                self.assertEqual(app.runner.abort_reason, "participant_window_closed")
                self.assertIsNone(app.task_window)
                self.assertIn("任务已中止", app.phase_text.get())
            finally:
                app.store.close("aborted")
                root.destroy()

    def test_failed_timing_precheck_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            app = CaptureApp(root, directory)
            app.clock = FakeClock()
            source = {**simulation_source(), "kind": "lsl"}
            app.store = SessionStore(directory, participant="TIMING", source=source, protocol={}, clock=app.clock)
            try:
                app.store.append(synthetic_samples(range(1000)), [1000. + i / 250 for i in range(1000)], offset=0.)
                with patch("ningsi_capture.gui.recent_window_quality", return_value=("nonmonotonic_timestamps",)), \
                        patch("ningsi_capture.gui.messagebox.showwarning") as warning:
                    app.start_tasks()
                self.assertIn("时间戳", warning.call_args.args[1])
                self.assertIsNone(app.runner)
                self.assertIsNone(app.task_window)
            finally:
                app.store.close("aborted")
                root.destroy()

    def test_fresh_pilot_records_tasks_despite_unit_and_packet_qc_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            app = CaptureApp(root, directory)
            app.clock = FakeClock()
            app.config = ProtocolConfig.short()
            source = {**simulation_source(), "kind": "lsl", "scale_to_uv": None}
            app.store = SessionStore(directory, participant="PILOT", cohort="pilot", source=source,
                                     protocol=app.config.as_dict(), clock=app.clock)
            try:
                app.store.append(synthetic_samples(range(1000)), [1000.] * 1000, offset=0.)
                with closing(open_readonly(app.store.path)) as db:
                    app.quality_monitor = Mock(latest=CaptureQuality().update(db, app.store.metadata))
                app.start_tasks()
                self.assertTrue(app.runner.running)
                with closing(open_readonly(app.store.path)) as db:
                    event = db.execute("SELECT payload FROM events WHERE kind='task_signal_precheck'").fetchone()[0]
                    self.assertIn("unit_unconfirmed", event)
                    self.assertIn("nonmonotonic_timestamps", event)
                    saved = json.loads(event)["signal_preflight"]
                    self.assertEqual((saved["sample_end_index"], saved["recent_usable"]), (999, 0))
                    self.assertEqual(len(saved["line_power_ratios"]), 2)
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE kind='protocol_start'").fetchone()[0], 1)
            finally:
                app.close_tasks(force=True)
                app.store.close("aborted")
                root.destroy()

    def test_formal_paired_start_requires_same_active_rec_but_accepts_deferred_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            app = CaptureApp(root, directory)
            app.clock = FakeClock()
            app.config = ProtocolConfig.short()
            binding = {"record_id": 3, "created_ms": 47}
            source = {**simulation_source(), "kind": "lsl", "scale_to_uv": None,
                      "timestamp_policy": BIOMULTI_REC_TIMING, "native_rec": binding}
            app.store = SessionStore(directory, participant="PAIR", cohort="development", source=source,
                                     protocol=app.config.as_dict(), clock=app.clock)
            try:
                app.store.append(synthetic_samples(range(1500)), [1000.] * 1500, offset=0.)
                with patch("ningsi_capture.gui.biomulti.active_record", return_value={"record_id": 4}), \
                        patch("ningsi_capture.gui.messagebox.showwarning") as warn:
                    app.start_tasks()
                    self.assertTrue(warn.called)
                    self.assertIsNone(app.runner)
                with patch("ningsi_capture.gui.biomulti.active_record", return_value=binding):
                    app.start_tasks()
                self.assertTrue(app.runner.running)
            finally:
                app.close_tasks(force=True)
                app.store.close("aborted")
                root.destroy()

    def test_callback_failure_stops_capture_and_saves_failed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            try:
                app = CaptureApp(root, directory)
                app.source_mode.set("模拟演示")
                app.protocol_mode.set("快速模拟联调")
                app.start_recording()
                app.on_callback_error(ValueError, ValueError("injected_test_error"), None)
                deadline = time.perf_counter() + 4
                while not app.store.closed and time.perf_counter() < deadline:
                    root.update()
                    time.sleep(.01)
                self.assertTrue(app.store.closed)
                connection = open_readonly(app.last_session)
                try:
                    metadata = read_metadata(connection)
                    self.assertEqual(metadata["state"], "failed")
                    self.assertIn("injected_test_error", metadata["reason"])
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM events WHERE kind='application_callback_error'").fetchone()[0], 1)
                finally:
                    connection.close()
            finally:
                root.update_idletasks()
                root.destroy()

    def test_callback_error_is_visible_and_does_not_escape_gui(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            try:
                app = CaptureApp(root, directory)
                app.on_callback_error(ValueError, ValueError("test failure"), None)
                self.assertIn("test failure", app.callback_error)
                self.assertIn("程序异常", app.status.get())
            finally:
                root.update_idletasks()
                root.destroy()

    def test_stop_controls_fit_at_minimum_supported_height(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            try:
                app = CaptureApp(root, directory)
                root.geometry("1100x700")
                root.update_idletasks()
                bottom = root.winfo_rooty() + root.winfo_height()
                for button in (app.start_button, app.task_button, app.stop_button):
                    self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), bottom)
            finally:
                root.update_idletasks()
                root.destroy()


if __name__ == "__main__":
    unittest.main()
