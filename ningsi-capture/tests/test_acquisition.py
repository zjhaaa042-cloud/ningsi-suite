import tempfile
import unittest
from unittest.mock import Mock, patch

from .test_support import FakeClock
from ningsi.acquisition.eeg import unit_conflict
from ningsi_capture.acquisition import LslRecorder, describe, discover, simulation_source
from ningsi_capture.storage import SessionStore, open_readonly


class EmptyNode:
    def child(self, _name):
        return self

    def empty(self):
        return True


class Info:
    def __init__(self, name="Test EEG", channels=2):
        self._name, self._channels = name, channels

    def name(self): return self._name
    def type(self): return "EEG"
    def channel_count(self): return self._channels
    def nominal_srate(self): return 250.
    def channel_format(self): return 1
    def source_id(self): return "test-source"
    def uid(self): return "test-uid"
    def hostname(self): return "test-host"
    def desc(self): return EmptyNode()
    def as_xml(self): return "<info><name>test</name></info>"


class AcquisitionTests(unittest.TestCase):
    def test_discovery_uses_stream_type_and_numeric_format(self):
        from pylsl import StreamInfo
        infos = [StreamInfo(name, kind, 2, rate, fmt, "test-id") for name, kind, rate, fmt in (
            ("BioMulti Lite EEG", "EEG", 250, 1), ("Vendor EEG", "", 250, 2),
            ("QA EEG Markers", "Markers", 250, 4), ("EEG", "Motion", 250, 1),
            ("EEG Quality", "EEG", 250, 1), ("EEG", "EEG", 250, 3), ("EEG", "EEG", 0, 4))]
        with patch("pylsl.resolve_streams", return_value=infos):
            self.assertEqual([s["name"] for _, s in discover()], ["BioMulti Lite EEG", "Vendor EEG"])

    def test_declared_units_checked_before_any_sample_is_received(self):
        from pylsl import StreamInfo
        short = StreamInfo("Device EEG", "EEG", 2, 250., 1, "test-inlet")
        full = StreamInfo("Device EEG", "EEG", 2, 250., 1, "test-inlet")
        channels = full.desc().append_child("channels")
        for label in ("F3", "F4"):
            node = channels.append_child("channel")
            node.append_child_value("label", label)
            node.append_child_value("unit", "mV")
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            clock.domain = "lsl_local_clock"
            source = {**describe(short), "source_unit": "uV", "scale_to_uv": 1.}
            store = SessionStore(directory, participant="QA", source=source, protocol={}, clock=clock)
            try:
                inlet = Mock()
                inlet.info.return_value = full
                recorder = LslRecorder(store, short, inlet_factory=lambda _: inlet)
                recorder.run()
                self.assertIn("mV", recorder.error)
                inlet.pull_chunk.assert_not_called()
                inlet.close_stream.assert_called_once()
                self.assertFalse(recorder.ready.is_set())
                self.assertEqual(store.sample_count, 0)
            finally:
                store.close("failed")

    def test_device_units_aliases_adc_and_mixed_channels(self):
        for declared, chosen, factor, conflict in [
                (["µV", "μV", "microvolts"], "uV", 1., False),
                (["millivolts", ""], "mV", 1000., False),
                (["V"], "uV", 1., True), (["uV", "mV"], "uV", 1., True),
                (["counts"], "ADC 计数", .125, False), (["counts"], "uV", 1., True),
                (["uV"], "ADC 计数", 1., True), (["mV"], "未知", None, False),
                (["", ""], "ADC 计数", .125, False)]:
            with self.subTest(declared=declared, chosen=chosen):
                self.assertEqual(bool(unit_conflict({"declared_units": declared, "source_unit": chosen,
                                                      "scale_to_uv": factor})), conflict)

    def test_missing_electrode_labels_are_not_fabricated_as_fp1(self):
        self.assertEqual(describe(Info())["channel_labels"], ["Ch1", "Ch2"])

    def test_obvious_simulated_lsl_stream_is_not_marked_real(self):
        self.assertEqual(describe(Info("ningsi-sim-eeg"))["data_origin"], "simulation_lsl")

    def test_lsl_samples_preserve_raw_timestamps_and_close_inlet(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            clock.domain = "lsl_local_clock"
            source = describe(Info("synthetic-test-eeg"))
            store = SessionStore(directory, participant="TEST", source=source, protocol={}, clock=clock)
            class Inlet:
                closed = False
                calls = 0
                def open_stream(self, timeout): pass
                def info(self, timeout): return Info("synthetic-test-eeg")
                def time_correction(self, timeout): return .125
                def pull_chunk(self, timeout, max_samples):
                    self.calls += 1
                    if self.calls == 2:
                        recorder.stop()
                    clock.advance(.01)
                    return [[1., 2.]], [999.875 + self.calls * .004]
                def close_stream(self): self.closed = True
            inlet = Inlet()
            recorder = LslRecorder(store, Info(), inlet_factory=lambda info: inlet)
            recorder.start()
            recorder.join(3)
            self.assertFalse(recorder.is_alive())
            self.assertIsNone(recorder.error)
            self.assertTrue(inlet.closed)
            store.close("aborted")
            reader = open_readonly(store.path)
            rows = reader.execute("SELECT * FROM samples ORDER BY sample_index").fetchall()
            reader.close()
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[0]["source_timestamp"], 999.879)
            self.assertAlmostEqual(rows[0]["corrected_timestamp"], 1000.004)


if __name__ == "__main__":
    unittest.main()
