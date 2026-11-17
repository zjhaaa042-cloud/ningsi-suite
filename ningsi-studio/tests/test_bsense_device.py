"""The supplied BSense-R manual configures the device without a profile form."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from ningsi_studio.core import live_source
from ningsi_studio.domain import bsense_device, eeg_units
from ningsi_studio.domain.preflight import evaluate_preflight


NAME = "BioMulti Lite EEG-00ac31"
DEVICE = "lsl:" + NAME
DESCRIPTOR = {"name": NAME, "kind": "eeg", "stream_type": "EEG", "supported": True,
              "channel_count": 2, "nominal_srate": 250., "channel_labels": ["Ch1", "Ch2"],
              "declared_units": ["", ""], "source_id": "manufacturer-device-stream"}


class BsenseDeviceTests(unittest.TestCase):
    def test_current_vendor_eeg_uses_documented_units_without_inventing_montage(self):
        descriptor = deepcopy(DESCRIPTOR)
        configured = bsense_device.configuration(descriptor)
        self.assertEqual((configured["channels"], configured["srate"], configured["source_unit"],
                          configured["scale_to_uv"]), (2, 250., "uV", 1.))
        self.assertEqual(configured["origin"], "manufacturer_manual")
        self.assertNotIn("reference", configured)
        self.assertNotIn("channel_labels", configured)
        self.assertEqual(descriptor, DESCRIPTOR)
        self.assertEqual(eeg_units.resolve_calibration(None, descriptor),
                         {"source_unit": "uV", "scale_to_uv": 1., "origin": "manufacturer_manual"})
        self.assertEqual(descriptor["channel_labels"], ["Ch1", "Ch2"])

    def test_scan_row_uses_actual_nominal_metadata_instead_of_display_fallback(self):
        row = {"key": DEVICE, "device": NAME, "kind": "lsl", "stream_kind": "eeg", "real": True,
               "srate": 250., "channels": 2, "declared_units": ["µV", "μV"]}
        self.assertEqual(eeg_units.resolve_calibration(None, row)["origin"], "manufacturer_manual")
        for actual in (None, 0, 249.65, float("nan")):
            with self.subTest(actual=actual), self.assertRaisesRegex(ValueError, "250 Hz"):
                eeg_units.resolve_calibration(None, {**row, "nominal_srate": actual})

    def test_known_vendor_with_conflicting_rate_or_channels_does_not_fall_back(self):
        for change in ({"channel_count": 4}, {"channel_count": 2.5}, {"channel_count": True},
                       {"channel_count": None}, {"nominal_srate": 500}, {"nominal_srate": True}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "BSense-R"):
                eeg_units.resolve_calibration(None, {**DESCRIPTOR, **change})

    def test_live_units_and_adc_are_never_silently_overridden(self):
        for units in (["mV", "mV"], ["uV", "mV"], ["ADC", "ADC"], ["unknown-unit", ""]):
            with self.subTest(units=units), self.assertRaisesRegex(ValueError, "当前流声明"):
                eeg_units.resolve_calibration(None, {**DESCRIPTOR, "declared_units": units})
        for selection in ({"source_unit": "mV"}, {"source_unit": "ADC", "scale_to_uv": 1.},
                          {"source_unit": "ADC", "scale_to_uv": .1}):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "BSense-R"):
                eeg_units.resolve_calibration(selection, DESCRIPTOR)

    def test_client_origin_never_establishes_document_evidence(self):
        supplied = {"source_unit": "uV", "origin": "manufacturer_manual", "scale_to_uv": 1.}
        self.assertEqual(eeg_units.resolve_calibration(supplied, DESCRIPTOR)["origin"], "manufacturer_manual")
        other = {**DESCRIPTOR, "name": "Another EEG"}
        self.assertIsNone(eeg_units.resolve_calibration(None, other))
        self.assertEqual(eeg_units.resolve_calibration(supplied, other)["origin"], "operator_confirmed")
        self.assertEqual(eeg_units.confirmation(supplied)["origin"], "operator_confirmed")

    def test_other_device_and_secondary_or_simulated_streams_do_not_inherit_defaults(self):
        variants = [
            {**DESCRIPTOR, "name": "Other EEG"},
            {**DESCRIPTOR, "kind": "metric"},
            {**DESCRIPTOR, "name": "BioMulti Lite Metric-00ac31"},
            {**DESCRIPTOR, "name": "BioMulti Lite EEG Metric-00ac31"},
            {**DESCRIPTOR, "source_id": "synthetic-eeg"},
            {**DESCRIPTOR, "simulated": True},
            {**DESCRIPTOR, "real": False},
            {**DESCRIPTOR, "supported": False},
        ]
        for descriptor in variants:
            with self.subTest(descriptor=descriptor):
                self.assertIsNone(bsense_device.configuration(descriptor))
                self.assertIsNone(eeg_units.resolve_calibration(None, descriptor))
        for name in ("BioMultiLite-EEG-123", "BSense-R EEG", "BSense R EEG-123"):
            self.assertIsNotNone(bsense_device.configuration({**DESCRIPTOR, "name": name}))


class BsensePreflightTests(unittest.TestCase):
    def check(self, descriptor=DESCRIPTOR, **payload):
        return evaluate_preflight({"device": DEVICE, **payload}, descriptors=[descriptor],
                                  device_profile={"saved": False, "missing_fields": [{"key": "reference"}]})

    def test_documented_configuration_is_ready_without_operator_forms(self):
        result = self.check()
        self.assertTrue(result["ready"])
        self.assertEqual(result["calibration"], {"source_unit": "uV", "scale_to_uv": 1.,
                                                  "origin": "manufacturer_manual"})
        self.assertEqual(result["device_configuration"]["srate"], 250.)
        self.assertFalse(any(row["key"] == "device_profile" for row in result["checks"]))
        unit = next(row for row in result["checks"] if row["key"] == "units")
        self.assertEqual(unit["status"], "pass")
        self.assertIn("250 Hz", unit["message"])
        self.assertIn("µV", unit["message"])
        self.assertEqual(next(row for row in result["checks"] if row["key"] == "quality")["status"], "pending")

    def test_actual_stream_conflict_remains_blocked_and_has_no_default_confirmation(self):
        for change in ({"nominal_srate": 500}, {"channel_count": 4}, {"declared_units": ["mV", "mV"]}):
            with self.subTest(change=change):
                result = self.check({**DESCRIPTOR, **change})
                self.assertFalse(result["ready"])
                self.assertIsNone(result["calibration"])
                self.assertEqual(next(row for row in result["checks"] if row["key"] == "units")["status"], "fail")

    def test_legacy_profile_is_not_a_requirement_or_a_unit_override(self):
        other = {**DESCRIPTOR, "name": "Other EEG"}
        result = evaluate_preflight({"device": "lsl:Other EEG"}, descriptors=[other],
            device_profile={"saved": True, "profile": {"source_unit": "uV", "scale_to_uv": 1.},
                            "missing_fields": [{"key": "reference"}]})
        self.assertFalse(result["ready"])
        self.assertIsNone(result["calibration"])
        self.assertFalse(any(row["key"] == "device_profile" for row in result["checks"]))


class BsenseScanTests(unittest.TestCase):
    def sources(self, descriptor):
        with patch('ningsi_studio.acquisition.lsl.probe_streams', return_value=[descriptor]):
            return live_source.list_available()

    def test_scan_exposes_the_same_parsed_device_configuration_for_vendor_names(self):
        for name in (NAME, 'BSense-R EEG-Review'):
            descriptor = {**DESCRIPTOR, 'name': name}
            with self.subTest(name=name):
                row = self.sources(descriptor)[0]
                self.assertEqual(row['device_configuration'], bsense_device.configuration(descriptor))
                self.assertEqual(row['device_configuration']['source_unit'], 'uV')

    def test_scan_never_marks_conflicting_unknown_or_simulated_sources_as_configured(self):
        for change in ({'nominal_srate': 0}, {'nominal_srate': None},
                       {'nominal_srate': 256}, {'channel_count': 4},
                       {'declared_units': ['mV', 'mV']}, {'name': 'Other EEG'},
                       {'source_id': 'synthetic-eeg'}):
            descriptor = {**DESCRIPTOR, **change}
            with self.subTest(change=change):
                row = self.sources(descriptor)[0]
                self.assertIsNone(row['device_configuration'])
                self.assertEqual(row['nominal_srate'], descriptor['nominal_srate'])
                self.assertEqual(row['declared_units'], descriptor['declared_units'])


if __name__ == "__main__":
    unittest.main()
