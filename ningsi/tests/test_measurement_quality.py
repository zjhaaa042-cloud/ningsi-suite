"""测量有效性回归：局部伪迹、重叠基线、无效行为证据。"""
import unittest
from dataclasses import replace

import numpy as np

from ningsi.behavior.pvt import PvtResult, PvtTrial
from ningsi.assessment.joint import assess
from ningsi.signal.window import analyze_window
from ningsi.signal.baseline import build_baseline
from ningsi.signal.indicators import compute_indicators


class MeasurementQualityTests(unittest.TestCase):
    def clean(self):
        t = np.arange(1000) / 250
        return 10 * np.sin(2 * np.pi * 10 * t)

    def test_second_channel_artifacts_are_not_hidden_by_first(self):
        t = np.arange(1000) / 250
        for frequency, reason in ((35, 'emg'), (2, 'eog')):
            with self.subTest(reason=reason):
                window = analyze_window(np.array([self.clean(), 10 * np.sin(2*np.pi*frequency*t)]), 250)
                self.assertFalse(window.usable)
                self.assertIn(reason, window.quality.reasons)
                self.assertEqual(len(window.quality.metrics['channels']), 2)

    def test_invalid_samples_do_not_produce_scores(self):
        for data in (np.array([]), np.full(1000, np.nan), self.clean()[:500]):
            self.assertFalse(analyze_window(data, 250).usable)

    def test_baseline_requires_duration_and_quality_and_tracks_drift(self):
        window = analyze_window(self.clean(), 250)
        duplicate = build_baseline([replace(window, t_end=4)] * 20)
        self.assertFalse(duplicate.valid)
        self.assertEqual(duplicate.quality['usable_seconds'], 4)
        windows = [replace(window, t_end=4+2*i) for i in range(15)]
        baseline = build_baseline(windows)
        self.assertTrue(baseline.valid)
        self.assertEqual(baseline.quality['usable_seconds'], 32)
        self.assertEqual(baseline.quality['half_mean_difference']['focus'], 0)
        bad = analyze_window(np.full(1000, np.nan), 250)
        self.assertFalse(build_baseline(windows + [bad] * 20).valid)
        changed = replace(window, channels=2)
        self.assertFalse(compute_indicators(changed, baseline).available)

    def test_pvt_boundaries_missing_and_invalid(self):
        trials = [PvtTrial(i, True, rt) for i, rt in enumerate((.1,.2,.3,.355,.5,.501,.09))]
        trials += [PvtTrial(8), PvtTrial(9, False, None, True)]
        result = PvtResult(trials).score()
        self.assertTrue(result['valid'])
        self.assertEqual(result['responded'], 6)
        self.assertEqual(result['false_starts'], 2)
        self.assertEqual(result['missed'], 1)
        self.assertEqual(result['lapses'], 1)
        self.assertEqual(result['lapses_355ms'], 3)
        self.assertEqual(result['rate_denominators']['omission_rate'], 7)
        trials.append(PvtTrial(10, True, float('nan')))
        invalid = PvtResult(trials).score()
        self.assertFalse(invalid['valid'])
        assessment = assess({}, behavior={'pvt': invalid})
        self.assertEqual(assessment.dimension_states['attention']['state'], '证据不足')

    def test_missing_responses_cannot_look_like_good_performance(self):
        result = PvtResult([PvtTrial(i, True, .2) for i in range(5)] + [PvtTrial(6), PvtTrial(7)]).score()
        assessment = assess({}, behavior={'pvt': result})
        self.assertIn('pvt_alertness', assessment.dimension_states['attention']['flagged'])

    def test_invalid_sart_is_not_normal_evidence(self):
        assessment = assess({}, behavior={'sart': {'valid': False, 'commission_rate': 0}})
        self.assertEqual(assessment.dimension_states['attention']['state'], '证据不足')
        self.assertFalse(next(e for e in assessment.evidence if e.code == 'sart_unavailable').available)
