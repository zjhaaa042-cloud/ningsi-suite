import unittest
from ningsi.assessment.diagnostics import summarize_windows, render_html
from ningsi.assessment.joint import assess

class DiagnosticsTests(unittest.TestCase):
    def test_sensitivity_does_not_remove_other_rejections(self):
        rows = [
            {'usable': False, 'reasons': ['amplitude'], 'quality_metrics': {'peak_uv': 200}},
            {'usable': False, 'reasons': ['amplitude', 'eog'], 'quality_metrics': {'peak_uv': 200}},
            {'usable': False, 'reasons': ['signal_too_short']},
        ]
        d = summarize_windows(rows)
        self.assertEqual(d['accepted'], 0)
        self.assertEqual(d['amplitude_sensitivity'][1]['accepted'], 1)
        self.assertEqual(d['amplitude_sensitivity'][1]['evaluated'], 2)

    def test_rejected_spectra_never_mix_into_accepted(self):
        d = summarize_windows([
            {'usable': True, 'relative_band_powers': {'alpha': .2}, 'reasons': []},
            {'usable': False, 'relative_band_powers': {'alpha': .9}, 'reasons': ['amplitude','eog']},
            {'usable': False, 'relative_band_powers': {}, 'reasons': ['no_data']},
        ])
        self.assertEqual(d['accepted'], 1)
        self.assertEqual(d['bands']['accepted']['alpha']['mean'], .2)
        self.assertEqual(d['bands']['rejected']['alpha']['mean'], .9)
        self.assertEqual(d['bands']['rejected']['alpha']['n'], 1)
        self.assertIsNone(d['bands']['accepted']['beta']['mean'])

    def test_no_baseline_still_has_honest_visual_report(self):
        html = render_html({'extras': {'signal_diagnostics': summarize_windows([])},
                            'assessment': {'advice': ['<script>alert(1)</script>']}})
        self.assertIn('没有可用个体脑电评分', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>', html)
        self.assertIn('行为表现', html)

    def test_unavailable_eeg_never_recommends_baseline_training(self):
        result = assess({}, behavior={'sart': {'valid': True, 'commission_rate': .8}})
        self.assertTrue(any('重新采集基线' in a for a in result.advice))
        self.assertFalse(any('按个体基线中位数设置训练目标' in a for a in result.advice))
