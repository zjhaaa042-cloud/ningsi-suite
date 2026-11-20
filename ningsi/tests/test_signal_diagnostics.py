import unittest
from html.parser import HTMLParser
from ningsi.assessment.diagnostics import summarize_windows, render_html
from ningsi.assessment.joint import assess


class DisclosureText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.visible = []
        self.collapsed = []

    def handle_starttag(self, tag, attrs):
        if tag == 'details':
            if 'open' in dict(attrs):
                raise AssertionError('Report details must start collapsed')
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag == 'details':
            self.hidden -= 1

    def handle_data(self, data):
        (self.collapsed if self.hidden else self.visible).append(data)

class DiagnosticsTests(unittest.TestCase):
    def test_full_report_shows_results_and_keeps_scoring_and_diagnostics_collapsed(self):
        report = {
            'quality': {'windows_total': 20, 'windows_usable': 18},
            'indicators': {'summary': {'focus': {'mean': .76, 'n': 18}}},
            'scales': {'SAS': {'answered': 20, 'missing_items': [], 'raw_score': 45,
                               'standard_score': 56, 'level': '测试用分级'}},
            'behavior': {'sart': {'trials': 180, 'omission_errors': 1, 'commission_errors': 0},
                         'pvt': {'trials': 72, 'responded': 71, 'rt_median': .1981, 'missed': 1}},
            'assessment': {'conclusion': '不得把测试结论归于参与者'},
            'extras': {'source_kind': 'sim', 'report_context': {'response_mode': 'automated'},
                       'training': {'valid_seconds': 234, 'on_target_seconds': 118}},
        }
        page = DisclosureText()
        page.feed(render_html(report))
        public, hidden = ''.join(page.visible), ''.join(page.collapsed)
        self.assertEqual(page.hidden, 0)
        for value in ('18 / 20 窗', '0.76', '20 / 20 题', '198 毫秒', '71 / 72 次',
                      '180 次', '234 秒', '118 秒', '脑电由模拟器生成'):
            self.assertIn(value, public)
        for value in ('标准分 56', '测试用分级', '采集窗口质量', '频带组成', '不得把测试结论归于参与者'):
            self.assertIn(value, hidden)
            self.assertNotIn(value, public)

    def test_unavailable_scores_have_no_bars_and_invalid_tasks_stay_qualified(self):
        report = {'quality': {'windows_total': 9, 'windows_usable': 0},
                  'indicators': {'summary': {'focus': {'mean': 0, 'n': 0}}},
                  'behavior': {'pvt': {'trials': 8, 'responded': 0, 'valid': False}}}
        page = DisclosureText()
        page.feed(render_html(report))
        public = ''.join(page.visible)
        self.assertIn('0 / 9 窗', public)
        self.assertIn('中位反应时未记录', public)
        self.assertIn('未达到有效性要求', public)
        self.assertNotIn('专注相关指数0', public)

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

    def test_presented_replay_keeps_missing_updates_and_automated_responses_distinct(self):
        from html import unescape
        import re
        report = {'assessment': {'conclusion': '隐藏的联合结论'},
                  'behavior': {'pvt': {'trials': 4, 'responded': 4, 'missed': 0, 'rt_median': .315}},
                  'extras': {'source_kind': 'lsl',
                             'report_context': {'title': '测试 <2>', 'source_label': '本地录波回放 · 自动反应时任务',
                                                'response_mode': 'automated'},
                             'ten_min': {'segments': [{'index': 1, 'stats': {'valid_seconds': 0}, 'duration_sec': 50,
                                                       'local_eeg_feedback': {'support_seconds': 22.5}}],
                                         'indicator_overview': {'quality': {'windows_total': 25, 'windows_usable': 2}}}}}
        html = render_html(report)
        visible = unescape(re.sub(r'<[^>]+>', '', html.split('<details class="raw">')[0]))
        self.assertIn('测试 &lt;2&gt;', html)
        self.assertIn('本地录波回放 · 自动反应时任务', visible)
        self.assertIn('显示专注指数的更新未记录', visible)
        self.assertIn('通过严格质检的训练窗2 窗', visible)
        self.assertIn('有效反馈时长0 秒', visible)
        self.assertIn('315 毫秒', visible)
        self.assertNotIn('隐藏的联合结论', visible)
        report['extras'].pop('report_context')
        self.assertNotIn('本地录波回放', render_html(report))
