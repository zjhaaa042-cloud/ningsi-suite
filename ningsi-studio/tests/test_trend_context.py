"""Trend cohorts must not mix protocols, simulated pacing or data sources."""
from pathlib import Path

from .helpers import StudioTestCase
from ningsi_studio.core import paired_ledger
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.settings import Settings


class TrendContextTests(StudioTestCase):
    @classmethod
    def make_settings(cls, data_dir):
        return Settings(data_dir=Path(data_dir), port=0)

    def setUp(self):
        super().setUp()
        self.subject = self.create_subject()

    def seed(self, *, protocol='full', mode='quick', scale=1., source='sim-bsense',
             score=.7, status='done', field='focus'):
        with store.connect(self.db_path) as conn:
            subject = repo.find_subject(conn, self.subject['public_id'])
            row = repo.create_session(conn, subject['id'], device='sim-bsense',
                                      protocol=protocol, training_mode=mode, time_scale=scale)
            repo.update_session(conn, row['uuid'], source=source, status=status, phase=status)
            if score is not None:
                repo.add_metric(conn, row['id'], 'indicator_summary', field, mean=score,
                                n=35, quality={'phase': 'monitor'})
            row = dict(repo.get_session(conn, row['uuid']))
        # Legacy ledger rows deliberately lack comparison context: the service
        # must recover it from this completed session, not assume full/1x.
        paired_ledger.append_history_record(self.app.settings.runs_root / row['uuid'], {
            'session_uuid': row['uuid'], 'participant': self.subject['public_id'],
            'device': row['device'], 'srate': row['srate'], 'channels': row['channels'],
            'indicators': {} if score is None else {field: score},
        })
        return row

    def trend(self, suffix=''):
        return self.json_body(self.get('/api/reports/trend?participant=' + self.subject['public_id'] + suffix))

    def test_mixed_protocols_sources_modes_and_pacing_do_not_enter_one_mean(self):
        self.seed(protocol='short', score=.1)
        self.seed(mode='full', score=.2)
        self.seed(scale=.03, score=.3)
        self.seed(source='lsl:QA', score=.4)
        self.seed(score=.9)
        self.seed(score=.7)
        # The latest completed session has no monitor score and cannot choose
        # the comparison group; cancelled records with ledger files are ignored.
        self.seed(protocol='effective_time', score=None)
        self.seed(score=1., status='cancelled')
        result = self.trend()
        self.assertEqual((result['sessions'], result['comparable'], result['rejected']), (7, 2, 5))
        self.assertEqual(result['points'][0]['mean'], .8)
        self.assertEqual(result['points'][0]['n'], 2)
        self.assertEqual(result['ledger_points'][0]['mean'], .8)
        self.assertEqual(result['ledger_points'][0]['n'], 2)
        self.assertEqual(result['rejected_reasons'], {'metric_unavailable': 1, 'source_kind_changed': 1,
            'time_scale_changed': 1, 'training_mode_changed': 1, 'protocol_changed': 1})
        self.assertEqual(result['comparison_group']['protocol'], 'full')
        self.assertEqual(result['comparison_group']['time_scale'], 1.)

    def test_session_trend_uses_selected_session_context(self):
        short = self.seed(protocol='short', score=.1)
        self.seed(score=.9)
        result = self.json_body(self.get(f"/api/sessions/{short['uuid']}/trend"))
        self.assertEqual(result['points'][0]['mean'], .1)
        self.assertEqual(result['comparable'], 1)
        self.assertEqual(result['comparison_group']['protocol'], 'short')

    def test_reference_is_selected_per_metric_and_empty_metric_counts_zero(self):
        self.seed(protocol='short', score=.2, field='focus')
        self.seed(protocol='full', score=.8, field='relax')
        focus = self.trend()
        relax = self.trend('&field=relax')
        self.assertEqual(focus['points'][0]['mean'], .2)
        self.assertEqual(focus['comparison_group']['protocol'], 'short')
        self.assertEqual(relax['points'][0]['mean'], .8)
        self.assertEqual(relax['comparison_group']['protocol'], 'full')
        empty = self.trend('&field=load')
        self.assertEqual((empty['points'], empty['ledger_points'], empty['comparable']), ([], [], 0))
        self.assertIsNone(empty['comparison_group'])

    def test_rebuilt_ledger_retains_comparison_context(self):
        row = self.seed(protocol='short', scale=.03)
        with store.read_only(self.db_path) as conn:
            exported = next(item for item in repo.all_sessions_for_ledger(conn)
                            if item['session_uuid'] == row['uuid'])
        self.assertEqual((exported['protocol'], exported['training_mode'], exported['source'], exported['time_scale']),
                         ('short', 'quick', 'sim-bsense', .03))
