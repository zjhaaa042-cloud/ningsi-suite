"""Regress completion values, phase attribution, and read-only history recovery."""
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.domain.results import training_indicator_overview
from .helpers import StudioTestCase
from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]


def good_window(**patch):
    return {"phase": "training", "usable": True, "indicator_available": True,
            "graded": {"grade": "A"}, "scores": {"focus": .8, "relax": .3, "load": .6}, **patch}


class TrainingOverviewTests(unittest.TestCase):
    def test_only_complete_a_grade_training_scores_enter_summary(self):
        rows = [good_window(), good_window(scores={"focus": 0., "relax": 0., "load": 0.}),
                good_window(phase="baseline_after"), good_window(usable=False),
                good_window(graded={"grade": "B"}), good_window(graded={"grade": "C"}),
                good_window(indicator_available=False, indicator_reason="baseline_invalid"),
                good_window(scores={"focus": float("nan"), "relax": .3, "load": .6}),
                good_window(scores={"focus": .9})]
        before = copy.deepcopy(rows)
        overview = training_indicator_overview(rows)
        self.assertEqual(overview["summary"]["focus"], {"mean": .4, "std": .5657, "n": 2})
        self.assertEqual(overview["quality"]["windows_total"], 8)
        self.assertEqual(overview["quality"]["windows_usable"], 7)
        self.assertEqual(overview["quality"]["windows_scored"], 2)
        self.assertEqual(overview["excluded_reasons"]["grade_not_a"], 2)
        self.assertEqual(rows, before, "summarizing cannot mutate diagnostic evidence")
        json.dumps(overview, allow_nan=False)

    def test_no_baseline_or_windows_explains_missing_instead_of_scoring_zero(self):
        for rows, reason in (([], "没有保存"),
                             ([good_window(indicator_available=False, indicator_reason="baseline_invalid")], "基线无效")):
            overview = training_indicator_overview(rows)
            self.assertFalse(overview["available"])
            self.assertEqual(overview["summary"], {})
            self.assertIn(reason, overview["reason"])


class CompletionHistoryTests(StudioTestCase):
    port_base = 0

    def seed_session(self, *, persisted=False, report=True):
        overview = training_indicator_overview([good_window()])
        with store.connect(self.db_path) as conn:
            subject = repo.create_subject(conn, repo.next_public_id(conn))
            row = repo.create_session(conn, subject, protocol="ten_min")
            run = repo.start_run(conn, row["id"], "training")
            repo.finish_run(conn, run, payload={"indicator_overview": overview} if persisted else {})
            repo.update_session(conn, row["uuid"], status="done", phase="done")
            path = self.data_dir / (row["uuid"] + "_report.json")
            if report:
                path.write_text(json.dumps({"extras": {"signal_diagnostics": {"windows": [good_window()]}}}),
                                encoding="utf-8")
                repo.add_artifact(conn, row["id"], "report_json", str(path))
        return row["uuid"], path, overview

    def test_legacy_report_recovers_all_three_indices_without_rewriting_evidence(self):
        uuid, path, expected = self.seed_session()
        original = path.read_bytes()
        detail = self.session_detail(uuid)
        self.assertEqual(detail["result_overview"], expected)
        self.assertEqual(detail["indicator_summary"], {}, "training stays outside assessment metrics")
        self.assertEqual(path.read_bytes(), original)
        with store.read_only(self.db_path) as conn:
            row = repo.get_session(conn, uuid)
            self.assertEqual(json.loads(repo.list_runs(conn, row["id"])[0]["payload"]), {})

    def test_persisted_overview_survives_missing_report_and_service_restart(self):
        uuid, _, expected = self.seed_session(persisted=True, report=False)
        self.restart_service()
        self.assertEqual(self.session_detail(uuid)["result_overview"], expected)

    def test_unrecoverable_history_has_an_explicit_missing_reason(self):
        uuid, _, _ = self.seed_session(report=False)
        overview = self.session_detail(uuid)["result_overview"]
        self.assertFalse(overview["available"])
        self.assertTrue(overview["reason"])
        self.assertEqual(overview["summary"], {})


@unittest.skipUnless(shutil.which("node"), "Node required for rendered completion-card regression")
class CompletionCardTests(unittest.TestCase):
    def test_real_completion_card_renders_three_values_scope_and_missing_reason(self):
        source = (ROOT / "web/js/views/flow.js").read_text(encoding="utf-8")
        functions = "\n".join(re.search(r"const " + name + r" = \(\) => \{.*?\n  \};", source, re.S).group()
                              for name in ("qualityStats", "renderDoneCard"))
        overview = training_indicator_overview([good_window()])
        script = NODE_STUB + r'''
        import assert from 'node:assert/strict';
        const {DASH, el, card, button, pick, list, fmtNum, fmtInt, statusText} = await import(UTIL_URI);
        StubNode.prototype.append = function(...nodes) {this.children.push(...nodes);};
        const api = {exportZipUrl: () => '/zip'};
        const uuid = 'test';
        const all = node => [node, ...(node.children || []).flatMap(all)];
        function show(overview) {
          const local = {detail:{result_overview:overview, source:'sim-bsense', runs:[
            {phase:'qc',payload:{windows:15,usable:15,passed:true}}]}, scaleResults:new Map()};
          const ctx = {store:{state:{currentSession:{status:'done'}}},navigate(){}};
          let completedDetails;
          const [stepHost, stageHost, noticeHost, interactionHost] = [1,2,3,4].map(() => new StubNode('div'));
          FUNCTIONS
          renderDoneCard();
          const nodes = all(stepHost);
          const metrics = nodes.filter(n=>n.classList?.contains('done__metric')).map(n=>n.children.map(x=>x.textContent));
          return {metrics,text:nodes.map(n=>n.textContent).join('\n')};
        }
        const overview = OVERVIEW;
        const result = show(overview);
        assert.deepEqual(result.metrics.slice(0,3).map(r=>r[1]), ['0.80','0.30','0.60']);
        assert.equal(result.metrics[3][0], '训练可评分窗');
        assert.equal(result.metrics[3][1], '1/1 窗');
        assert.ok(result.text.includes('训练阶段A级窗口'));
        assert.ok(result.text.includes('启动质检 15/15 窗'));
        assert.ok(result.text.includes('仿真数据'));
        const missing = show({...overview,summary:{},available:false,reason:'睁眼基线无效'});
        assert.deepEqual(missing.metrics.slice(0,3).map(r=>r[1]), ['—','—','—']);
        assert.ok(missing.text.includes('睁眼基线无效'));
        const monitor = show(null);
        assert.equal(monitor.metrics[3][0], '启动质检');
        assert.ok(monitor.text.includes('本次没有可用的监测阶段指数'));
        '''
        script = script.replace("UTIL_URI", json.dumps((ROOT / "web/js/util.js").as_uri()))
        script = script.replace("FUNCTIONS", functions).replace("OVERVIEW", json.dumps(overview))
        result = subprocess.run(["node", "--input-type=module", "-e", script],
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
