"""Availability replay must recover real windows without inventing scores."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi_studio.core.runtime import SessionRuntime
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.domain import heatmap
from ningsi_studio.domain.indicators import WindowEngine
from .helpers import StudioTestCase


def payload(**patch):
    return {"index": 1, "phase": "baseline_open", "t_end": 2., "usable": True,
            "scores": {}, "reasons": [], "indicator_reason": "baseline_invalid", **patch}


class AvailabilityCellsTests(unittest.TestCase):
    def test_good_signal_without_baseline_is_visible_but_never_scored(self):
        source = payload()
        before = copy.deepcopy(source)
        cell = heatmap.cell_from_payload(source)
        self.assertEqual(cell["status"], "baseline_pending")
        self.assertTrue(cell["signal_usable"])
        self.assertIsNone(cell["score"])
        self.assertEqual(cell["index"], -1)
        self.assertNotIn("低质量", cell["label"])
        self.assertEqual(cell["window_id"], "baseline_open:1")
        self.assertEqual(source, before)

    def test_absent_short_timing_and_rejected_signal_have_distinct_states(self):
        for reason, status in (("no_data", "no_data"), ("signal_too_short", "no_data"),
                               ("clock_unsynchronized", "timing_issue"), ("eog", "signal_limited")):
            with self.subTest(reason=reason):
                cell = heatmap.cell_from_payload(payload(usable=False, reasons=[reason],
                    scores={"focus": .9}, indicator_reason="window_unusable"))
                self.assertEqual(cell["status"], status)
                self.assertFalse(cell["signal_usable"])
                self.assertIsNone(cell["score"], "stray score cannot promote a rejected signal")
                self.assertIn(reason, cell["reason_codes"])

    def test_zero_score_is_preserved_and_invalid_scores_do_not_fill_cells(self):
        cell = heatmap.cell_from_payload(payload(scores={"focus": 0.}, indicator_reason=""))
        self.assertEqual(cell["status"], "scored")
        self.assertEqual(cell["score"], 0.)
        for value in (True, float("nan"), float("inf"), 1.1, -.1):
            self.assertIsNone(heatmap.cell_from_payload(payload(scores={"focus": value}))["score"])

    def test_legacy_metadata_does_not_invent_a_baseline_failure_or_quality(self):
        rows = [{"id": index+1, "t_sec": index*2, "value": None, "quality": "{}", "valid_ratio": ratio}
                for index, ratio in enumerate((1., 0., None))]
        cells = heatmap.cells_from_metrics(rows)
        self.assertEqual([cell["signal_usable"] for cell in cells], [True, False, None])
        self.assertEqual([cell["status"] for cell in cells], ["unscored", "signal_limited", "unscored"])
        self.assertTrue(all(not cell["indicator_reason"] for cell in cells))
        self.assertEqual(cells[1]["reason_codes"], ["window_unusable"])
        self.assertEqual(cells[2]["reason_codes"], ["unknown_quality"])

    def test_summary_denominator_covers_all_windows_and_keeps_reasons_overlapping(self):
        cells = [heatmap.cell_from_payload(payload()),
                 heatmap.cell_from_payload(payload(index=2, phase="training", scores={"focus": .4})),
                 heatmap.cell_from_payload(payload(index=3, usable=False, reasons=["eog", "eog", "amplitude"])),
                 heatmap.cell_from_payload(payload(index=4, usable=None))]
        result = heatmap.response("test", cells)
        summary = result["summary"]
        self.assertEqual((summary["total"], summary["signal_usable"], summary["scored"], summary["missing"]), (4, 2, 1, 3))
        self.assertEqual((summary["signal_coverage"], summary["score_coverage"]), (.5, .25))
        self.assertEqual(summary["unknown_quality"], 1)
        self.assertEqual(next(item["count"] for item in summary["reason_counts"] if item["code"] == "eog"), 1)
        self.assertEqual(summary["phase_counts"][1]["scored"], 1)
        self.assertEqual(summary["coverage_unit"], "windows")
        json.dumps(result, allow_nan=False)

    def test_svg_uses_separate_availability_track_and_escapes_labels(self):
        cells = [heatmap.cell_from_payload(payload(phase='<script>bad()</script>')),
                 heatmap.cell_from_payload(payload(index=2, scores={"focus": .8}, indicator_reason=""))]
        before = copy.deepcopy(cells)
        svg = heatmap.render_svg(cells, title='<script>alert("x")</script>')
        ET.fromstring(svg)
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("基线准备窗不等于信号不合格", svg)
        self.assertIn("指数=未评分", svg)
        self.assertIn("指数=0.8", svg)
        self.assertIn(heatmap.STATUS_META["baseline_pending"][1], svg)
        self.assertNotIn("低质量缺失", svg)
        self.assertEqual(cells, before)


class AvailabilityReplayTests(StudioTestCase):
    port_base = 0

    def seed(self):
        with store.connect(self.db_path) as conn:
            subject = repo.create_subject(conn, repo.next_public_id(conn))
            row = dict(repo.create_session(conn, subject, protocol="ten_min"))
            repo.update_session(conn, row["uuid"], status="done", phase="done")
        return row

    def get_heatmap(self, uuid):
        return self.json_body(self.get(f"/api/sessions/{uuid}/heatmap"), 200)

    def test_training_cells_persist_and_replay_without_entering_indicator_trends(self):
        row = self.seed()
        runtime = SessionRuntime(row["uuid"], settings=self.app.settings)
        runtime.session = row
        engine = WindowEngine(SyntheticEEG(seed=42), 250, device="sim-bsense")
        pending = runtime._record_heatmap(engine.step("rest"), "baseline_open")
        engine.build_baseline("rest", 30)
        scored = runtime._record_heatmap(engine.step("focused"), "training")
        before = self.get_heatmap(row["uuid"])
        self.assertEqual(before["cells"], [pending, scored])
        self.assertEqual(before["summary"]["scored"], 1)
        with store.read_only(self.db_path) as conn:
            self.assertEqual(repo.list_metrics(conn, row["id"], kind="indicator"), [])
            self.assertEqual(repo.latest_metric_summary(conn, row["id"]), {})
        self.restart_service()
        self.assertEqual(self.get_heatmap(row["uuid"]), before)

    def test_window_sse_and_rest_share_the_same_cell_and_quality_provenance(self):
        row = self.seed()
        runtime = SessionRuntime(row["uuid"], settings=self.app.settings)
        runtime.session, runtime.source, runtime.bus = row, SimpleNamespace(kind="sim"), Mock()
        runtime._record_diagnostic = Mock()
        step = WindowEngine(SyntheticEEG(seed=42), 250, device="sim-bsense").step("rest")
        runtime._publish_window(step, phase="qc", include_signal=False)
        event = next(call.args[1] for call in runtime.bus.publish.call_args_list if call.args[0] == "window")
        self.assertEqual(self.get_heatmap(row["uuid"])["cells"], [event["heatmap_cell"]])
        with store.read_only(self.db_path) as conn:
            quality = json.loads(repo.list_metrics(conn, row["id"], kind="indicator", name="focus")[0]["quality"])
        self.assertEqual(quality["phase"], "qc")
        self.assertEqual(quality["indicator_reason"], "baseline_invalid")

    def test_old_report_recovers_baseline_and_omitted_training_without_modification(self):
        row = self.seed()
        path = self.data_dir / f'{row["uuid"]}_report.json'
        windows = [payload(), payload(index=2, phase="training", t_end=90., scores={"focus": .6}, indicator_reason="")]
        path.write_text(json.dumps({"extras": {"signal_diagnostics": {"windows": windows}}}), encoding="utf-8")
        before = path.read_bytes()
        with store.connect(self.db_path) as conn:
            repo.add_artifact(conn, row["id"], "report_json", str(path))
            repo.add_metric(conn, row["id"], "indicator", "focus", t_sec=2., value=None, valid_ratio=1.)
        result = self.get_heatmap(row["uuid"])
        self.assertEqual(result["evidence_source"], "report_diagnostics")
        self.assertEqual([cell["status"] for cell in result["cells"]], ["baseline_pending", "scored"])
        self.assertEqual(result["summary"]["signal_usable"], 2)
        self.assertEqual(path.read_bytes(), before)
        with store.read_only(self.db_path) as conn:
            self.assertEqual(repo.list_metrics(conn, row["id"], kind="heatmap"), [])

    def test_unrecoverable_legacy_null_is_unscored_and_new_durable_cells_take_precedence(self):
        row = self.seed()
        with store.connect(self.db_path) as conn:
            repo.add_metric(conn, row["id"], "indicator", "focus", t_sec=2., value=None, valid_ratio=1.)
        before = self.get_heatmap(row["uuid"])
        self.assertEqual(before["cells"][0]["status"], "unscored")
        self.assertTrue(before["cells"][0]["signal_usable"])
        evidence = payload(window_id="qc:1", usable=False, reasons=["signal_too_short"], phase="qc")
        with store.connect(self.db_path) as conn:
            repo.add_metric(conn, row["id"], "heatmap", "focus", t_sec=2., value=None, valid_ratio=0., quality=heatmap.provenance(evidence))
        after = self.get_heatmap(row["uuid"])
        self.assertEqual(len(after["cells"]), 1, "new and legacy paths must not duplicate windows")
        self.assertEqual(after["cells"][0]["status"], "no_data")
        self.assertEqual(after["evidence_source"], "heatmap_metrics")
