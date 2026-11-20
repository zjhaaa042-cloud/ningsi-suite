"""Archived tests remain browsable without changing live sessions or their aggregates."""
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ningsi_studio.api.routes import build_router
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.http.router import NotFound, Request
from ningsi_studio.settings import Settings


class ArchivedHistoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.settings = Settings(data_dir=Path(temp.name))
        store.initialize(self.settings.db_path)
        self.router = build_router(self.settings)
        self.addCleanup(self.router.human_models.close)
        self.archive = self.settings.data_dir / "history_archive"
        (self.archive / "reports").mkdir(parents=True)
        self.uuid = "a" * 32
        self.record = {"uuid": self.uuid, "label": "<session-label>", "status": "done", "participant": "cap01",
                       "origin_label": "生成输入", "summary": "100 次更新，0 次评分",
                       "phase_label": "完成", "sample_count": 0, "duration_seconds": 0,
                       "source_ref": "a/<b>", "created_at": "2026-01-02T08:00:00+08:00",
                       "report_file": f"reports/{self.uuid}.html"}
        (self.archive / self.record["report_file"]).write_text("<h1>report</h1>", encoding="utf-8")
        self.save([self.record])

    def save(self, items):
        (self.archive / "index.json").write_text(json.dumps({"version": 1, "items": items}), encoding="utf-8")

    def get(self, suffix="", **query):
        return self.router.dispatch(Request("GET", "/api/test-history" + suffix, query, {}))

    def test_live_priority_applies_before_filters_and_pagination(self):
        with store.connect(self.settings.db_path) as conn:
            subject = repo.create_subject(conn, "p01")
            live = repo.create_session(conn, subject, label="live")
            conn.execute("UPDATE sessions SET created_at='2026-01-02T01:00:00Z' WHERE id=?", (live["id"],))
        self.save([self.record, dict(self.record, uuid=live["uuid"], participant="obsolete")])
        first = json.loads(self.get(limit="1").body)
        self.assertEqual(first["total"], 2)
        self.assertEqual(first["participants"], ["cap01", "p01"])
        self.assertEqual(first["live_participants"], ["p01"])
        self.assertEqual(first["items"][0]["uuid"], live["uuid"])
        self.assertFalse(first["items"][0]["archived"])
        self.assertFalse(first["items"][0]["has_report"])
        second = json.loads(self.get(limit="1", page="2").body)["items"][0]
        self.assertTrue(second["archived"])
        self.assertEqual(second["uuid"], self.uuid)
        self.assertEqual(json.loads(self.get(participant="obsolete").body)["items"], [])
        self.assertEqual(json.loads(self.get(status="done", participant="sub-CAP01").body)["total"], 1)
        with self.assertRaises(NotFound):
            self.get(f"/{live['uuid']}/evidence")
        sessions = self.router.dispatch(Request("GET", "/api/sessions", {}, {}))
        self.assertEqual(json.loads(sessions.body)["total"], 1)

    def test_same_letters_different_participants_remain_separately_selectable(self):
        with store.connect(self.settings.db_path) as conn:
            subject = repo.create_subject(conn, "qa12")
            live = repo.create_session(conn, subject, label="live")
        self.save([dict(self.record, participant="QA12")])
        listing = json.loads(self.get().body)
        self.assertEqual(listing["participants"], ["QA12", "qa12"])
        self.assertEqual(listing["live_participants"], ["qa12"])
        for code, expected in [("qa12", live["uuid"]), ("sub-QA12", self.uuid), ("QA12", self.uuid)]:
            self.assertEqual([r["uuid"] for r in json.loads(self.get(participant=code).body)["items"]], [expected])
        self.assertEqual(json.loads(self.get(participant="Qa12").body)["items"], [])

    def test_evidence_escapes_metadata_and_only_serves_named_report(self):
        evidence = self.get(f"/{self.uuid}/evidence").body.decode()
        self.assertIn("&lt;session-label&gt;", evidence)
        self.assertIn("a/&lt;b&gt;", evidence)
        self.assertIn("100 次更新，0 次评分", evidence)
        self.assertIn("采样点数</dt><dd>0</dd>", evidence)
        self.assertIn('<details class="appendix"><summary>查看记录明细</summary>', evidence)
        self.assertNotIn('<details open', evidence)
        self.assertIn("/report", evidence)
        self.settings.api_token = "configured"
        self.assertNotIn("/report", self.get(f"/{self.uuid}/evidence").body.decode())
        report = self.get(f"/{self.uuid}/report")
        self.assertEqual(report.body, b"<h1>report</h1>")
        self.assertIn("sandbox", report.headers["Content-Security-Policy"])
        for name in ("../studio.sqlite3", "index.json", "reports/" + "b" * 32 + ".html", ""):
            with self.subTest(name=name):
                self.save([dict(self.record, report_file=name)])
                self.assertIsNone(json.loads(self.get().body)["items"][0]["report_url"])
                with self.assertRaises(NotFound):
                    self.get(f"/{self.uuid}/report")

    def test_reparse_point_in_report_directory_is_rejected(self):
        original = Path.lstat
        def reparse(path):
            info = original(path)
            if path.name == "reports":
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return info
        with patch.object(Path, "lstat", reparse):
            self.assertIsNone(json.loads(self.get().body)["items"][0]["report_url"])
            with self.assertRaises(NotFound):
                self.get(f"/{self.uuid}/report")

    def test_legacy_fast_device_demo_recovers_automatic_response_context(self):
        from ningsi_studio.api.history_archive import contextual_report
        report = {"extras": {"device": "sim-bsense", "source_kind": "sim", "time_scale": .05}}
        context = contextual_report(report, {})["extras"]["report_context"]
        self.assertEqual(context["response_mode"], "automated")
        self.assertNotIn("report_context", report["extras"])
        self.assertEqual(contextual_report(report, {"response_mode": "manual"})["extras"]["report_context"]["response_mode"], "manual")
        for changed in ({"time_scale": 1}, {"source_kind": "lsl"}):
            other = {"extras": dict(report["extras"], **changed)}
            self.assertNotIn("response_mode", contextual_report(other, {})["extras"]["report_context"])

    def test_presentation_uses_this_sessions_saved_json_and_keeps_original(self):
        folder = self.archive / "report_data"
        folder.mkdir()
        report = folder / (self.uuid + ".json")
        report.write_text(json.dumps({"extras": {"source_kind": "lsl"}}), encoding="utf-8")
        self.save([dict(self.record, report_data_file="report_data/" + report.name,
                        response_mode="automated", signal_origin="generated")])
        item = json.loads(self.get().body)["items"][0]
        self.assertTrue(item["presentation_url"].endswith("/presentation"))
        with patch("ningsi.assessment.diagnostics.render_html", return_value="展示") as render:
            self.assertEqual(self.get(f"/{self.uuid}/presentation").body.decode(), "展示")
            context = render.call_args.args[0]["extras"]["report_context"]
            self.assertEqual((context["title"], context["signal_origin"], context["response_mode"]),
                             ("<session-label>", "generated", "automated"))
        self.assertEqual(self.get(f"/{self.uuid}/report").body, b"<h1>report</h1>")
        self.save([dict(self.record, report_data_file="report_data/" + "b"*32 + ".json")])
        with self.assertRaises(NotFound):
            self.get(f"/{self.uuid}/presentation")
