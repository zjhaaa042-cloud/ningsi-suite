"""Previously saved operational sessions use the current presentation alongside their original evidence."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from ningsi_studio.api.routes import build_router
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.http.router import Request
from ningsi_studio.settings import Settings


class PresentationExportTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        settings = Settings(data_dir=Path(folder.name))
        store.initialize(settings.db_path)
        self.router = build_router(settings)
        self.addCleanup(self.router.human_models.close)
        self.original = b"<!doctype html><h1>Original saved layout</h1>"
        original_file = settings.data_dir / "old.html"
        self.json_file = settings.data_dir / "session.json"
        original_file.write_bytes(self.original)
        self.json_file.write_text(json.dumps({"extras": {"source_kind": "sim"},
                                             "quality": {"windows_usable": 1, "windows_total": 2}}),
                                  encoding="utf-8-sig")
        with store.connect(settings.db_path) as conn:
            participant = repo.create_subject(conn, "p01")
            row = repo.create_session(conn, participant, label="旧会话测试")
            repo.update_session(conn, row["uuid"], phase="done", status="done")
            repo.add_artifact(conn, row["id"], "report_html", str(original_file))
            repo.add_artifact(conn, row["id"], "report_json", str(self.json_file))
        self.base = "/api/sessions/" + row["uuid"]

    def get(self, suffix, **query):
        return self.router.dispatch(Request("GET", self.base + suffix, query, {}))

    def test_download_and_zip_match_presentation_and_preserve_original_artifact(self):
        view = self.get("/presentation")
        download = self.get("/presentation", download="1")
        self.assertEqual(view.body, download.body)
        self.assertIn("旧会话测试", view.body.decode("utf-8"))
        self.assertIn(b"<details", view.body)
        self.assertIn("attachment", download.headers["Content-Disposition"])
        self.assertNotIn("Content-Disposition", self.get("/presentation").headers)
        self.assertEqual(self.get("/artifacts/report_html").body, self.original)
        with zipfile.ZipFile(io.BytesIO(self.get("/export.zip").body)) as bundle:
            self.assertEqual(bundle.read("report_html/old.html"), self.original)
            self.assertEqual(bundle.read("report_json/session.json"), self.json_file.read_bytes())
            html = bundle.read("展示报告.html")
            self.assertEqual(html, view.body)
            files = json.loads(bundle.read("manifest.json"))["files"]
            self.assertEqual(files["presentation"]["sha256"], hashlib.sha256(html).hexdigest())
            self.assertEqual(files["presentation"]["bytes"], len(html))

    def test_unreadable_or_missing_structured_report_still_exports_original_html(self):
        self.json_file.write_bytes(b"{invalid}")
        for missing in (False, True):
            if missing:
                self.json_file.unlink()
            with zipfile.ZipFile(io.BytesIO(self.get("/export.zip").body)) as bundle:
                self.assertEqual(bundle.read("report_html/old.html"), self.original)
                self.assertNotIn("展示报告.html", bundle.namelist())
                self.assertNotIn("presentation", json.loads(bundle.read("manifest.json"))["files"])
