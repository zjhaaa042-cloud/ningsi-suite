"""端到端会话与上游数据导入的集成测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.helpers import temp_root
from ningsi.app.pipeline import SessionConfig, import_studio_history, run_session
from ningsi.adapters import studio
from ningsi.monitoring import history


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = temp_root("pipeline")
        cls.artifacts = run_session(SessionConfig(root=str(cls.root), participant="p99", seed=13))

    def test_all_steps_produce_results(self):
        data = self.artifacts.as_dict()
        self.assertTrue(self.artifacts.quality["passed"])
        self.assertEqual(self.artifacts.baseline["reference"], "eyes_open")
        self.assertTrue(self.artifacts.baseline["eyes_open"]["valid"])
        self.assertTrue(self.artifacts.baseline["eyes_closed"]["valid"])
        self.assertIn("SAS", self.artifacts.scales)
        self.assertIn("sds", {k.lower() for k in self.artifacts.scales})
        self.assertEqual(self.artifacts.behavior["sart"]["trials"], 180)
        self.assertTrue(self.artifacts.behavior["pvt"]["valid"])
        for key in ("focus", "relax", "load"):
            self.assertIn(key, data["indicators"]["summary"])
        self.assertIn("conclusion", self.artifacts.assessment)

    def test_artifacts_written_to_disk(self):
        for key in ("report_md", "report_json", "heatmap_svg", "trend_svg", "history"):
            path = Path(self.artifacts.paths[key])
            self.assertTrue(path.exists(), key)
        report = Path(self.artifacts.paths["report_md"]).read_text(encoding="utf-8")
        self.assertIn("联合评估结论", report)
        self.assertIn("证据回填", report)
        payload = json.loads(Path(self.artifacts.paths["report_json"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["participant"], "p99")
        self.assertEqual(payload["versions"]["spectrum"], "welch-v1")

    def test_history_record_appended(self):
        records = history.load_records(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["participant"], "p99")
        self.assertIn("focus", records[0]["indicators"])

    def test_studio_import(self):
        dataset = self.root / "upstream" / "derived" / "features"
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "records.csv").write_text(
            "participant,session,task,run,quality_status,eeg_valid_ratio,mean_reaction_time_s,xdf_path\n"
            "p77,01,m6_readiness_reference,001,pass,0.93,0.31,raw/sub-p77/x.xdf\n",
            encoding="utf-8",
        )
        count = import_studio_history(self.root / "upstream", self.root)
        self.assertEqual(count, 1)
        imported = [r for r in history.load_records(self.root) if r.get("source") == "bsense-dataset-studio"]
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["behavior"]["mean_reaction_time_s"], 0.31)
        self.assertIsNone(studio.find_records_csv(self.root / "missing"))


if __name__ == "__main__":
    unittest.main()
