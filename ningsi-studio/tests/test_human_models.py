"""Meaningful leakage, label, import, inference and actual HTTP regression tests."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import zipfile

import numpy as np

from ningsi.models import human, human_data
from ningsi.models.capture_protocols import FULL_PROTOCOL, SHORT_PROTOCOL
from ningsi.models.logistic import FEATURE_NAMES
from ningsi.models.research_features import aggregate_probe, condition_window
from ningsi.acquisition.timing import analysis_timestamps, PACKET_TIMING
from ningsi_studio.core.human_inference import HumanAttentionPredictor
from .helpers import StudioTestCase
from .human_fixtures import fixture_files


class HumanImportTests(unittest.TestCase):
    def test_protocol_identity_survives_import_and_only_development_protocols_enter_model(self):
        files = fixture_files(development=5, test=1)
        prefix = "QA005/"
        def set_protocol(manifest_name, exported_name):
            for name, field in (("manifest.json", {"protocol": {"name": manifest_name}}),
                                ("export.json", {"protocol_id": exported_name, "checksum_required": True})):
                payload = json.loads(files[prefix + name])
                files[prefix + name] = json.dumps({**payload, **field})
            hashes = {name: hashlib.sha256(files[prefix + name].encode()).hexdigest()
                      for name in ("manifest.json", "export.json", "features.csv")}
            files[prefix + "SHA256.json"] = json.dumps(hashes)
        set_protocol(SHORT_PROTOCOL, SHORT_PROTOCOL)
        data = human_data.import_files(files)
        summary = human_data.summarize(data)
        self.assertEqual({p["id"]: p["sessions"] for p in summary["research_protocols"]},
                         {FULL_PROTOCOL: 5, SHORT_PROTOCOL: 1})
        self.assertEqual(len(data["test"]), 12)
        self.assertEqual(human.train_development(data)[0]["training_protocols"], [FULL_PROTOCOL])
        set_protocol(SHORT_PROTOCOL, FULL_PROTOCOL)
        with self.assertRaisesRegex(ValueError, "协议编号不一致"):
            human_data.import_files(files)
        set_protocol("unknown-protocol", "unknown-protocol")
        with self.assertRaisesRegex(ValueError, "协议"):
            human_data.import_files(files)

    def test_legacy_conflicting_unit_and_mislabeled_marker_cannot_supply_features(self):
        for fields, message in (({"declared_units": ["mV"]}, "单位"), ({"type": "Markers"}, "EEG")):
            with self.subTest(fields=fields):
                files = fixture_files(development=1, test=0)
                manifest = json.loads(files["QA000/manifest.json"])
                manifest["source"].update(fields)
                files["QA000/manifest.json"] = json.dumps(manifest)
                del files["QA000/SHA256.json"]
                with self.assertRaisesRegex(ValueError, message):
                    human_data.import_files(files)

    def test_new_exports_require_their_checksum_manifest(self):
        files = fixture_files(development=1, test=0)
        exported = json.loads(files["QA000/export.json"])
        exported["checksum_required"] = True
        files["QA000/export.json"] = json.dumps(exported)
        del files["QA000/SHA256.json"]
        with self.assertRaisesRegex(ValueError, "SHA256.json"):
            human_data.import_files(files)

    def test_merged_dataset_provenance_and_legacy_limit(self):
        data = human_data.import_files(fixture_files())
        fields = list(data["development"][0])
        files = {}
        for role in ("development", "test", "excluded"):
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader(); writer.writerows(data[role])
            files[role + ".csv"] = "\ufeff" + output.getvalue()
        meta = {"schema": "ningsi-capture-dataset-v1", "feature_names": list(FEATURE_NAMES),
                "label_column": "label", "group_column": "participant", "participants": data["participants"],
                "development_rows": len(data["development"]), "test_rows": len(data["test"]), "excluded_rows": 0,
                "engine_fingerprint": data["engine_fingerprint"], "source_profiles": data["source_profiles"],
                "implementation_fingerprint": data["implementation_fingerprint"],
                "session_metadata": data["session_metadata"],
                "file_sha256": {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in files.items()}}
        files["dataset.json"] = json.dumps(meta)
        imported = human_data.import_files(files)
        self.assertTrue(human_data.summarize(imported)["live_compatible_metadata"])
        meta.pop("implementation_fingerprint")
        files["dataset.json"] = json.dumps(meta)
        unknown = human_data.import_files(files)
        self.assertFalse(human_data.summarize(unknown)["live_compatible_metadata"])
        self.assertIsNone(human.train_development(unknown)[0]["implementation_fingerprint"])
        meta.pop("engine_fingerprint"); meta.pop("source_profiles"); meta.pop("session_metadata")
        files["dataset.json"] = json.dumps(meta)
        legacy = human_data.import_files(files)
        self.assertFalse(human_data.summarize(legacy)["live_compatible_metadata"])
        self.assertEqual(len(legacy["development"]), 72)

    def test_csv_bom_and_crlf_roundtrip_and_simulation_exclusion(self):
        files = fixture_files()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode("utf-8"))
            data = human_data.import_files(human_data.files_from_paths([root]))
        self.assertEqual(len(data["development"]), 72)
        self.assertEqual(len(data["test"]), 36)
        reverse = human_data.import_files(dict(reversed(list(files.items()))))
        self.assertEqual(data["dataset_id"], reverse["dataset_id"])
        simulation = fixture_files()
        for name in list(simulation):
            if name.endswith("SHA256.json"):
                del simulation[name]
            elif name.endswith("manifest.json"):
                meta = json.loads(simulation[name]); meta["source"]["data_origin"] = "simulation"
                simulation[name] = json.dumps(meta)
            elif name.endswith("features.csv"):
                simulation[name] = simulation[name].replace(",real,", ",simulation,")
        excluded = human_data.import_files(simulation)
        self.assertEqual(len(excluded["development"]), 0)
        self.assertEqual(len(excluded["test"]), 0)
        self.assertEqual(len(excluded["excluded"]), 108)

    def test_modified_files_and_duplicate_probe_rejected(self):
        files = fixture_files()
        modified = dict(files)
        modified["QA000/features.csv"] += "\n"
        with self.assertRaisesRegex(ValueError, "校验失败"):
            human_data.import_files(modified)
        duplicate = {**files, **{"duplicate/" + k.split("/", 1)[1]: v for k, v in files.items() if k.startswith("QA000/")}}
        with self.assertRaisesRegex(ValueError, "重复"):
            human_data.import_files(duplicate)

    def test_cross_cohort_and_reversed_labels_rejected(self):
        files = fixture_files()
        for name in list(files):
            if name.endswith("SHA256.json"):
                del files[name]
            elif name.startswith("QA006/"):
                if name.endswith("manifest.json"):
                    meta = json.loads(files[name]); meta["participant"] = "QA000"
                    files[name] = json.dumps(meta)
                elif name.endswith("features.csv"):
                    files[name] = files[name].replace("\r\nQA006,", "\r\nQA000,")
        with self.assertRaisesRegex(ValueError, "跨分组"):
            human_data.import_files(files)
        files = fixture_files()
        del files["QA000/SHA256.json"]
        files["QA000/features.csv"] = files["QA000/features.csv"].replace(",1,0,on_task,", ",1,1,internal_mind_wandering,")
        with self.assertRaisesRegex(ValueError, "标签方向"):
            human_data.import_files(files)

    def test_path_traversal_and_incomplete_import_rejected(self):
        with self.assertRaises(ValueError):
            human_data.import_files({"../features.csv": "x"})
        with self.assertRaisesRegex(ValueError, "需要"):
            human_data.import_files({"export.json": json.dumps({"schema": "ningsi-capture-export-v1"})})


class HumanTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = human_data.import_files(fixture_files())
        cls.model, cls.cv = human.train_development(cls.data)

    def test_group_isolation_and_fold_scaling(self):
        people = {r["participant"] for r in self.data["development"]}
        seen = []
        for split in self.cv["folds"]:
            self.assertFalse(set(split["train_participants"]) & set(split["validation_participants"]))
            seen.extend(split["validation_participants"])
        self.assertEqual(set(seen), people)
        self.assertEqual(len(seen), len(people))
        x = np.asarray([[1.], [3.], [10000.]])
        parameters = human.fit(x[:2], np.asarray([0, 1]), np.asarray([.5, .5]), .01)
        self.assertEqual(parameters["mean"], [2.])
        self.assertEqual(parameters["scale"], [1.])

    def test_test_data_cannot_change_model_parameters_or_threshold(self):
        changed = copy.deepcopy(self.data)
        for row in changed["test"]:
            for name in FEATURE_NAMES:
                row[name] = str(float(row[name]) * 100 + 500)
            row["label"] = str(1-int(row["label"]))
        model, cv = human.train_development(changed)
        self.assertEqual(model["parameters"], self.model["parameters"])
        self.assertEqual(model["calibration"], self.model["calibration"])
        self.assertEqual(cv["threshold"], self.cv["threshold"])

    def test_model_polarity_integrity_and_explanation_identity(self):
        p = human.predict(self.model, human.matrix(self.data["test"]))
        truth = [int(r["label"]) for r in self.data["test"]]
        self.assertGreater(np.mean(p[np.asarray(truth) == 1]), np.mean(p[np.asarray(truth) == 0]))
        explanation = human.explain(self.model, human.matrix(self.data["test"])[0])
        summed = explanation["intercept"] + sum(r["log_odds_contribution"] for r in explanation["contributions"])
        self.assertAlmostEqual(float(human.sigmoid(summed)), p[0], places=12)
        changed = copy.deepcopy(self.model); changed["positive_label"] = 0
        with self.assertRaises(ValueError):
            human.validate_model(changed)

    def test_independent_report_coverage_and_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            result = human.run_training(self.data, output)
            report = result["evaluation"]
            self.assertEqual(report["evidence_status"], "exploratory_insufficient_participants")
            self.assertEqual(report["coverage"]["eeg_coverage_given_binary_label"], 1.)
            self.assertEqual(report["paired_auc_difference"]["participants"], 3)
            self.assertEqual(len(report["test"]["primary_auc_ci95"]), 2)
            freeze = json.loads((output / "freeze.json").read_text())
            self.assertTrue(freeze["test_opened"])
            with zipfile.ZipFile(output / "model_bundle.zip") as archive:
                self.assertIn("report.html", archive.namelist())
                loaded = json.loads(archive.read("attention_model.json"))
            expected = human.predict(loaded, human.matrix(self.data["test"]))
            np.testing.assert_allclose(expected, human.predict(json.loads((output / "attention_model.json").read_text()), human.matrix(self.data["test"])))
            with self.assertRaises(FileExistsError):
                human.run_training(self.data, output)

    def test_missing_class_metrics_are_unavailable(self):
        metrics = human.scalar_metrics([0, 0], [.1, .2])
        self.assertIsNone(metrics["auc"])
        self.assertIsNone(metrics["sensitivity"])
        self.assertIsNone(metrics["balanced_accuracy"])

    def test_shared_offline_live_features_match_and_bad_units_abstain(self):
        rate = 100
        stamps = np.arange(2400) / rate
        raw = (8*np.sin(2*np.pi*6*stamps) + 10*np.sin(2*np.pi*10*stamps) + 12*np.sin(2*np.pi*20*stamps))[None, :]
        predictor = HumanAttentionPredictor(self.model)
        class Source:
            count = 400
            age = .01
            def model_snapshot(self):
                return {"samples": raw[:, :self.count], "timestamps": stamps[:self.count],
                        "total_samples": self.count, "buffer_id": 1, "age": self.age, "research_timing": True,
                        "descriptor": {"name": "FixtureEEG", "source_id": "qa-software-fixture", "channel_count": 1,
                                       "nominal_srate": 100., "channel_labels": ["Fp1"]}}
        source = Source()
        for count in range(400, 1801, 200):
            source.count = count
            prediction = predictor.update(source, "sart", baseline_valid=True)
        self.assertTrue(prediction["available"], prediction)
        offline = [condition_window(raw[:, max(0, end-600):end], stamps[max(0, end-600):end], rate, 1.)
                   for end in range(400, 1801, 200)]
        features = aggregate_probe(offline, 18., baseline_valid=True, phase_start=4.)["features"]
        for row in prediction["contributions"]:
            self.assertAlmostEqual(row["value"], features[row["feature"]], places=12)
        self.assertAlmostEqual(prediction["mind_wandering_probability"], human.explain(self.model, [features[name] for name in FEATURE_NAMES])["mind_wandering_probability"], places=12)
        source.age = 4
        self.assertFalse(predictor.update(source, "sart", baseline_valid=True)["available"])
        source.age = .01
        self.assertFalse(predictor.update(source, "sart", baseline_valid=False)["available"])
        with self.assertRaisesRegex(ValueError, "units_mismatch"):
            predictor.match_profile({"name": "FixtureEEG", "source_id": "qa-software-fixture", "channel_count": 1,
                                     "nominal_srate": 100., "channel_labels": ["Fp1"], "declared_units": ["V"]})
        self.assertFalse(predictor.update(source, "training", baseline_valid=True)["available"])

    def test_same_buffer_reconnect_discards_pre_connection_model_history(self):
        rate = 100
        stamps = np.arange(3600) / rate
        raw = sum(amplitude * np.sin(2*np.pi*frequency*stamps)
                  for frequency, amplitude in ((6, 8), (10, 10), (20, 12)))[None, :]
        predictor = HumanAttentionPredictor(self.model)

        class Source:
            count = 400
            generation = 1

            def model_snapshot(self):
                return {"samples": raw[:, :self.count], "timestamps": stamps[:self.count],
                        "total_samples": self.count, "buffer_id": 1,
                        "connection_generation": self.generation, "age": .01, "research_timing": True,
                        "descriptor": {"name": "FixtureEEG", "source_id": "qa-software-fixture",
                                       "channel_count": 1, "nominal_srate": rate, "channel_labels": ["Fp1"]}}

        source = Source()
        for source.count in range(400, 1801, 200):
            result = predictor.update(source, "sart", baseline_valid=True)
        self.assertTrue(result["available"], result)
        source.generation += 1
        result = predictor.update(source, "sart", baseline_valid=True)
        self.assertFalse(result["available"], "重连后不能复用旧连接的模型输出")
        self.assertIn("lookback_crosses_phase", result["reason"])
        for source.count in range(2000, 3001, 200):
            result = predictor.update(source, "sart", baseline_valid=True)
        self.assertTrue(result["available"], result)
        self.assertGreaterEqual(result["window_start_lsl"], 18.)

    def test_causal_filter_import_training_and_live_features(self):
        from ningsi.signal import conditioning
        files = fixture_files(preprocessing_policy=conditioning.POLICY)
        legacy = fixture_files()
        mixed = {**files, **{k: v for k, v in legacy.items() if k.startswith("QA001/")}}
        with self.assertRaisesRegex(ValueError, "不能混用"):
            human_data.import_files(mixed)
        data = human_data.import_files(files)
        model, _ = human.train_development(data)
        self.assertEqual(model["source_profiles"][0]["preprocessing_policy"], conditioning.POLICY)
        with self.assertRaisesRegex(ValueError, "不同的脑电预处理"):
            human.independent_test(data, self.model)
        predictor = HumanAttentionPredictor(model)
        rate = 250
        stamps = np.arange(4500) / rate
        raw = (11*np.sin(2*np.pi*10*stamps) + 7*np.sin(2*np.pi*6*stamps)
               + 5*np.sin(2*np.pi*20*stamps) + 800*np.sin(2*np.pi*50*stamps))[None, :]
        class Source:
            def model_snapshot(self):
                return {"samples": raw[:, :self.count], "timestamps": stamps[:self.count],
                        "total_samples": self.count, "buffer_id": 1, "age": .01, "research_timing": True,
                        "descriptor": {"name": "FixtureEEG", "source_id": "qa-software-fixture", "channel_count": 1,
                                       "nominal_srate": rate, "channel_labels": ["Fp1"]}}
        source = Source()
        offline = []
        for end in range(1000, 4501, 500):
            source.count = end
            prediction = predictor.update(source, "sart", baseline_valid=True)
            region = slice(max(0, end-1500), end)
            offline.append(conditioning.condition(raw[:, region], stamps[region], rate, 1., conditioning.POLICY))
        features = aggregate_probe(offline, 18., baseline_valid=True, phase_start=4.)["features"]
        self.assertTrue(prediction["available"], prediction)
        for item in prediction["contributions"]:
            self.assertAlmostEqual(item["value"], features[item["feature"]], places=12)

    def test_packet_timing_policy_survives_import_and_matches_live_features(self):
        model = copy.deepcopy(self.model)
        model["source_profiles"][0]["timestamp_policy"] = PACKET_TIMING
        model["model_id"] = human_data.digest({k:v for k,v in model.items() if k != "model_id"})
        predictor = HumanAttentionPredictor(model)
        rate = 100
        clean = np.arange(2400) / rate
        times = clean + .015 * np.sin(np.floor(np.arange(2400)/10) * 1.7)
        raw = (8*np.sin(2*np.pi*6*clean) + 10*np.sin(2*np.pi*10*clean) + 12*np.sin(2*np.pi*20*clean))[None, :]
        class Source:
            count = 400
            def model_snapshot(self):
                return {"samples":raw[:,:self.count], "timestamps":times[:self.count], "total_samples":self.count,
                        "buffer_id":1, "age":.01, "research_timing":True,
                        "descriptor":{"name":"FixtureEEG", "source_id":"qa-software-fixture", "channel_count":1,
                                      "nominal_srate":100., "channel_labels":["Fp1"]}}
        source = Source()
        for count in range(400, 1801, 200):
            source.count = count
            result = predictor.update(source, "sart", baseline_valid=True)
        self.assertTrue(result["available"], result)
        offline = []
        for end in range(400, 1801, 200):
            start = max(0,end-600)
            adjusted,_ = analysis_timestamps(times[start:end],rate,PACKET_TIMING)
            offline.append(condition_window(raw[:,start:end],adjusted,rate,1.))
        features = aggregate_probe(offline,times[1799]+.01,baseline_valid=True,phase_start=times[399]+.01)["features"]
        for row in result["contributions"]:
            self.assertAlmostEqual(row["value"],features[row["feature"]],places=12)
        profile = human_data.source_profile({**model["source_profiles"][0]})
        self.assertEqual(profile["timestamp_policy"],PACKET_TIMING)

    def test_biomulti_model_matches_recording_aligned_windows_across_buffer_trims(self):
        from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, biomulti_packet_clock
        model = copy.deepcopy(self.model)
        profile = model["source_profiles"][0]
        profile.update(srate=250., channels=2, channel_labels=["Fp1", "Fp2"], declared_units=["", ""],
                       timestamp_policy=BIOMULTI_LSL_TIMING, unit_basis="biomulti_software_inference")
        model["model_id"] = human_data.digest({k: v for k, v in model.items() if k != "model_id"})
        predictor = HumanAttentionPredictor(model)
        grid = np.arange(8000) / 250.
        lag = np.resize([0., .002, .004, .085, .052, .02], 800)
        stamps = grid + np.repeat(lag, 10)
        signal = sum(8 * np.sin(2 * np.pi * f * grid) for f in (5, 10, 20))
        values = np.stack((signal, .9 * signal))
        class Source:
            count = 1000
            def model_snapshot(self):
                start = max(0, self.count - 3267)
                return {"samples": values[:, start:self.count], "timestamps": stamps[start:self.count],
                        "total_samples": self.count, "buffer_id": 1, "age": .01, "research_timing": True,
                        "descriptor": {"name": profile["name"], "source_id": profile["source_id"],
                                       "channel_count": 2, "nominal_srate": 250., "channel_labels": ["Fp1","Fp2"]}}
        source = Source()
        for count in range(1000, 8001, 250):
            source.count = count
            result = predictor.update(source, "sart", baseline_valid=True)
        self.assertTrue(result["available"], result)
        clock = biomulti_packet_clock(stamps)
        end = predictor.processed
        expected = condition_window(values[:, end-1500:end], clock[end-1500:end], 250., 1.)
        self.assertEqual(predictor.windows[-1]["end"], expected["end"])
        self.assertEqual(predictor.windows[-1]["features"], expected["features"])

    def test_native_rec_model_cannot_apply_offline_timestamps_to_live_lsl(self):
        from ningsi.acquisition.timing import BIOMULTI_REC_TIMING
        model = copy.deepcopy(self.model)
        model["source_profiles"][0]["timestamp_policy"] = BIOMULTI_REC_TIMING
        model["model_id"] = human_data.digest({key: value for key, value in model.items() if key != "model_id"})
        predictor = HumanAttentionPredictor(model)
        with self.assertRaisesRegex(ValueError, "native_rec_clock_required"):
            predictor.match_profile({"name":"FixtureEEG", "source_id":"qa-software-fixture", "channel_count":1,
                                     "nominal_srate":100., "channel_labels":["Fp1"]})

    def test_sart_model_ticks_preserve_phase_and_do_not_advance_window_engine(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from ningsi_studio.core.runtime import SessionRuntime
        runtime = SessionRuntime("software-test-session")
        calls = []
        runtime.source = SimpleNamespace(engine=object())
        runtime.engine = SimpleNamespace(baseline=SimpleNamespace(valid=True))
        runtime.human_predictor = SimpleNamespace(update=lambda source, phase, **kw: {"available": False, "reason": phase})
        runtime.bus = SimpleNamespace(publish=lambda name, payload: calls.append((name, payload)))
        with patch("ningsi_studio.core.runtime.time.monotonic", return_value=100.):
            runtime._tick_attention_model("sart_practice")
            runtime._tick_attention_model("sart_practice")
            runtime._tick_attention_model("sart")
        self.assertEqual([item[1]["reason"] for item in calls], ["sart_practice", "sart"])
        self.assertEqual(len(runtime.attention_predictions), 2)


class HumanModelApiTests(StudioTestCase):
    port_base = 19720

    def test_import_train_download_activate_and_persistent_results(self):
        files = fixture_files()
        response = self.post("/api/models/human/import", {"files": files})
        self.assertEqual(response.status, 201, response.text())
        key = response.json()["dataset_id"]
        response = self.post("/api/models/human/train", {"dataset_id": key})
        self.assertIn(response.status, (200, 202), response.text())
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = self.get(f"/api/models/human/jobs/{run_id}").json()
            if job["state"] != "running":
                break
            time.sleep(.05)
        self.assertEqual(job["state"], "completed", job)
        model = self.get(f"/api/models/human/jobs/{run_id}/artifacts/model").json()
        self.assertEqual(model["labels"], {"0": "on_task", "1": "internal_mind_wandering"})
        archive = self.get(f"/api/models/human/jobs/{run_id}/artifacts/bundle")
        self.assertEqual(archive.status, 200)
        with zipfile.ZipFile(io.BytesIO(archive.body)) as zipped:
            self.assertIn("evaluation.json", zipped.namelist())
        preview = self.get(f"/api/models/human/jobs/{run_id}/artifacts/report?preview=1")
        self.assertEqual(preview.status, 200)
        self.assertTrue(preview.content_disposition.startswith("inline;"))
        self.assertIn("探索性结果", preview.text())
        activated = self.post("/api/models/human/activate", {"run_id": run_id})
        self.assertEqual(activated.status, 200, activated.text())
        state = self.get("/api/models/human").json()
        self.assertEqual(state["active"]["model_id"], model["model_id"])
        repeated = self.post("/api/models/human/train", {"dataset_id": key})
        self.assertEqual(repeated.json()["run_id"], run_id)
        new_data = self.post("/api/models/human/import", {"files": fixture_files(per_class=7)})
        self.assertEqual(new_data.status, 201)
        reused = self.post("/api/models/human/train", {"dataset_id": new_data.json()["dataset_id"]})
        self.assertEqual(reused.status, 422)
        self.assertIn("测试参与者", reused.json()["error"]["message"])
        invalid = copy.deepcopy(model); invalid["labels"] = {"0": "mind_wandering", "1": "focused"}
        self.assertEqual(self.post("/api/models/human/activate", {"model": invalid}).status, 422)
        self.assertEqual(self.delete("/api/models/human/active").status, 200)

    def test_import_failures_and_unknown_jobs(self):
        response = self.post("/api/models/human/import", {"files": {"../manifest.json": "{}"}})
        self.assertEqual(response.status, 422)
        self.assertEqual(self.get("/api/models/human/jobs/" + "a"*32).status, 404)
        self.assertEqual(self.post("/api/models/human/activate", {}).status, 422)

    def test_local_directory_uses_the_same_import_and_cannot_be_called_remotely(self):
        from ningsi_studio.http.router import Forbidden, Request
        files = fixture_files()
        root = self.data_dir / "local-capture-fixtures"
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
        imported = self.post("/api/models/human/import-local", {"paths": [str(root)]})
        self.assertEqual(imported.status, 201, imported.text())
        self.assertEqual(imported.json()["dataset_id"], human_data.import_files(files)["dataset_id"])
        request = Request(method="POST", path="/api/models/human/import-local", query={}, headers={},
                          body=json.dumps({"paths": [str(root)]}).encode("utf-8"), remote="192.0.2.20")
        with self.assertRaises(Forbidden) as denied:
            self.app.server.router.dispatch(request)
        self.assertEqual(denied.exception.status, 403)

    def test_local_directory_rejects_relative_network_and_disk_root(self):
        for value in ("relative-directory", "\\\\server\\share", str(Path(self.data_dir.anchor))):
            response = self.post("/api/models/human/import-local", {"paths": [value]})
            self.assertEqual(response.status, 422, value)

    def test_feature_import_accepts_over_one_megabyte_without_expanding_other_endpoints(self):
        files = fixture_files()
        files["QA000/baseline.json"] = json.dumps({"valid": True, "note": "x" * 1100000})
        response = self.post("/api/models/human/import", {"files": files})
        self.assertEqual(response.status, 201, response.text())

    def test_external_entry_navigation_does_not_allow_cross_origin_api(self):
        headers = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}
        self.assertEqual(self.get("/", headers=headers).status, 200)
        self.assertEqual(self.get("/api/models/human", headers=headers).status, 403)
        self.assertEqual(self.post("/api/models/human/train", {"dataset_id": "a"*64}, headers=headers).status, 403)
