"""Local human-model datasets, bounded training jobs, frozen results and activation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
import threading
import uuid

from ningsi.models import human, human_data
from ningsi.models.research_features import engine_fingerprint
from ningsi.signal import conditioning


def _check_training_input(data):
    human.group_folds(data["development"])
    if len({r["label"] for r in data["test"]}) != 2:
        raise ValueError("独立测试组需要两类可用标签，一键评估不会使用开发数据代替。")


def _import_summary(data):
    result = human_data.summarize(data)
    try:
        _check_training_input(data)
    except ValueError as exc:
        result["training_blocker"] = str(exc)
    else:
        result["training_blocker"] = None
    return result


class HumanModelService:
    def __init__(self, data_dir):
        self.root = Path(data_dir) / "human_models"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        # A crashed job is visible and cannot masquerade as a completed test.
        for path in self.root.glob("runs/*/job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["state"] == "running":
                job.update(state="failed", error="上次服务退出时任务中断；已保存的冻结模型和记录保留。")
                human.write_json(path, job)

    def _dataset_path(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", str(key)):
            raise ValueError("数据集编号无效。")
        return self.root / "datasets" / f"{key}.json"

    def _run_path(self, key):
        if not re.fullmatch(r"[0-9a-f]{32}", str(key)):
            raise ValueError("训练任务编号无效。")
        return self.root / "runs" / str(key)

    def import_data(self, files):
        data = human_data.import_files(files)
        with self.lock:
            roles_path = self.root / "participant_roles.json"
            roles = json.loads(roles_path.read_text(encoding="utf-8")) if roles_path.exists() else {}
            for person, role in data["participants"].items():
                if person in roles and roles[person] != role:
                    raise ValueError(f"参与者 {person} 已在当前研究中登记为 {roles[person]}，不能改组。不同研究请使用独立数据目录。")
            path = self._dataset_path(data["dataset_id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            human.write_json(path, data)
            roles.update(data["participants"])
            human.write_json(roles_path, roles)
        return _import_summary(data)

    def import_paths(self, paths):
        if not isinstance(paths, list) or not 1 <= len(paths) <= 100:
            raise ValueError("请填写 1～100 个本机采集目录。")
        directories = []
        for value in paths:
            if not isinstance(value, str):
                raise ValueError("采集目录必须是文字路径。")
            value = value.strip().strip('"')
            if not value or len(value) > 4096:
                raise ValueError("请填写有效的本机采集目录。")
            path = Path(value).expanduser()
            if (not path.is_absolute() or value.startswith(("\\\\", "//"))
                    or path.resolve().parent == path.resolve()):
                raise ValueError("请选择具体的本机采集目录，不能使用相对路径、网络共享或磁盘根目录。")
            directories.append(path)
        try:
            files = human_data.files_from_paths(directories)
        except OSError as exc:
            raise ValueError("无法读取采集目录，请确认路径存在且当前账户有读取权限。") from exc
        return self.import_data(files)

    def dataset(self, key):
        path = self._dataset_path(key)
        if not path.is_file():
            raise FileNotFoundError("数据集不存在，请先导入。")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["dataset_id"] != key or human_data.dataset_digest(data) != key:
            raise ValueError("数据集存储校验失败。")
        return data

    def list_state(self):
        with self.lock:
            datasets = [_import_summary(json.loads(p.read_text(encoding="utf-8")))
                        for p in sorted((self.root / "datasets").glob("*.json"))]
            jobs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("runs/*/job.json"))]
            active_path = self.root / "active_model.json"
            active = json.loads(active_path.read_text(encoding="utf-8")) if active_path.is_file() else None
        jobs.sort(key=lambda item: item.get("created_utc", ""))
        return {"datasets": datasets, "jobs": jobs, "active": active,
                "policy": "预分组按参与者锁定；同一数据集的一键训练评估复用既有结果，测试不参与调参。"}

    def start(self, dataset_id):
        data = self.dataset(dataset_id)
        # Validate before starting or exposing any test results.
        _check_training_input(data)
        with self.lock:
            if self.stop_event.is_set():
                raise ValueError("服务正在停止。")
            for path in self.root.glob("runs/*/job.json"):
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing["dataset_id"] == dataset_id and existing["state"] in ("running", "completed"):
                    return existing
                # Once a test has been opened, a failed report cannot silently trigger another training round.
                if existing["dataset_id"] == dataset_id and (path.parent / "artifacts" / "freeze.json").exists():
                    raise ValueError("此数据集已有冻结测试记录；请检查失败任务保留的产物，避免反复训练挑选测试成绩。")
                split_path = path.parent / "artifacts" / "splits.json"
                freeze_path = path.parent / "artifacts" / "freeze.json"
                if split_path.is_file() and freeze_path.is_file():
                    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
                    prior = json.loads(split_path.read_text(encoding="utf-8"))
                    if frozen.get("test_opened") and set(prior["test"]) & {r["participant"] for r in data["test"]}:
                        raise ValueError("这些测试参与者已用于先前的冻结测试；新的开发轮次需要新的独立测试参与者。既有结果仍可下载。")
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("已有真人训练任务运行，请等待完成。")
            run_id = uuid.uuid4().hex
            folder = self._run_path(run_id)
            folder.mkdir(parents=True)
            job = {"run_id": run_id, "dataset_id": dataset_id, "state": "running", "stage": "准备训练", "progress": 0.,
                   "created_utc": datetime.now(timezone.utc).isoformat()}
            human.write_json(folder / "job.json", job)
            self.thread = threading.Thread(target=self._train, args=(job, data), name="human-model-training", daemon=True)
            self.thread.start()
            return dict(job)

    def _train(self, job, data):
        folder = self._run_path(job["run_id"])
        last = [-1., ""]
        def progress(stage, fraction):
            if self.stop_event.is_set():
                raise InterruptedError("服务退出，训练已停止。")
            if fraction == last[0] and stage == last[1]:
                return
            last[:] = [fraction, stage]
            with self.lock:
                job.update(stage=stage, progress=fraction)
                human.write_json(folder / "job.json", job)
        try:
            result = human.run_training(data, folder / "artifacts", progress)
            with self.lock:
                job.update(state="completed", progress=1., stage="训练与独立测试完成", result=result)
                human.write_json(folder / "job.json", job)
        except Exception as exc:
            with self.lock:
                job.update(state="failed", error=str(exc), stage="任务停止，记录已保留")
                human.write_json(folder / "job.json", job)

    def job(self, run_id):
        path = self._run_path(run_id) / "job.json"
        with self.lock:
            if not path.is_file():
                raise FileNotFoundError("训练任务不存在。")
            return json.loads(path.read_text(encoding="utf-8"))

    def artifact(self, run_id, kind):
        allowed = {"model": "attention_model.json", "bundle": "model_bundle.zip", "report": "report.html",
                   "markdown": "report.md", "metrics": "evaluation.json", "predictions": "test_predictions.csv"}
        if kind not in allowed:
            raise ValueError("不支持的产物类型。")
        job = self.job(run_id)
        if job["state"] != "completed":
            raise ValueError("任务尚未完成，不能导出正式成果。")
        path = self._run_path(run_id) / "artifacts" / allowed[kind]
        if not path.is_file():
            raise FileNotFoundError("产物文件不存在。")
        return path

    def activate(self, run_id=None, bundle=None):
        if run_id is not None:
            bundle = json.loads(self.artifact(run_id, "model").read_text(encoding="utf-8"))
        human.validate_model(bundle)
        if not bundle.get("engine_fingerprint") or bundle["engine_fingerprint"] != engine_fingerprint():
            raise ValueError("模型的特征引擎与当前平台不一致；请统一版本后重新导出、训练。")
        policy = next((p.get("preprocessing_policy") for p in bundle["source_profiles"]), None)
        if bundle.get("implementation_fingerprint") != conditioning.implementation(policy):
            raise ValueError("模型的特征提取或时间聚合实现与当前平台不一致，请使用同一版本。")
        if not bundle.get("source_profiles"):
            raise ValueError("模型缺少采集设备与单位信息，无法启用实时推理。请导入完整采集导出目录。")
        with self.lock:
            human.write_json(self.root / "active_model.json", bundle)
        return {"active": True, "model_id": bundle["model_id"], "effective": "仅对新启动的会话生效，当前会话使用启动时冻结的模型。"}

    def deactivate(self):
        with self.lock:
            (self.root / "active_model.json").unlink(missing_ok=True)
        return {"active": False, "effective": "新会话不加载真人模型；当前会话继续使用启动时版本。"}

    def close(self):
        self.stop_event.set()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=10)


def load_active(data_dir):
    path = Path(data_dir) / "human_models" / "active_model.json"
    if not path.is_file():
        return None
    return human.validate_model(json.loads(path.read_text(encoding="utf-8")))
