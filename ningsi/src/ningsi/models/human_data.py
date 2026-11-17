"""Strict, bounded import of the standalone collector's exports and datasets.

No pickle or arbitrary executable model format is accepted. Source flags are
operator declarations; hashes establish consistency, not proof of human EEG.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import io
import json
import math
from ningsi.acquisition.timing import BIOMULTI_REC_TIMING
from pathlib import Path, PurePosixPath
import re

from ningsi.acquisition.eeg import source_type_conflict, unit_conflict
from ningsi.signal import conditioning
from .capture_protocols import RESEARCH_PROTOCOLS
from .logistic import FEATURE_NAMES
from .research_features import CONTRACT, ENGINE_FILES

ALLOWED_FILES = {"manifest.json", "export.json", "features.csv", "SHA256.json", "baseline.json",
                 "dataset.json", "development.csv", "test.csv", "excluded.csv"}
MAX_BYTES = 12 * 1024 * 1024
MAX_ROWS = 50000
LABELS = {"0": "on_task", "1": "internal_mind_wandering"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def dataset_digest(data):
    identity = {k: data[k] for k in ("development", "test", "excluded", "engine_fingerprint", "source_profiles")}
    if "implementation_fingerprint" in data:
        identity["implementation_fingerprint"] = data["implementation_fingerprint"]
        identity["session_metadata"] = data["session_metadata"]
    return digest(identity)


def _json(text):
    result = json.loads(text.lstrip("\ufeff"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("JSON 不允许 NaN/Inf")))
    if not isinstance(result, dict):
        raise ValueError("数据清单必须是 JSON 对象。")
    return result


def _rows(text):
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff"), newline=""))
    names = reader.fieldnames or []
    required = {"participant", "session_id", "visit", "probe_id", "cohort", "data_origin", "label",
                "choice", "practice", "baseline_valid", "feature_usable", "binary_label_valid",
                "train_eligible", "test_eligible", "unavailable_reasons", *FEATURE_NAMES}
    if len(set(names)) != len(names) or not required <= set(names):
        raise ValueError("CSV 缺少采集器字段或有重复列，请使用完整导出。")
    result = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV 列数不一致。")
        result.append(row)
        if len(result) > MAX_ROWS:
            raise ValueError("单次导入超过 50000 条探针。")
    return result


def _bool(row, field):
    value = str(row.get(field, ""))
    if value not in ("True", "False"):
        raise ValueError(f"{field} 必须为采集器的 True/False 标记。")
    return value == "True"


def _fingerprint(value, names=ENGINE_FILES):
    if not isinstance(value, dict) or set(value) != set(names) or any(
            not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in value.values()):
        raise ValueError("缺少有效的特征引擎指纹，请从原始采集记录重新导出。")
    return value


def _check_hashes(files, checksums, names):
    for name in names:
        if not isinstance(checksums, dict) or hashlib.sha256(files[name].encode("utf-8")).hexdigest() != checksums.get(name):
            raise ValueError(f"采集文件校验失败，请从原记录重新导出（{name}）。")


def source_profile(source):
    """Fields that can be checked against a live stream, plus operator metadata."""
    fields = ("name", "source_id", "channels", "srate", "channel_labels", "reference", "scale_to_uv", "source_unit", "declared_units")
    profile = {name: source.get(name) for name in fields}
    if source.get("timestamp_policy"):
        profile["timestamp_policy"] = source["timestamp_policy"]
    if source.get("unit_basis"):
        profile["unit_basis"] = source["unit_basis"]
    if source.get("preprocessing_policy"):
        profile["preprocessing_policy"] = source["preprocessing_policy"]
    return profile


def files_from_paths(paths):
    files = {}
    total_bytes = 0
    for index, item in enumerate(paths):
        root = Path(item).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"请选择采集导出目录或合并数据集目录：{root}")
        roots = [root] if (root / "export.json").is_file() or (root / "dataset.json").is_file() else [
            p.parent for p in sorted(root.rglob("export.json")) if "_MEI" not in str(p)]
        if not roots:
            raise ValueError("目录中没有 export.json 或 dataset.json；请先在采集器停止采集并导出。")
        for n, folder in enumerate(roots):
            for name in ALLOWED_FILES:
                path = folder / name
                if path.is_file():
                    size = path.stat().st_size
                    total_bytes += size
                    if total_bytes > MAX_BYTES or len(files) >= 3000:
                        raise ValueError("特征文件合计超过 12 MiB 或文件过多，请分批导入。")
                    files[f"input{index}_{n}/{name}"] = path.read_bytes().decode("utf-8")
    return files


def import_files(files):
    if not isinstance(files, dict) or not files or len(files) > 3000:
        raise ValueError("请选择采集器导出目录，单次最多导入 3000 个特征文件。")
    groups = {}
    total = 0
    for key, content in files.items():
        path = PurePosixPath(str(key).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path) or path.name not in ALLOWED_FILES:
            raise ValueError("仅允许导入采集器的特征 CSV 与配套清单。")
        if not isinstance(content, str):
            raise ValueError("导入文件必须是 UTF-8 文本。")
        total += len(content.encode("utf-8"))
        if total > MAX_BYTES:
            raise ValueError("特征文件合计超过 12 MiB，请分批整理数据集。")
        group = groups.setdefault(str(path.parent), {})
        if path.name in group:
            raise ValueError("重复的输入文件。")
        group[path.name] = content
    rows, fingerprints, implementations, profiles, sessions, notices = [], [], [], [], {}, []
    legacy_metadata = False
    for group in groups.values():
        if "dataset.json" in group:
            if not {"development.csv", "test.csv", "excluded.csv"} <= group.keys():
                raise ValueError("合并数据集需要 development.csv、test.csv、excluded.csv 和 dataset.json。")
            meta = _json(group["dataset.json"])
            if meta.get("schema") != "ningsi-capture-dataset-v1" or meta.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError("不支持的数据集规格或特征顺序。")
            if meta.get("label_column") != "label" or meta.get("group_column") != "participant":
                raise ValueError("数据集标签/分组定义不匹配。")
            if "file_sha256" in meta or meta.get("engine_fingerprint"):
                _check_hashes(group, meta.get("file_sha256", {}), ("development.csv", "test.csv", "excluded.csv"))
            for name, role in (("development.csv", "development"), ("test.csv", "test"), ("excluded.csv", "excluded")):
                part = _rows(group[name])
                count_key = {"development": "development_rows", "test": "test_rows", "excluded": "excluded_rows"}[role]
                if meta.get(count_key) != len(part):
                    raise ValueError("数据集行数与清单不一致。")
                for row in part:
                    if role != "excluded" and row["cohort"] != role:
                        raise ValueError("数据集人员分组不一致。")
                    if role != "excluded" and (not _bool(row, "train_eligible" if role == "development" else "test_eligible")
                                               or row["data_origin"] != "real"):
                        raise ValueError("开发/测试文件中含未通过采集验收的探针。")
                    if meta.get("participants", {}).get(row["participant"]) != row["cohort"]:
                        raise ValueError("数据集参与者清单不一致。")
                    if role == "excluded" and (_bool(row, "train_eligible") or _bool(row, "test_eligible")) and row["data_origin"] == "real":
                        raise ValueError("excluded.csv 含标记可用的真人样本。")
                if role != "excluded" and meta.get("engine_fingerprint") and any(
                        row["session_id"] not in meta.get("session_metadata", {}) for row in part):
                    raise ValueError("合格行缺少对应的采集会话清单。")
                rows.extend(part)
            if meta.get("engine_fingerprint"):
                fingerprints.append(_fingerprint(meta["engine_fingerprint"]))
                if not meta.get("session_metadata"):
                    raise ValueError("新版合并数据集缺少会话完成清单，请重新合并原始导出。")
            else:
                legacy_metadata = True
                notices.append("旧合并数据集未记录引擎/设备清单，可训练评估，但不能启用实时推理；建议导入原始导出目录。")
            for profile in meta.get("source_profiles", []):
                if unit_conflict(profile):
                    raise ValueError("合并数据的设备声明单位与换算倍率不符，请从原始记录重新导出并合并。")
                profiles.append(profile)
            sessions.update(meta.get("session_metadata", {}))
            implementation = meta.get("implementation_fingerprint")
            policy = meta.get("preprocessing_policy")
            if any(p.get("preprocessing_policy") != policy for p in meta.get("source_profiles", [])):
                raise ValueError("数据集设备与特征预处理版本不一致。")
        elif "export.json" in group:
            if not {"manifest.json", "features.csv"} <= group.keys():
                raise ValueError("每个导出目录需要 manifest.json、export.json 和 features.csv。")
            meta, exported = _json(group["manifest.json"]), _json(group["export.json"])
            if exported.get("schema") != "ningsi-capture-export-v1":
                raise ValueError("不支持的采集器导出规格。")
            if "session_id" in exported and exported["session_id"] != meta.get("session_id"):
                raise ValueError("采集清单与导出清单的会话编号不一致。")
            protocol = meta.get("protocol", {}).get("name")
            if "protocol_id" in exported and exported["protocol_id"] != protocol:
                raise ValueError("采集清单与导出清单的协议编号不一致。")
            fingerprint = _fingerprint(exported.get("engine_fingerprint_at_export"))
            fingerprints.append(fingerprint)
            if exported.get("checksum_required") and "SHA256.json" not in group:
                raise ValueError("该版本的采集导出缺少 SHA256.json 校验清单，请重新导出。")
            if "SHA256.json" in group:
                _check_hashes(group, _json(group["SHA256.json"]), ("manifest.json", "features.csv", "export.json"))
            implementation = exported.get("implementation_fingerprint_at_export")
            part = _rows(group["features.csv"])
            source = meta.get("source", {})
            policy = source.get("preprocessing_policy")
            if (exported.get("preprocessing") or {}).get("policy") != policy:
                raise ValueError("采集清单与导出的预处理版本不一致。")
            captured_impl = meta.get("implementation_fingerprint_at_capture")
            expected_capture = implementation
            if exported.get("preprocessing_recomputed"):
                original = meta.get("source_at_capture")
                if policy != conditioning.POLICY or not isinstance(original, dict) or original.get("preprocessing_policy"):
                    raise ValueError("缺少旧记录复算的来源信息。")
                expected_capture = {k: v for k, v in (implementation or {}).items() if k != conditioning.FILE}
            conflict = source_type_conflict(source) or unit_conflict(source)
            if "probes_total" in exported and exported["probes_total"] != len(part):
                raise ValueError("探针数与导出清单不一致。")
            for row in part:
                if conflict and any(_bool(row, field) for field in ("feature_usable", "train_eligible", "test_eligible")):
                    raise ValueError(conflict + " 请用新版采集器从原始记录重新导出。")
                if any(row[name] != str(meta.get(name)) for name in ("participant", "session_id", "visit", "cohort")) or row["data_origin"] != source.get("data_origin"):
                    raise ValueError("特征 CSV 与采集清单不一致。")
                if _bool(row, "train_eligible") or _bool(row, "test_eligible"):
                    if source.get("timestamp_policy") == BIOMULTI_REC_TIMING and (
                            (exported.get("native_alignment") or {}).get("status") != "matched"
                            or not source.get("scale_to_uv")):
                        raise ValueError("原生双录训练特征缺少设备对时或 EEG 单位。")
                    if (meta.get("state") != "completed" or not exported.get("protocol_complete")
                            or protocol not in RESEARCH_PROTOCOLS
                            or exported.get("engine_changed_since_capture")
                            or (captured_impl is not None and captured_impl != expected_capture)
                            or meta.get("engine_fingerprint_at_capture") != fingerprint):
                        raise ValueError("标记可用的样本来自未完成协议或变化的特征引擎。")
            rows.extend(part)
            if not conflict and source.get("data_origin") == "real" and meta.get("cohort") == "development":
                profiles.append(source_profile(source))
            sessions[meta["session_id"]] = {"participant": meta["participant"], "cohort": meta["cohort"],
                                             "state": meta.get("state"), "protocol_complete": exported.get("protocol_complete"),
                                             "protocol": protocol}
        else:
            # Browser selection can include a raw-session manifest beside its export.
            continue
        if policy not in (None, conditioning.POLICY) or (policy and implementation is None):
            raise ValueError("预处理版本未知或缺少相应的实现指纹。")
        files = ["models/research_features.py", "models/logistic.py"]
        if policy:
            files.append(conditioning.FILE)
        implementations.append(_fingerprint(implementation, files)
                               if implementation is not None else None)
    if not fingerprints and not rows:
        raise ValueError("没有采集器导出数据，请在停止采集后选择导出目录。")
    if len(rows) > MAX_ROWS:
        raise ValueError("单次导入超过 50000 条探针。")
    if fingerprints and len({canonical(f) for f in fingerprints}) != 1:
        raise ValueError("不能混用不同版本的特征引擎数据。")
    known_implementations = {canonical(f) for f in implementations if f is not None}
    if (len(known_implementations) > 1 or
            any(f and conditioning.FILE in f for f in implementations) and None in implementations):
        raise ValueError("不能混用不同特征提取实现的数据。")
    implementation = implementations[0] if all(implementations) else None
    if implementation is None:
        notices.append("部分旧导出缺少特征提取实现指纹。可离线评估，实时加载请用本版采集器从原始记录重新导出。")
    seen, cohorts, development, test, excluded = set(), {}, [], [], []
    for row in rows:
        person = row["participant"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", person) or not row["session_id"] or not row["probe_id"]:
            raise ValueError("匿名参与者编号或会话/探针编号无效。")
        if person in cohorts and cohorts[person] != row["cohort"]:
            raise ValueError(f"同一参与者跨分组，存在泄漏：{person}")
        cohorts[person] = row["cohort"]
        key = (row["session_id"], row["probe_id"])
        if key in seen:
            raise ValueError("重复的会话/探针，不能累计为新样本。")
        seen.add(key)
        train, heldout = _bool(row, "train_eligible"), _bool(row, "test_eligible")
        if train and heldout or train and row["cohort"] != "development" or heldout and row["cohort"] != "test":
            raise ValueError("训练/测试资格与预分组不一致。")
        if not (train or heldout) or row["data_origin"] != "real":
            excluded.append(row)
            continue
        session = sessions.get(row["session_id"])
        if session and (session.get("participant") != person or session.get("cohort") != row["cohort"]
                        or session.get("state") != "completed" or not session.get("protocol_complete")
                        or session.get("protocol") not in RESEARCH_PROTOCOLS):
            raise ValueError("数据集合格行与协议完成记录不一致。")
        if (_bool(row, "practice") or not all(_bool(row, k) for k in ("baseline_valid", "feature_usable", "binary_label_valid"))
                or row["label"] not in LABELS or row.get("unavailable_reasons")):
            raise ValueError("标记可用的样本未通过基线、标签或信号质控。")
        if row["choice"] not in ("1", "3", "5") or (row["label"] == "0") != (row["choice"] == "1"):
            raise ValueError("标签方向与探针原始回答不一致（0=专注，1=内部走神）。")
        if row.get("label_name", LABELS[row["label"]]) != LABELS[row["label"]]:
            raise ValueError("label_name 与标签含义不一致。")
        if not all(math.isfinite(float(row[name])) for name in FEATURE_NAMES):
            raise ValueError("可用特征含 NaN/Inf。")
        if any(not 0 <= float(row[k]) <= 1 for k in ("rel_theta", "rel_alpha", "rel_beta")):
            raise ValueError("相对功率超出 [0,1]。")
        if not math.isfinite(float(row.get("focus", "nan"))) or not 0 <= float(row["focus"]) <= 1:
            raise ValueError("缺少有效的原专注指数，无法进行配对基线比较。")
        count, usable = int(row.get("windows_total", 0)), int(row.get("windows_usable", 0))
        seconds = float(row.get("usable_seconds", 0))
        if _bool(row, "timeout") or _bool(row, "cancelled"):
            raise ValueError("超时或取消的探针不能用于训练或测试。")
        if not 3 <= usable <= count or usable < .6 * count or not math.isfinite(seconds) or not 6 <= seconds <= 10.000001:
            raise ValueError("可用样本的窗口覆盖不足。")
        (development if train else test).append(row)
    profiles = sorted({canonical(p): p for p in profiles if p.get("scale_to_uv") is not None}.values(), key=canonical)
    result = {"schema": "ningsi-human-dataset-v1", "contract": CONTRACT, "labels": LABELS,
              "engine_fingerprint": fingerprints[0] if fingerprints and not legacy_metadata else None,
              "implementation_fingerprint": implementation,
              "source_profiles": profiles, "session_metadata": sessions, "participants": cohorts,
              "development": sorted(development, key=lambda r: (r["participant"], r["session_id"], r["probe_id"])),
              "test": sorted(test, key=lambda r: (r["participant"], r["session_id"], r["probe_id"])),
              "excluded": sorted(excluded, key=lambda r: (r["participant"], r["session_id"], r["probe_id"])), "notices": sorted(set(notices))}
    for session in sessions.values():
        person, cohort = session["participant"], session["cohort"]
        if person in cohorts and cohorts[person] != cohort:
            raise ValueError(f"同一参与者跨分组：{person}")
        cohorts[person] = cohort
    result["dataset_id"] = dataset_digest(result)
    return result


def summarize(data):
    def part(rows):
        return {"rows": len(rows), "participants": len({r["participant"] for r in rows}),
                "labels": dict(Counter(r["label"] for r in rows))}
    reasons = Counter()
    pilot = [r for r in data["excluded"] if r["cohort"] == "pilot"]
    formal = [r for r in pilot if not _bool(r, "practice")]
    protocols = Counter(s.get("protocol") for s in (data.get("session_metadata") or {}).values()
                        if s.get("protocol") in RESEARCH_PROTOCOLS)
    for row in data["excluded"]:
        codes = set(filter(None, row.get("unavailable_reasons", "").split("|")))
        if row["cohort"] in ("pilot", "unassigned"):
            codes.add("pilot_cohort" if row["cohort"] == "pilot" else "unassigned_cohort")
        reasons.update(codes)
    return {"dataset_id": data["dataset_id"], "development": part(data["development"]), "test": part(data["test"]),
            "excluded_rows": len(data["excluded"]), "excluded_reasons": dict(reasons), "notices": data["notices"],
            "source_profiles": data["source_profiles"], "labels": LABELS,
            "research_protocols": [{"id": key, "label": RESEARCH_PROTOCOLS[key], "sessions": count}
                                   for key, count in sorted(protocols.items())],
            "pilot": {"participants": sum(role == "pilot" for role in data["participants"].values()), "formal_probes": len(formal),
                      "usable_eeg": sum(_bool(r, "feature_usable") for r in formal),
                      "binary_labels": sum(_bool(r, "binary_label_valid") for r in formal)},
            "live_compatible_metadata": bool(data["engine_fingerprint"] and data.get("implementation_fingerprint")
                                             and data["source_profiles"])}
