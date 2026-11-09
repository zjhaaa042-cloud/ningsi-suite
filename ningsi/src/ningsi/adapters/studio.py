"""读取 bsense-dataset-studio 的产物，把真实试采记录并入凝思的历史与证据链。

对应上游列名：participant / session / task / run / quality_status / eeg_valid_ratio /
mean_reaction_time_s / reaction_time_cv 等（缺失列按空处理，不假设固定表结构）。
"""

from __future__ import annotations

import csv
from pathlib import Path


BEHAVIOR_FIELDS = (
    "mean_reaction_time_s",
    "median_reaction_time_s",
    "reaction_time_sd_s",
    "reaction_time_cv",
    "reaction_time_slope",
)
QUALITY_FIELDS = ("quality_status", "quality_grade", "eeg_valid_ratio", "fnirs_valid_ratio", "motion_artifact_ratio")


def find_records_csv(dataset_root) -> Path | None:
    root = Path(dataset_root)
    candidate = root / "derived" / "features" / "records.csv"
    return candidate if candidate.exists() else None


def load_rows(path) -> list:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize(row: dict) -> dict:
    """把一行上游记录整理成凝思历史记录格式。"""
    behavior = {field: _number(row.get(field)) for field in BEHAVIOR_FIELDS if field in row}
    quality = {field: row.get(field) for field in QUALITY_FIELDS if field in row}
    return {
        "source": "bsense-dataset-studio",
        "participant": row.get("participant"),
        "session": row.get("session"),
        "task": row.get("task"),
        "run": row.get("run"),
        "recorded_at": row.get("recorded_at") or "",
        "quality": quality,
        "behavior": behavior,
        "evidence": {"xdf_path": row.get("xdf_path"), "protocol_version": row.get("protocol_version")},
    }


def import_from_studio(dataset_root, writer) -> int:
    """把上游记录写入凝思历史；writer 为可调用对象（例如 history.append_record 的偏函数）。"""
    path = find_records_csv(dataset_root)
    if path is None:
        return 0
    count = 0
    for row in load_rows(path):
        writer(normalize(row))
        count += 1
    return count
