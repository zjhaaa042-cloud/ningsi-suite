"""Frozen external pilot model for blind, probe-locked scoring in Capture."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np

from .human import logits, sigmoid
from .human_data import digest
from .logistic import FEATURE_NAMES
from .research_features import CONTRACT, condition_window, engine_fingerprint, implementation_fingerprint


DATA_DOI = "10.5281/zenodo.23592484"
MODEL_SPEC = "attlapse-healthy-fp1-minus-fp2-v1"
MODEL_FILE = "attlapse_healthy_model.json"
MONTAGE = ["Fp1", "Fp2"]
SOURCE_LABELS = {"0": "on_task", "1": "spontaneous_or_deliberate_mind_wandering"}


def channel_indices(labels):
    """Look up actual electrode positions. The amplifier may report them in either order."""
    names = [str(label).strip().casefold() for label in labels]
    if any(names.count(label) != 1 for label in ("fp1", "fp2")):
        raise ValueError("候选模型需要各一个已确认的 Fp1 和 Fp2 电极。")
    return names.index("fp1"), names.index("fp2")


def bipolar_window(values, timestamps, rate, factor, indices):
    """Reject a bad input electrode, then compute the common-reference-cancelling derivation."""
    pair = np.asarray(values, dtype=float)[list(indices)]
    quality = condition_window(pair, timestamps, rate, factor)
    if not quality["window"].usable:
        return quality
    return condition_window(pair[0:1] - pair[1:2], timestamps, rate, factor)


def candidate_id(model):
    return digest({k: v for k, v in model.items() if k != "model_id"})


def montage_fingerprint():
    source = Path(__file__)
    if getattr(sys, "frozen", False):
        source = Path(sys._MEIPASS) / "capture_engine_source" / "models" / "attlapse.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def load_candidate(path=None):
    """Load a checked JSON model beside the EXE or from the installed module."""
    if path is None:
        path = Path(__file__).with_name(MODEL_FILE)
        if getattr(sys, "frozen", False):
            adjacent = Path(sys.executable).with_name(MODEL_FILE)
            if adjacent.is_file():
                path = adjacent
    path = Path(path)
    if not path.is_file():
        return None
    if path.stat().st_size > 512_000:
        raise ValueError("候选模型文件过大。")
    model = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(model, dict) or model.get("spec") != MODEL_SPEC
            or model.get("model_id") != candidate_id(model)
            or model.get("source_doi") != DATA_DOI or model.get("cohort") != "healthy_controls"
            or model.get("feature_contract") != CONTRACT or model.get("montage") != MONTAGE
            or model.get("labels") != SOURCE_LABELS or model.get("threshold") != 0.5
            or model.get("montage_fingerprint") != montage_fingerprint()
            or model.get("engine_fingerprint") != engine_fingerprint()
            or model.get("implementation_fingerprint") != implementation_fingerprint()):
        raise ValueError("候选模型的版本、特征定义或校验值不匹配。")
    parameters = model["parameters"]
    for name in ("mean", "scale", "weights"):
        values = np.asarray(parameters[name], dtype=float)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError(f"候选模型参数无效：{name}")
        if name == "scale" and np.any(values <= 0):
            raise ValueError("候选模型标准化倍率无效。")
    if not np.isscalar(parameters["bias"]) or not np.isfinite(parameters["bias"]):
        raise ValueError("候选模型偏置无效。")
    return model


def score_probe(candidate, features):
    vector = [features[name] for name in FEATURE_NAMES] if isinstance(features, dict) else features
    x = np.asarray(vector, dtype=float)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError("探针六维特征缺失或无效。")
    return float(sigmoid(logits(candidate["parameters"], x)))
