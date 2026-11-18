"""NumPy inference for the frozen, experimental 250 Hz probe classifier."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .attlapse_shadow import FEATURE_NAMES, LOOKBACK, RATE
from .human import sigmoid


MODEL_FILE = "focus_tcn_250.json"
MODEL_SHA256 = "9c6d13e2661efde7773288da6164da01c26a9d2d433e30cbaefafcaab1020b28"
SCHEMA = "ningsi-focus-tcn-v1"
N_FRAMES = 6
CHANNELS = 8
DILATIONS = (1, 2)
MEMBER_COUNT = 3
PARAMETER_SHAPES = {
    "input.weight": (len(FEATURE_NAMES), CHANNELS),
    "input.bias": (CHANNELS,),
    "block0.weight": (3, CHANNELS, CHANNELS),
    "block0.bias": (CHANNELS,),
    "block1.weight": (3, CHANNELS, CHANNELS),
    "block1.bias": (CHANNELS,),
    "head.weight": (CHANNELS,),
    "head.bias": (1,),
}


@dataclass(frozen=True)
class FocusTcn:
    model_id: str
    mean: np.ndarray
    scale: np.ndarray
    members: tuple[dict[str, np.ndarray], ...]


def _array(values, shape):
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid TCN parameter shape or value")
    result.setflags(write=False)
    return result


def load_candidate(path=None):
    """Load the frozen parameters and identify the exact bytes used for scoring."""
    model_path = Path(path) if path is not None else Path(__file__).with_name(MODEL_FILE)
    data = model_path.read_bytes()
    model_id = hashlib.sha256(data).hexdigest()
    if path is None and model_id != MODEL_SHA256:
        raise ValueError("Frozen TCN bundle checksum mismatch")
    payload = json.loads(data)
    if (payload["schema"] != SCHEMA or payload["input_rate_hz"] != RATE
            or tuple(payload["lookback_sec"]) != LOOKBACK
            or tuple(payload["feature_names"]) != FEATURE_NAMES
            or len(payload["members"]) != MEMBER_COUNT):
        raise ValueError("Unexpected TCN model specification")
    mean = _array(payload["mean"], (len(FEATURE_NAMES),))
    scale = _array(payload["scale"], (len(FEATURE_NAMES),))
    if np.any(scale <= 0):
        raise ValueError("Invalid TCN feature scale")
    members = []
    for member in payload["members"]:
        parameters = member["parameters"]
        if parameters.keys() != PARAMETER_SHAPES.keys():
            raise ValueError("Unexpected TCN parameters")
        members.append({name: _array(parameters[name], shape) for name, shape in PARAMETER_SHAPES.items()})
    return FocusTcn(model_id, mean, scale, tuple(members))


def _member_score(parameters, values, valid):
    hidden = np.maximum(values @ parameters["input.weight"] + parameters["input.bias"], 0)
    hidden[~valid] = 0
    for block, dilation in enumerate(DILATIONS):
        weight = parameters[f"block{block}.weight"]
        conv = np.broadcast_to(parameters[f"block{block}.bias"], hidden.shape).copy()
        for tap, kernel in enumerate(weight):
            lag = tap * dilation
            conv[lag:] += hidden[: N_FRAMES - lag] @ kernel
        hidden = np.maximum(hidden + conv, 0)
        hidden[~valid] = 0
    pooled = hidden[valid].mean(axis=0)
    return sigmoid(pooled @ parameters["head.weight"] + parameters["head.bias"][0])


def score_sequence(model, values, mask):
    """Score an ordered sequence, with physical-QC exclusions at their original slots."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(mask)
    if (values.shape != (N_FRAMES, len(FEATURE_NAMES)) or valid.shape != (N_FRAMES,)
            or valid.dtype != np.bool_ or not valid.any() or not np.isfinite(values[valid]).all()):
        raise ValueError("Invalid TCN feature sequence or validity mask")
    features = np.zeros_like(values)
    features[valid] = (values[valid] - model.mean) / model.scale
    score = float(np.mean([_member_score(member, features, valid) for member in model.members]))
    if not np.isfinite(score):
        raise FloatingPointError("Non-finite TCN score")
    return score
