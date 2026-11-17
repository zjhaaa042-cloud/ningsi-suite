"""Research scores for reported stress and emotional valence on paired frontal EEG."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .human import logits, sigmoid


SPEC = "frontal-affect-v1"
BUNDLE = "frontal_affect.json"
DIMENSIONS = ("stress", "valence")
BANDS = ("theta", "alpha", "beta")
FEATURES = tuple(f"log_rel_{band}" for band in BANDS)


def feature_vector(window):
    powers = np.array([window.rel.get(band, np.nan) for band in BANDS], dtype=float)
    if not np.isfinite(powers).all() or np.any(powers <= 0) or np.any(powers > 1):
        raise ValueError("Invalid relative band powers")
    return np.log(powers)


@lru_cache(maxsize=1)
def load_models():
    data = json.loads(Path(__file__).with_name(BUNDLE).read_text(encoding="utf-8"))
    if (data["spec"] != SPEC or data["features"] != list(FEATURES)
            or data["electrodes"] != ["Fp1", "Fp2"] or data["input_rate_hz"] != 250):
        raise ValueError("Invalid frontal affect feature contract")
    for name in DIMENSIONS:
        model = data["models"][name]
        params = model["parameters"]
        encoded = json.dumps(params, sort_keys=True, allow_nan=False).encode()
        if sha256(encoded).hexdigest() != model["model_id"]:
            raise ValueError("Invalid affect model checksum")
        for key in ("mean", "scale", "weights"):
            v = np.asarray(params[key], dtype=float)
            if v.shape != (len(FEATURES),) or not np.isfinite(v).all():
                raise ValueError("Invalid affect parameters")
        if np.min(params["scale"]) <= 0 or not np.isfinite(params["bias"]):
            raise ValueError("Invalid affect scale or bias")
    return data["models"]


def predict_window(window, channel_labels, srate):
    """Use one quality-approved 4 s window. Scores are uncalibrated research outputs."""
    labels = [str(s).strip().casefold() for s in (channel_labels or ())]
    missing = ""
    if window is None or not window.usable:
        missing = "eeg_quality_failed"
    elif window.channels != 2 or sorted(labels) != ["fp1", "fp2"]:
        missing = "requires_fp1_fp2"
    elif srate != 250 or window.srate != srate:
        missing = "requires_250hz"
    if not missing:
        try:
            values = feature_vector(window)
        except (TypeError, ValueError):
            missing = "invalid_band_features"
    if not missing:
        try:
            models = load_models()
        except (OSError, KeyError, TypeError, ValueError):
            missing = "model_unavailable"
    result = {}
    for name in DIMENSIONS:
        if missing:
            result[name] = dict(available=False, score=None, reason=missing, label="", model_id=None, display_class=False)
        else:
            model = models[name]
            score = float(sigmoid(logits(model["parameters"], values)))
            # A public-cohort group holdout is insufficient to label an individual
            # wearing a different device. Keep candidates visible only as research outputs.
            result[name] = dict(available=True, score=score, reason="", label="候选研究分数，目标设备尚未验证",
                                display_class=False,
                                model_id=model["model_id"], source=model["dataset"],
                                validation="public_participant_holdout", target_device_validated=False)
    return result
