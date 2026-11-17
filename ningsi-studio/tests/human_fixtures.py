"""Constructed SOFTWARE TEST fixtures, never evidence from real participants."""
import csv
import hashlib
import io
import json
import math

import numpy as np

from ningsi.models.logistic import FEATURE_NAMES
from ningsi.models.research_features import engine_fingerprint
from ningsi.signal import conditioning


def fixture_files(development=6, test=3, per_class=6, *, preprocessing_policy=None):
    rng = np.random.default_rng(42)
    files = {}
    fingerprint = engine_fingerprint()
    implementation = conditioning.implementation(preprocessing_policy)
    for index in range(development + test):
        person = f"QA{index:03d}"
        cohort = "development" if index < development else "test"
        session = f"software-fixture-{person}"
        source = {"data_origin": "real", "name": "FixtureEEG", "source_id": "qa-software-fixture",
                  "channels": 1, "srate": 100., "channel_labels": ["Fp1"], "reference": "QA-only",
                  "source_unit": "uV", "scale_to_uv": 1.}
        meta = {"participant": person, "visit": "01", "cohort": cohort, "session_id": session,
                "state": "completed", "source": source, "protocol": {"name": "attention-validation-v1"},
                "engine_fingerprint_at_capture": fingerprint,
                "implementation_fingerprint_at_capture": implementation,
                "notes": "CONSTRUCTED SOFTWARE FIXTURE. NOT HUMAN EEG. real flag exercises validation only."}
        exported = {"schema": "ningsi-capture-export-v1", "protocol_complete": True,
                    "engine_fingerprint_at_export": fingerprint, "engine_changed_since_capture": False,
                    "implementation_fingerprint_at_export": implementation}
        if preprocessing_policy:
            source.update(preprocessing_policy=preprocessing_policy, srate=250.)
            exported["preprocessing"] = conditioning.description(source["srate"])
        rows = []
        for n in range(per_class * 2):
            label = n % 2
            theta = .22 + label * .12 + rng.normal(0, .04)
            alpha = .25 + rng.normal(0, .02)
            beta = .30 - label * .10 + rng.normal(0, .04)
            row = {"participant": person, "session_id": session, "visit": "01", "cohort": cohort,
                   "data_origin": "real", "probe_id": f"p{n}", "phase": "sart_block_1", "practice": False,
                   "onset_lsl": 100 + n * 20, "choice": 1 if label == 0 else 3, "label": label,
                   "label_name": "on_task" if label == 0 else "internal_mind_wandering", "timeout": False,
                   "cancelled": False, "window_start_lsl": 88 + n * 20, "window_end_lsl": 98 + n * 20,
                   "windows_total": 4, "windows_usable": 4, "usable_seconds": 10, "baseline_valid": True,
                   "focus": float(np.clip(.65 - label * .30 + rng.normal(0, .17), .04, .96)), "focus_z": 0,
                   "rel_theta": theta, "rel_alpha": alpha, "rel_beta": beta,
                   "log_beta_theta": math.log(beta / theta), "log_alpha_beta": math.log(alpha / beta),
                   "log_theta_alpha": math.log(theta / alpha), "feature_usable": True,
                   "train_eligible": cohort == "development", "test_eligible": cohort == "test",
                   "binary_label_valid": True, "unavailable_reasons": ""}
            rows.append(row)
        csv_file = io.StringIO(newline="")
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        texts = {"manifest.json": json.dumps(meta, ensure_ascii=False), "export.json": json.dumps(exported),
                 "features.csv": "\ufeff" + csv_file.getvalue()}
        texts["SHA256.json"] = json.dumps({k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in texts.items()})
        files.update({f"{person}/{name}": text for name, text in texts.items()})
    return files
