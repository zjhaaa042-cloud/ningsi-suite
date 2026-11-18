"""Refit the frozen public ATTLAPSE TCN candidate for Capture shadow scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import benchmark_focus_tcn as benchmark
import numpy as np

from attlapse_two_channel_eval import training_weights
from tcn_focus_numpy import fit_tcn


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_inputs(benchmark_dir, data_dir):
    evidence = json.loads((benchmark_dir / "metrics.json").read_text(encoding="utf-8"))
    archive = benchmark_dir / "sequences.npz"
    if evidence["input_sha256"] != sha256(archive):
        raise ValueError("Sequence archive differs from the frozen benchmark")
    feature_digest = sha256(data_dir / "features250.jsonl")
    scope = (benchmark.WINDOW, benchmark.GATE, list(benchmark.BIPOLAR),
             list(benchmark.SEEDS), benchmark.TCN_SETTINGS, feature_digest)
    if scope != tuple(evidence[name] for name in (
            "window", "quality", "features", "tcn_seeds", "tcn_settings", "reference_feature_sha256")):
        raise ValueError("Refit inputs or settings differ from the frozen benchmark")
    for filename, expected in evidence["implementation_sha256"].items():
        if sha256(Path(__file__).with_name(filename)) != expected:
            raise ValueError(f"Frozen benchmark source no longer matches: {filename}")
    with np.load(archive, allow_pickle=False) as data:
        if data["source_feature_sha256"].item() != feature_digest:
            raise ValueError("Cached sequences are not from the frozen feature archive")
        arrays = {key: data[key] for key in ("features", "mask", "label", "participants", "probes")}
    features, mask = arrays["features"], arrays["mask"]
    if (features.shape != tuple(evidence["input_shape"]) or mask.shape != features.shape[:2]
            or mask.dtype != np.bool_ or not np.isfinite(features[mask]).all()
            or not np.isin(arrays["label"], (0, 1)).all()):
        raise ValueError("Invalid ATTLAPSE sequence inputs")
    return evidence, arrays


def fit_members(x, mask, y, groups):
    observed = x[mask]
    mean, scale = observed.mean(axis=0), observed.std(axis=0)
    scale[scale < 1e-9] = 1.
    scaled = np.where(mask[:, :, None], (x - mean) / scale, 0.)
    weights = training_weights(y, groups)
    models = [fit_tcn(scaled, mask, y, weights, seed=seed, **benchmark.TCN_SETTINGS)
              for seed in benchmark.SEEDS]
    return mean, scale, models


def refit(benchmark_dir, data_dir):
    evidence, arrays = load_frozen_inputs(benchmark_dir, data_dir)
    x, mask, y, groups = (arrays[name] for name in ("features", "mask", "label", "participants"))
    accepted = mask.any(axis=1)
    if int(accepted.sum()) != evidence["evaluation"]["scores"]["ordered_tcn"]["scored_labelled"]:
        raise ValueError("Probe coverage differs from the frozen benchmark")
    mean, scale, models = fit_members(x[accepted], mask[accepted], y[accepted], groups[accepted])
    return {
        "schema": "ningsi-focus-tcn-v1",
        "input_rate_hz": 250,
        "lookback_sec": [-18, -2],
        "feature_names": evidence["features"],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "members": [{"parameters": {name: value.tolist() for name, value in model._parameters.items()}}
                    for model in models],
        "provenance": {
            "dataset_doi": evidence["data_doi"],
            "source_rate_hz": evidence["source_rate_hz"],
            "dataset_probes": int(len(accepted)),
            "dataset_participants": int(len(np.unique(groups))),
            "training_probes": int(accepted.sum()),
            "training_participants": int(len(np.unique(groups[accepted]))),
            "training_labels": np.bincount(y[accepted], minlength=2).tolist(),
            "source_sequences_sha256": evidence["input_sha256"],
            "benchmark_sha256": sha256(benchmark_dir / "metrics.json"),
            "benchmark_implementation_sha256": evidence["implementation_sha256"],
            "training_settings": benchmark.TCN_SETTINGS,
            "seeds": list(benchmark.SEEDS),
            "parameters_per_member": models[0].parameter_count,
            "quality_gate": benchmark.GATE,
            "numpy_version": np.__version__,
            "label_1": "self_reported_spontaneous_or_deliberate_mind_wandering",
            "scope": "experimental shadow score for SART probes; no target-device calibration",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=benchmark.OUTPUT)
    parser.add_argument("--data", type=Path, default=benchmark.DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = refit(args.benchmark, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"{args.output} · sha256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
