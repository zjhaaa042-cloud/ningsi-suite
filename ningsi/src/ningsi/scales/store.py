"""量表作答与结果的落盘（JSONL 逐行追加，便于审计与复算）。"""

from __future__ import annotations

import json
from pathlib import Path


def append_record(path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


def read_records(path) -> list:
    target = Path(path)
    if not target.exists():
        return []
    out = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def save_scale_run(root, participant: str, session: str, run: str, result, responses=None, scale_code: str | None = None) -> Path:
    payload = {
        "kind": "scale",
        "participant": participant,
        "session": session,
        "run": run,
        "scale": result.code if hasattr(result, "code") else scale_code,
        "result": result.as_dict() if hasattr(result, "as_dict") else result,
    }
    if responses is not None:
        payload["responses"] = list(responses) if not isinstance(responses, dict) else dict(responses)
    return append_record(Path(root) / "scales" / f"sub-{participant}_ses-{session}_run-{run}_scales.jsonl", payload)
