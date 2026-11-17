"""Verify files downloaded through the desktop against the observed training run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observed = json.loads(args.observations.read_text(encoding="utf-8"))
    assert observed["software_fixtures_only"] and not observed["human_accuracy_claim"]
    assert observed["entry_loaded"]
    for key in ("data_import", "one_click_training", "report_preview", "model_activation"):
        assert observed[key]["passed"], key
    assert observed["model_activation"]["persisted_after_navigating_away_and_back"]
    run_id = observed["one_click_training"]["run_id"]
    run = args.data / "human_models" / "runs" / run_id
    job = json.loads((run / "job.json").read_text(encoding="utf-8"))
    assert job["run_id"] == run_id and job["state"] == "completed"
    artifacts = run / "artifacts"
    model = json.loads((artifacts / "attention_model.json").read_text(encoding="utf-8"))
    active = json.loads((args.data / "human_models" / "active_model.json").read_text(encoding="utf-8"))
    assert active["model_id"] == model["model_id"] == observed["report_preview"]["model_id"]
    assert model["model_id"][:16] in observed["model_activation"]["observed_status"]
    evaluation = json.loads((artifacts / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["model_id"] == model["model_id"]
    for role in ("development", "test"):
        assert evaluation["data"][role]["participants"] == observed["data_import"][f"{role}_participants"]
        assert evaluation["data"][role]["rows"] == observed["data_import"][f"{role}_probes"]
    assert evaluation["evidence_status"] == "exploratory_insufficient_participants"
    assert observed["report_preview"]["exploratory_banner_visible"]
    report = (artifacts / "report.html").read_text(encoding="utf-8")
    assert "探索性结果" in report and model["model_id"] in report
    downloads = {}
    for name in observed["desktop_download_buttons"]:
        downloaded = args.downloads / name
        expected = sha(artifacts / name)
        assert sha(downloaded) == expected, f"Desktop download differs: {name}"
        downloads[name] = {"sha256": expected, "bytes": downloaded.stat().st_size}
    with zipfile.ZipFile(args.downloads / "model_bundle.zip") as bundle:
        assert bundle.testzip() is None
        expected_hashes = json.loads(bundle.read("SHA256.json"))
        for name, expected in expected_hashes.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == expected, name
            assert (artifacts / name).read_bytes() == bundle.read(name), name
    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "exe_sha256": sha(args.exe),
        "complete_desktop_workflow": True,
        "software_fixtures_only": True,
        "human_accuracy_claim": False,
        "ui_observations": observed,
        "run_id": run_id,
        "model_id": model["model_id"],
        "downloaded_files_match_run_artifacts": downloads,
        "bundle_crc_and_all_member_hashes": True,
        "active_model_matches_exported_model": True,
        "real_person_or_hardware_model_validation": False,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(args.output), "exe_sha256": result["exe_sha256"],
                      "desktop_workflow": True, "downloads_verified": len(downloads),
                      "model_matches_active": True}, indent=2))


if __name__ == "__main__":
    main()
