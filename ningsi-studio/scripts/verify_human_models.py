"""Frozen-binary acceptance with constructed fixtures, never real-person evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "ningsi" / "src"))
from tests.human_fixtures import fixture_files
from ningsi.models.research_features import engine_fingerprint, implementation_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable, root = args.exe.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture_root, clean = root / "SOFTWARE_FIXTURES_NOT_HUMAN", root / "clean-cwd"
    clean.mkdir()
    files = fixture_files()
    for name, content in files.items():
        path = fixture_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUTF8"] = "1"
    env["NINGSI_STUDIO_DATA"] = str(root / "default-data")
    options = {"cwd": clean, "env": env, "encoding": "utf-8", "errors": "replace"}
    doctor = subprocess.run([str(executable), "doctor", "--data", str(root / "doctor")],
                            capture_output=True, timeout=60, **options)
    (root / "doctor.log").write_text(doctor.stdout + doctor.stderr, encoding="utf-8")
    assert doctor.returncode == 0, doctor.stdout + doctor.stderr
    assert str(PROJECT.parent / "ningsi" / "src") not in doctor.stdout, "EXE imported the source checkout"
    cli = subprocess.run([str(executable), "train-real", "--input", str(fixture_root), "--output", str(root / "cli-output")],
                         capture_output=True, timeout=90, **options)
    (root / "cli.log").write_text(cli.stdout + cli.stderr, encoding="utf-8")
    assert cli.returncode == 0, cli.stdout + cli.stderr
    model = json.loads((root / "cli-output" / "attention_model.json").read_text(encoding="utf-8"))
    assert model["engine_fingerprint"] == engine_fingerprint()
    assert model["implementation_fingerprint"] == implementation_fingerprint()
    hashes = json.loads((root / "cli-output" / "SHA256.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        assert hashlib.sha256((root / "cli-output" / name).read_bytes()).hexdigest() == expected, name
    with zipfile.ZipFile(root / "cli-output" / "model_bundle.zip") as archive:
        assert {"attention_model.json", "evaluation.json", "report.html", "freeze.json"} <= set(archive.namelist())
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        port = server_socket.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def request(path, payload=None, method=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(base + path, data=body, method=method,
                                     headers={"Content-Type": "application/json"} if body else {})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            return json.loads(data) if "application/json" in response.headers.get("Content-Type", "") else data

    log = (root / "service.log").open("w", encoding="utf-8")
    process = subprocess.Popen([str(executable), "serve", "--port", str(port), "--data", str(root / "studio")],
                               stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
                               **options)
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                request("/api/health")
                break
            except (OSError, ValueError):
                if process.poll() is not None:
                    raise RuntimeError("Frozen service exited during startup")
                time.sleep(.2)
        else:
            raise TimeoutError("Frozen service did not start")
        assert b"readCaptureFiles" in request("/js/views/human-models.js")
        for resource in ("js/views/human-models.js", "js/views/attention-prediction.js", "js/views/live.js", "js/api.js"):
            assert request("/" + resource) == (PROJECT / "web" / resource).read_bytes(), resource
        assert "/api/models/human/train" in request("/api/openapi.json")["paths"]
        dataset = request("/api/models/human/import", {"files": files})
        job = request("/api/models/human/train", {"dataset_id": dataset["dataset_id"]})
        run_id = job["run_id"]
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            job = request(f"/api/models/human/jobs/{run_id}")
            if job["state"] != "running":
                break
            time.sleep(.1)
        assert job["state"] == "completed", job
        downloaded = request(f"/api/models/human/jobs/{run_id}/artifacts/model")
        request("/api/models/human/activate", {"model": model})
        assert request("/api/models/human")["active"]["model_id"] == model["model_id"]
        request("/api/models/human/activate", {"run_id": run_id})
        assert request("/api/models/human")["active"]["model_id"] == downloaded["model_id"]
        report = request(f"/api/models/human/jobs/{run_id}/artifacts/report")
        assert "探索性结果".encode("utf-8") in report
        (root / "downloaded_report.html").write_bytes(report)
        bundle = request(f"/api/models/human/jobs/{run_id}/artifacts/bundle")
        (root / "downloaded_model_bundle.zip").write_bytes(bundle)
        request("/api/models/human/active", method="DELETE")
        result = {"software_fixtures_only": True, "human_accuracy_claim": False,
                  "exe": str(executable), "exe_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                  "clean_cwd_without_pythonpath": True, "doctor": True, "cli_training": True,
                  "embedded_engine_source_fingerprints_match": True,
                  "api_import_train_report_export_activation": True,
                  "exploratory_label_preserved": True, "file_checksums": True,
                  "bundled_frontend_matches_source": True,
                  "raw_eeg_or_fixture_model_in_distribution": False,
                  "development_participants": dataset["development"]["participants"],
                  "test_participants": dataset["test"]["participants"], "run_id": run_id}
        (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if process.poll() is None:
            # PyInstaller onefile has a bootloader parent and a service child.
            # Terminating only the parent leaves the child holding the EXE open.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        process.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
