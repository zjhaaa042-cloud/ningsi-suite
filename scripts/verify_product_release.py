"""Verify a packaged product on loopback using explicit software fixtures only."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import tomllib
import uuid as uuid_module
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--capture-exe", required=True, type=Path)
    parser.add_argument("--capture-doctor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = urlparse(args.url)
    if target.scheme != "http" or target.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Only loopback verification is supported")
    if args.output.exists():
        raise FileExistsError("Choose a new verification output")
    checks = []

    def request(path, body=None, *, binary=False, expected=200):
        req = Request(args.url.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type": "application/json"})
        try:
            response = urlopen(req, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            data = response.read()
            if response.status != expected:
                raise AssertionError(f"{path}: HTTP {response.status}, expected {expected}: {data[:150]!r}")
        return data if binary else json.loads(data)

    def passed(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    health = request("/api/health")
    root = Path(__file__).resolve().parents[1]
    studio_version = tomllib.loads((root / "ningsi-studio/pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    capture_version = tomllib.loads((root / "ningsi-capture/pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    cli_version = subprocess.run([str(args.exe), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    passed("packaged Studio version", cli_version == "ningsi-studio " + studio_version)
    passed("packaged service healthy", health["status"] == "ok" and health["service"] == "ningsi-studio")
    capture_doctor = json.loads(args.capture_doctor.read_text(encoding="utf-8-sig"))
    passed("packaged Capture version and LSL dependency", capture_doctor["version"] == capture_version and capture_doctor["liblsl_version"] > 0)
    index = request("/", binary=True).decode("utf-8")
    navigation = request("/js/main.js", binary=True).decode("utf-8")
    passed("product navigation has no competition route", "#/showcase" not in navigation and "A09" not in index)
    for asset in ("/js/views/device-profile.js", "/js/views/session-preflight.js", "/js/views/state-heatmap.js", "/js/training-metrics.js", "/css/workspace.css"):
        passed("packaged asset " + asset, len(request(asset, binary=True)) > 100)
    before = request("/api/sessions")["total"]
    device = "lsl:QA-SOFTWARE-FIXTURE-UNCONNECTED-" + uuid_module.uuid4().hex[:8]
    payload = {"device": device, "protocol": "ten_min"}
    result = request("/api/sessions/preflight", payload)
    passed("unconnected source cannot start formal assessment", result["ready"] is False)
    passed("quality and baseline await acquisition", any(row["key"] == "quality" and row["status"] == "pending" for row in result["checks"]))
    passed("preflight creates no session", request("/api/sessions")["total"] == before)
    passed("source has no invented profile", not request("/api/device-profile?" + urlencode({"device": device}))["saved"])
    profile = {"manufacturer": "SOFTWARE FIXTURE", "model": "Not a physical device", "electrodes": "QA1",
               "reference": "synthetic fixture", "srate": 250, "source_unit": "uV", "scale_to_uv": 1,
               "unit_basis": "synthetic software verification", "document_reference": "synthetic fixture"}
    saved = request("/api/device-profile", {"device": device, "profile": profile})
    passed("profile persists as an unverified declaration", saved["saved"] and saved["status"] == "pending_verification")
    passed("profile round trip", request("/api/device-profile?" + urlencode({"device": device}))["profile"] == saved["profile"])
    passed("different device profiles are isolated", not request("/api/device-profile?" + urlencode({"device": device + "-OTHER"}))["saved"])
    request("/api/device-profile", {"device": device, "profile": {"source_unit": "mV", "scale_to_uv": 1}}, expected=422)
    checks.append("contradictory calibration rejected")
    simulated = request("/api/sessions/preflight", {"device": "sim-bsense", "protocol": "ten_min"})
    passed("simulation excluded from formal preflight", simulated["ready"] is False)
    subject = request("/api/subjects", {"auto_id": True, "label": "Packaged software fixture"}, expected=201)
    created = request("/api/sessions", {"participant": subject["public_id"], "create_subject": False,
        "device": "sim-bsense", "protocol": "ten_min", "training_mode": "quick", "time_scale": .01}, expected=201)
    uuid = created["session"]["uuid"]
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        session = request(f"/api/sessions/{uuid}")
        if session["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(.5)
    passed("complete explicit simulation", session["status"] == "done" and session["device"] == "sim-bsense")
    heatmap = request(f"/api/sessions/{uuid}/heatmap")
    cells, coverage = heatmap["cells"], heatmap["summary"]
    passed("heatmap coverage uses all recorded windows", bool(cells)
           and coverage["total"] == len(cells)
           and coverage["scored"] == sum(cell["status"] == "scored" for cell in cells)
           and coverage["signal_usable"] == sum(cell["signal_usable"] is True for cell in cells)
           and all(cell["score"] is None for cell in cells if cell["status"] != "scored"))
    prepared = [cell for cell in cells if cell["status"] == "baseline_pending"]
    passed("valid preparation windows retain quality without invented scores", bool(prepared)
           and all(cell["signal_usable"] is True and cell["score"] is None for cell in prepared))
    passed("heatmap retains quality baseline and training phases", {"qc", "baseline_open", "training"}
           <= {cell["phase"] for cell in cells} and heatmap["evidence_source"] == "heatmap_metrics")
    heatmap_svg = request(f"/api/sessions/{uuid}/artifacts/heatmap_svg", binary=True).decode("utf-8")
    passed("exported heatmap distinguishes availability from score", "信号可用" in heatmap_svg
           and "低质量缺失" not in heatmap_svg)
    report = request(f"/api/sessions/{uuid}/report")
    passed("report retains simulation source", report["extras"]["source_kind"] == "sim")
    passed("standalone HTML report", b"<html" in request(f"/api/sessions/{uuid}/artifacts/report_html", binary=True).lower())
    with zipfile.ZipFile(io.BytesIO(request(f"/api/sessions/{uuid}/export.zip", binary=True))) as archive:
        passed("session export has valid CRC", archive.testzip() is None)
    profiles = {item["key"]: item for item in request("/api/config")["profiles"]}
    policy = profiles["effective_time"]["effective_time"]
    passed("fixed and effective time protocols remain distinct", profiles["ten_min"]["total_sec"] == 600
           and policy["baseline_target_seconds"] == 60 and policy["baseline_max_seconds"] == 180
           and policy["segment_target_seconds"] == 50 and policy["segment_max_seconds"] == 100
           and policy["max_acquisition_seconds"] == 1040)
    created = request("/api/sessions", {"participant": subject["public_id"], "create_subject": False,
        "device": "sim-bsense", "protocol": "effective_time", "training_mode": "quick", "time_scale": .02}, expected=201)
    effective_uuid = created["session"]["uuid"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        effective_session = request(f"/api/sessions/{effective_uuid}")
        if effective_session["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(.5)
    passed("complete effective time simulation", effective_session["status"] == "done"
           and effective_session["protocol"] == "effective_time")
    effective_report = request(f"/api/sessions/{effective_uuid}/report")
    timing = effective_report["extras"]["effective_time"]
    passed("effective acquisition outcome is explicit", timing["acquisition_complete"] is True
           and timing["collection_status"] == "complete" and effective_report["extras"]["source_kind"] == "sim")
    phases = {row["key"]: row for row in effective_session["phases"]}
    passed("effective mode keeps fixed PVT and practice budgets", phases["practice"]["duration_sec"] == 30
           and phases["pvt"]["duration_sec"] == 180)
    passed("effective baseline has real union coverage and valid quality", all(
        timing["phase_results"][key]["completed"] and timing["phase_results"][key]["baseline_valid"]
        and timing["phase_results"][key]["valid_seconds"] >= 60
        for key in ("baseline_open", "baseline_after")))
    effective_training = request(f"/api/sessions/{effective_uuid}/training")
    segments = effective_training["segments"]
    passed("effective feedback uses observed intervals and actual segment denominator", len(segments) == 4 and all(
        segment["effective_time"]["completed"] and segment["effective_time"]["timing_consistent"]
        and abs(segment["effective_time"]["valid_seconds"] - segment["stats"]["valid_seconds"]) < .0002
        and segment["effective_time"]["time_basis"] == "protocol_seconds"
        and abs(segment["effective_time"]["actual_elapsed_seconds"] - segment["duration_sec"] * .02) < .002
        and segment["stats"]["valid_seconds"] >= 50
        and segment["stats"]["valid_seconds"] <= segment["duration_sec"]
        and segment["stats"]["on_target_ratio"] <= segment["stats"]["valid_time_ratio"] + .0001
        and segment["stats"]["on_target_valid_ratio"] is not None for segment in segments))
    passed("effective training summary retains separate coverage and performance",
           effective_training["summary"]["mode"] == "effective_time"
           and effective_training["summary"]["valid_seconds"] >= 200
           and effective_training["summary"]["on_target_valid_ratio"] is not None)
    effective_html = request(f"/api/sessions/{effective_uuid}/artifacts/report_html", binary=True).decode("utf-8")
    passed("effective standalone report explains performance and missing time", "有效时长反馈训练" in effective_html
           and "有效时间内达标" in effective_html and "有效覆盖" in effective_html and "实际时长" in effective_html)
    with zipfile.ZipFile(io.BytesIO(request(f"/api/sessions/{effective_uuid}/export.zip", binary=True))) as archive:
        passed("effective session export has valid CRC", archive.testzip() is None)
    with args.exe.open("rb") as stream:
        exe_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with args.capture_exe.open("rb") as stream:
        capture_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"schema": "ningsi-product-frozen-verification-v1", "passed": True,
              "verified_at": datetime.now(timezone.utc).isoformat(),
              "exe_sha256": {"studio": exe_hash, "capture": capture_hash},
              "versions": {"studio": studio_version, "capture": capture_version},
              "check_count": len(checks), "checks": checks, "url": args.url,
              "fixture_sessions": {"fixed": uuid, "effective": effective_uuid},
              "hardware_or_human_validity_tested": False,
              "scope": "packaged EXE and explicit synthetic fixtures; no real participant answers or efficacy claims"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
