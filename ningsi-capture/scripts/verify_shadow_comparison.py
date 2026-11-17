"""Replay public 250 Hz epochs through the capture store, then compare with the frozen offline benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "ningsi/src"), str(REPO / "ningsi-capture/src")]

from attlapse_two_channel_250 import iter_probes
from attlapse_two_channel_eval import predict
from ningsi_capture.demo import VirtualClock
from ningsi_capture.export import export_session
from ningsi_capture.protocol import ProtocolConfig, TaskRunner
from ningsi_capture.storage import SessionStore


CASES = [("sub-T13", "probe-03", 1), ("sub-T13", "probe-01", None), ("sub-T07", "probe-01", 5)]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def replay(p, choice, output):
    # These QA sessions use public human EEG and the real-stream adapter. All are pilot, abbreviated,
    # deliberately aborted, and stored separately from actual local participant sessions.
    source = dict(kind="lsl", type="EEG", data_origin="real", name="ATTLAPSE public file replay",
                  channels=2, srate=250, channel_labels=["Fp2", "Fp1"], source_unit="mV", scale_to_uv=1000.,
                  reference="unknown in public source")
    clock = VirtualClock()
    origin = clock.seconds
    conf = ProtocolConfig.short()
    session = SessionStore(output / "public_replays", participant="PUBLIC_" + p["participant"], cohort="pilot",
                           visit=p["probe_id"], source=source, protocol=conf.as_dict(), clock=clock,
                           notes="QA replay of public ATTLAPSE EEG. Software clock and scripted responses; no local participant.")
    runner = TaskRunner(session, lambda _: clock.now(), conf, clock)
    runner.plan = [phase for phase in runner.plan if phase["name"] == "sart_1"]
    try:
        runner.start()
        # Align recording origin to the source's 2-second grid. Reverse the wire order and
        # change units to exercise the live adapter. Contaminate the excluded last 1.8 seconds.
        first = ((p["start_sample"] + 499) // 500) * 500
        offset = first - p["start_sample"]
        times = origin + np.arange(p["stop_sample"] - first) / 250
        onset = origin + p["onset_sec"] - first / 250
        values = (p["values_uv"][::-1, offset:].T / 1000).copy()
        values[times > onset - 1.8] += 3 * np.cos(np.arange((times > onset - 1.8).sum()))[:, None]
        for i in range(0, len(values), 64):
            j = min(i + 64, len(values))
            clock.seconds = min(onset, times[j - 1] + 1/250)
            session.append(values[i:j], times[i:j], offset=0.)
        clock.seconds = onset
        runner.show_probe()
        clock.seconds += .4 if choice else conf.probe_timeout + .01
        if choice is None:
            runner.tick()
        else:
            runner.key(str(choice))
        runner.abort("public_replay_complete")
    finally:
        session.close("aborted", "public_replay_complete")
    return session.path, onset - origin


def check_source(bench, out):
    bundle = json.loads((bench / "evaluation250/shadow_comparison_candidates.json").read_text(encoding="utf8"))
    reference = {}
    with (bench / "features250.jsonl").open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            if row["window"] == "pre18_2":
                reference[row["participant"], row["probe_id"], row["quality"]] = row
    evidence = []
    for participant, probe_id, choice in CASES:
        p = next(item for item in iter_probes(bench, participant) if item["probe_id"] == probe_id)
        session, elapsed = replay(p, choice, out)
        dest = out / f"export_{participant}_{probe_id}"
        summary = export_session(session, dest)
        row, = read_rows(dest / "probe_comparison.csv")
        assert int(row["probe_ordinal"]) == 1 and abs(float(row["task_elapsed_sec"]) - elapsed) < 1e-9
        assert row["choice"] == (str(choice) if choice else "") and row["label"] == {1:"0", 5:"1", None:""}[choice]
        assert row["response_status"] == ("binary" if choice else "timeout")
        assert row["time_comparable"] == str(choice is not None)
        assert float(row["window_start_lsl"]) == float(row["onset_lsl"]) - 18
        assert float(row["window_end_lsl"]) == float(row["onset_lsl"]) - 2
        physical = reference[participant, probe_id, "physical"]
        strict = reference[participant, probe_id, "strict"]
        for name, ref in (("physical", physical), ("strict", strict)):
            assert row[name+"_qc_status"] == ("failed" if ref["reasons"] else "passed"), row
        assert int(row["windows_total"]) == physical["windows_total"]
        assert int(row["windows_usable"]) == physical["windows_usable"]
        assert int(row["strict_windows_usable"]) == strict["windows_usable"]
        assert row["comparable"] == str(choice is not None and not physical["reasons"])
        assert row["live_input_window_status"] and row["live_input_window_end_lsl"]
        error = {}
        if not physical["reasons"]:
            x = np.array([[physical["features"][key] for key in bundle["feature_names"]]])
            for family in ("logistic", "rbf"):
                expected = float(predict(bundle["models"][family]["fitted_parameters"], x)[0])
                error[family] = abs(expected - float(row[family+"_mw_score"]))
                assert error[family] < 1e-9, (family, row, expected)
        else:
            assert row["status"] == "unavailable" and row["logistic_mw_score"] == row["rbf_mw_score"] == ""
        assert summary["train_eligible"] == summary["test_eligible"] == 0
        evidence.append({"session":str(session), "export":str(dest), "source_probe":probe_id, "row":row,
                         "absolute_score_error_vs_offline_refit":error})
    return evidence


def check_executable(exe, out, evidence):
    target = out / "exe"
    target.mkdir()
    subprocess.run([exe, "doctor", "--output", target/"doctor.json"], check=True, timeout=120)
    doctor = json.loads((target/"doctor.json").read_text(encoding="utf8"))
    sample = evidence[0]["row"]
    assert doctor["shadow_250_hz"]["model_ids"] == {k:sample[k+"_model_id"] for k in ("logistic","rbf")}
    assert doctor["shadow_250_hz"]["bundle_id"] == sample["model_bundle_id"]
    for i, r in enumerate(evidence):
        dest = target / f"replay_{i}"
        subprocess.run([exe, "export", r["session"], "--out", dest, "--output", target/f"export_{i}.json"],
                       check=True, timeout=120)
        for name in ("probe_comparison.csv", "probe_predictions.csv"):
            assert (dest / name).read_bytes() == (Path(r["export"])/name).read_bytes()
    with exe.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(exe_sha256=checksum, app_version=doctor["version"], models=doctor["shadow_250_hz"],
                matched_replay_exports=len(evidence), legacy_model_id=doctor["attlapse_model_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=REPO / "var/attlapse/two-channel-benchmark")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exe", type=Path, help="Verify the frozen build against the saved source replay.")
    args = parser.parse_args()
    if args.exe:
        results = json.loads((args.out/"source_replay.json").read_text(encoding="utf8"))
        write(args.out/"frozen_check.json", check_executable(args.exe.resolve(), args.out, results))
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        results = check_source(args.benchmark, args.out)
        write(args.out/"source_replay.json", results)
