"""Exercise the deployed IMU and EMG gates on public sensor recordings with declared units."""

from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ningsi" / "src"))
from ningsi.signal.live_artifacts import check_motion_reference, process_live_artifacts


def review_imu(path):
    with ZipFile(path) as outer, ZipFile(BytesIO(outer.read("UCI HAR Dataset.zip"))) as z:
        base = "UCI HAR Dataset/test/"
        labels = np.loadtxt(z.open(base+"y_test.txt"), dtype=int)
        people = np.loadtxt(z.open(base+"subject_test.txt"), dtype=int)
        values = np.stack([np.loadtxt(z.open(base+f"Inertial Signals/total_acc_{a}_test.txt")) for a in "xyz"], axis=1)
    results = {s: Counter() for s in ("walking_or_stairs", "standing_sitting_lying")}
    n = 0
    included = set()
    for i in range(0, len(labels)-2, 4):
        if people[i] != people[i+2] or not np.all(labels[i:i+3] == labels[i]):
            continue
        if not all(np.allclose(values[j,:,64:], values[j+1,:,:64], rtol=1e-5, atol=1e-6) for j in (i, i+1)):
            continue
        record = np.concatenate((values[i], values[i+1,:,64:], values[i+2,:,64:]), axis=1)[:, :200]
        n += 1
        included.add(int(people[i]))
        snap = {"age": 0., "timestamps": np.arange(200)/50, "samples": record,
                "descriptor": {"declared_units": ["g"]*3}}
        evidence = check_motion_reference(snap, np.arange(1000)/250, [0,1,2], "g")
        group = results["walking_or_stairs" if labels[i] <= 3 else "standing_sitting_lying"]
        group.update(total=1, motion=int(evidence["state"] == "movement"))
    return {"source": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
            "people": len(included), "windows": n, "groups": results,
            "scope": "Waist accelerometer, published test set only. Nonoverlapping 4 s windows reconstructed only across identical source overlaps, labels and subjects. Validates sensor activity detection, not EEG contamination or forehead transfer."}


def review_emg(path):
    # WFDB header: little-endian 16-bit samples at 4 kHz, 10000 ADC units/mV.
    raw = np.fromfile(path, dtype="<i2")
    if len(raw) != 50860 or int(raw.astype(int).sum()) % 65536 != (-29438) % 65536:
        raise ValueError("PhysioNet header length or checksum mismatch")
    taps = np.arange(-128, 129)
    kernel = np.sinc(2*100/4000*taps) * np.hamming(len(taps))
    muscle = np.convolve(raw*.1, kernel/kernel.sum(), "same")[::16]
    windows = []
    for start in range(0, len(muscle)-999, 1000):
        segment = muscle[start:start+1000]
        alpha_control = 12*np.sin(2*np.pi*10*np.arange(1000)/250)
        data = np.stack((alpha_control, segment))
        _, result = process_live_artifacts(data, data, 250, 1000, ["Fp1","EMG"], ["uV","uV"])
        windows.append({"start_seconds": start / 250, "emg_peak_to_peak_uv": float(np.ptp(segment)),
                        "reference_gate_triggered": "emg_reference_burst" in result["reasons"]})
    return {"source": "https://physionet.org/content/emgdb/1.0.0/", "record": "emg_healthy",
            "windows": windows,
            "scope": "One real independently recorded EMG source paired with synthetic 10 Hz control EEG. Shows response of the EMG-reference gate; no simultaneous cranial EEG or contamination labels."}


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    out = {"imu": review_imu(folder/"uci-har.zip"), "emg": review_emg(folder/"emg_healthy.dat")}
    (folder/"reference_sensors.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf8")
