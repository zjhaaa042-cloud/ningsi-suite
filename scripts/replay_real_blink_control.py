"""Read-only replay of local device recordings; never invent personal blink labels.

Compare matched Studio filtering/QC with and without the new uncalibrated blink
veto. Capture's historical QC is recorded separately, because it is a different
pipeline. No production model, baseline, export, or original recording is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ningsi/src'))
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, biomulti_packet_clock
from ningsi.signal.blinks import calibrate_blinks, process_blinks
from ningsi.signal.live_artifacts import process_live_artifacts, with_artifact_evidence
from ningsi.signal.preprocess import preprocess
from ningsi.signal.window import analyze_window

LOCAL_SOURCES = ROOT / 'var/blink-replay-sources.local.json'


def recording_sources(recordings, manifest):
    base = Path.cwd()
    if not recordings:
        manifest = manifest or LOCAL_SOURCES
        if not manifest.is_file():
            raise ValueError('Pass --recording NAME DIRECTORY or --sources FILE with your recording paths.')
        mappings = json.loads(manifest.read_text(encoding='utf-8-sig'))
        if not isinstance(mappings, dict) or not mappings:
            raise ValueError('The sources file must map recording names to directory paths.')
        recordings, base = mappings.items(), manifest.resolve().parent
    sources = []
    names = set()
    for name, directory in recordings:
        if not re.fullmatch(r'[\w-]+', name) or PureWindowsPath(name).is_reserved() or name.casefold() in names:
            raise ValueError(f'Use a unique recording name made of letters, digits, "_" or "-": {name}')
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError(f'Missing directory for {name}')
        path = (base / directory).resolve()
        if not (path / 'manifest.json').is_file():
            raise ValueError(f'Missing recording manifest in {path}')
        sources.append((name, path))
        names.add(name.casefold())
    return sources


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_source(key, directory, output):
    """Use NPY exports or a consistent SQLite backup that incorporates committed WAL."""
    paths = [directory / 'manifest.json']
    metadata = json.loads(paths[0].read_text(encoding='utf-8-sig'))
    archived_rows = []
    if (directory / 'raw_samples.npy').exists():
        paths += [directory / 'raw_samples.npy', directory / 'timestamps.npy', directory / 'events.csv']
        before = {str(p): digest(p) for p in paths}
        checksum_file = directory / 'SHA256.json'
        if checksum_file.exists():
            declared = json.loads(checksum_file.read_text(encoding='utf-8-sig'))
            for p in paths:
                if declared.get(p.name) != before[str(p)]:
                    raise ValueError(f'Export checksum mismatch: {p}')
        values = np.load(paths[1], allow_pickle=False)
        stamps = np.load(paths[2], allow_pickle=False)
        events = list(csv.DictReader(paths[3].open(encoding='utf-8-sig', newline='')))
        packet = directory / 'packet_timestamps.npy'
        if packet.exists():
            paths.append(packet)
            before[str(packet)] = digest(packet)
            if checksum_file.exists() and declared.get(packet.name) != before[str(packet)]:
                raise ValueError(f'Packet clock checksum mismatch: {packet}')
            axis = np.load(packet, allow_pickle=False)
        else:
            axis = stamps[:, 1]
        windows = directory / 'windows.csv'
        if windows.exists():
            archived_rows = list(csv.DictReader(windows.open(encoding='utf-8-sig', newline='')))
        snapshot = None
    else:
        database = directory / 'recording.sqlite3'
        paths += [database]
        wal = database.with_name(database.name + '-wal')
        if wal.exists():
            paths.append(wal)
        before = {str(p): digest(p) for p in paths}
        snapshot = output / 'source-snapshots' / key / 'recording.sqlite3'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            raise FileExistsError(f'Choose a fresh output directory: {snapshot}')
        source = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        target = sqlite3.connect(snapshot)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        connection = sqlite3.connect(snapshot.as_uri() + '?mode=ro', uri=True)
        try:
            metadata = json.loads(connection.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0])
            rows = connection.execute('SELECT sample_index,values_f64,source_timestamp_bytes,corrected_timestamp,'
                                      'received_lsl,received_perf FROM samples ORDER BY sample_index').fetchall()
            if [r[0] for r in rows] != list(range(len(rows))):
                raise ValueError('Noncontiguous stored sample indices')
            values = np.stack([np.frombuffer(r[1], dtype='<f8') for r in rows])
            stamps = np.array([[np.frombuffer(r[2], dtype='<f8')[0],
                                r[3] if r[3] is not None else np.nan, r[4], r[5]] for r in rows])
            events = [{'kind': r[0], 'payload': r[1]} for r in connection.execute('SELECT kind,payload FROM events')]
        finally:
            connection.close()
        source = metadata['source']
        axis = stamps[:, 1].copy()
        axis[np.abs(stamps[:, 2]-axis) >= 3] = np.nan
        if source.get('timestamp_policy') == BIOMULTI_LSL_TIMING:
            axis = biomulti_packet_clock(axis)
    if values.shape != (len(axis), 2):
        raise ValueError(f'Expected synchronized two-channel raw data: {directory}')
    return metadata, values, stamps, axis, events, archived_rows, paths, before, snapshot


def timing_reasons(axis, rate, expected):
    if not np.isfinite(axis).all():
        return ['clock_unsynchronized']
    delta = np.diff(axis)
    reasons = []
    if np.any(delta <= 0):
        reasons.append('nonmonotonic_timestamps')
    if np.any(delta > .1):
        reasons.append('timestamp_gap_over_100ms')
    elif np.any(delta > 1.75/rate):
        reasons.append('sample_gap_in_filter_history')
    if abs((axis[-1]-axis[-expected])*rate/(expected-1)-1) > .05:
        reasons.append('sample_rate_mismatch_over_5pct')
    return reasons


def line_fraction(values, rate):
    centered = values-values.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered*np.hanning(values.shape[1]), axis=1))**2
    freq = np.fft.rfftfreq(values.shape[1], 1/rate)
    return power[:, (freq >= 49)&(freq <= 51)].sum(axis=1)/np.maximum(
        power[:, (freq >= 1)&(freq <= min(100, rate/2))].sum(axis=1), 1e-30)


def calibration_feasibility(values, rate, labels, events):
    """A deliberately separate exploratory check; its automatic candidates are NOT labels."""
    marks = [r for r in events if 'blink' in r['kind'].lower() or '眨眼' in r['kind']]
    if marks:
        raise ValueError('Explicit blink events found; inspect their sample mapping before replay')
    prefix = values[:round(120*rate)].T
    width, hop, history = round(4*rate), round(2*rate), round(6*rate)
    peaks = []
    for end in range(width, prefix.shape[1]+1, hop):
        first = max(0, end-history)
        result = process_blinks(prefix[:, first:end], rate, channel_labels=labels)
        for event in result.events:
            p = first+event['peak_index']
            if .6*rate <= p < prefix.shape[1]-.6*rate:
                peaks.append(p)
    selected = []
    for peak in sorted(peaks):
        if not selected or peak-selected[-1] >= 1.5*rate:
            selected.append(peak)
    selected = selected[:20]
    try:
        profile = calibrate_blinks(prefix, rate, selected, channel_labels=labels,
                                   montage_id='exploratory-candidates-NOT-personal-calibration')
        failure = ''
        status = 'exploratory_fit_passed_but_not_applied_unverified_events'
        accepted = profile.events
    except ValueError as exc:
        failure, status, accepted = str(exc), 'exploratory_fit_rejected', 0
    return {'status': status, 'formal_calibration_status': 'not_available_no_marked_blinks',
            'marker_count': len(marks), 'prefix_seconds': prefix.shape[1]/rate,
            'candidate_origin': 'automatic unverified events from prefix only; never applied to task data',
            'candidate_count': len(selected), 'accepted_candidates': accepted, 'failure': failure,
            'correction_applied': False}


def replay(key, directory, output, rows_out, examples):
    loaded = load_source(key, directory, output)
    metadata, values, stamps, axis, events, archived, paths, hashes, snapshot = loaded
    source = metadata['source']
    rate = float(source['srate'])
    factor = source.get('scale_to_uv')
    if factor is None or not np.isfinite(factor) or factor <= 0 or source.get('data_origin') != 'real':
        raise ValueError('Real source declaration and explicit recorded unit factor required')
    labels = tuple(source.get('channel_labels') or ['Ch1','Ch2'])
    if len(labels) != 2 or len(set(labels)) != 2:
        labels = ('Ch1', 'Ch2')
    values_uv = values * factor
    calibration = calibration_feasibility(values_uv, rate, labels, events)
    counters = {name: Counter() for name in ('old', 'new', 'raw_core', 'clock')}
    width, hop, context = round(4*rate), round(2*rate), round(6*rate)
    totals = Counter()
    powers = []
    for end in range(width, len(values)+1, hop):
        first = max(0, end-context)
        raw, uv = values[first:end].T, values_uv[first:end].T
        blink = process_blinks(uv, rate, channel_labels=labels)
        # With no personal template, the deployed algorithm MUST leave samples unchanged.
        if not np.array_equal(blink.cleaned_uv, uv, equal_nan=True):
            raise AssertionError('Uncalibrated replay modified original signal')
        centered = uv-uv.mean(axis=1, keepdims=True)
        filtered = np.stack([preprocess(channel, rate)[0] for channel in centered])
        old_wave, old_info = process_live_artifacts(filtered, raw, rate, width, labels,
            source.get('declared_units', ()), factor, factor)
        blink_info = blink.as_dict(start_index=max(0, uv.shape[1]-width))
        new_wave, new_info = process_live_artifacts(filtered, raw, rate, width, labels,
            source.get('declared_units', ()), factor, factor, blink_evidence=blink_info)
        if not np.array_equal(old_wave, new_wave, equal_nan=True):
            raise AssertionError('Waveform changed without a personal template')
        base_quality = analyze_window(old_wave, rate).quality
        old_q = with_artifact_evidence(base_quality, old_info)
        new_q = with_artifact_evidence(base_quality, new_info)
        clock = timing_reasons(axis[first:end], rate, width)
        corrected = bool(blink_info['corrected_events'])
        totals.update(windows=1, before=old_q.ok, after=new_q.ok,
                      before_timed=old_q.ok and not clock, after_timed=new_q.ok and not clock,
                      extra_rejected=old_q.ok and not new_q.ok, corrected=corrected)
        for name, reasons in (('old', old_q.reasons), ('new', new_q.reasons),
                              ('raw_core', blink_info['reasons']), ('clock', clock)):
            counters[name].update(reasons)
        powers.append(line_fraction(uv[:, -width:], rate))
        rows_out.append({'record': key, 'end_sample_exclusive': end, 'nominal_end_sec': end/rate,
            'context_start_sample': first, 'before_usable': old_q.ok, 'after_usable': new_q.ok,
            'before_timed_usable': old_q.ok and not clock, 'after_timed_usable': new_q.ok and not clock,
            'before_reasons': '|'.join(old_q.reasons), 'after_reasons': '|'.join(new_q.reasons),
            'blink_reasons': '|'.join(blink_info['reasons']), 'clock_reasons': '|'.join(clock),
            'corrected_events': blink_info['corrected_events'], 'masked_fraction': blink_info['masked_fraction']})
        # Fixed early complete context, chosen independently of apparent effectiveness.
        if end == context:
            examples[key+'_raw'] = uv
            examples[key+'_filtered'] = filtered
            examples[key+'_time'] = np.arange(first, end)/rate
    unchanged = {str(p): digest(p) for p in paths} == hashes
    if not unchanged:
        raise ValueError(f'Original input changed during replay: {directory}')
    baseline = directory/'baseline.json'
    existing_baseline = json.loads(baseline.read_text(encoding='utf-8-sig')) if baseline.exists() else {}
    record = {'key': key, 'source_path': str(directory), 'participant': metadata.get('participant'),
        'visit': metadata.get('visit'), 'samples': len(values), 'nominal_seconds': len(values)/rate,
        'state': metadata.get('state'), 'created_utc': metadata.get('created',{}).get('utc'),
        'source': {k: source.get(k) for k in ('name','data_origin','channel_labels','reference',
                   'source_unit','scale_to_uv','unit_basis','srate','timestamp_policy','preprocessing_policy')},
        'windows_total': totals['windows'], 'before_usable': totals['before'], 'after_usable': totals['after'],
        'before_timed_usable': totals['before_timed'], 'after_timed_usable': totals['after_timed'],
        'extra_rejected': totals['extra_rejected'], 'corrected_windows': totals['corrected'],
        'raw_core_reasons': dict(counters['raw_core']), 'old_reasons': dict(counters['old']),
        'new_reasons': dict(counters['new']), 'clock_reasons': dict(counters['clock']),
        'line_50hz_median_by_channel': np.median(powers,axis=0).tolist(),
        'calibration': calibration, 'raw_hash': hashes.get(str(directory/'raw_samples.npy')),
        'inputs_sha256': hashes, 'inputs_unchanged': unchanged,
        'snapshot_path': str(snapshot) if snapshot else None,
        'snapshot_sha256': digest(snapshot) if snapshot else None,
        'raw_values_sha256': hashlib.sha256(values.astype('<f8').tobytes()).hexdigest(),
        'historical_capture_windows': len(archived),
        'historical_capture_usable': sum(row['usable']=='True' for row in archived),
        'existing_capture_baseline_valid': existing_baseline.get('valid'),
        'scores_produced': False, 'score_reason': 'No newly validated baseline; this replay measures signal QC only',
        'sample_values_changed': False}
    print(json.dumps({k:record[k] for k in ('key','windows_total','before_usable','after_usable',
          'before_timed_usable','after_timed_usable','corrected_windows')}, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--recording', nargs=2, metavar=('NAME', 'DIRECTORY'), action='append',
                           help='Recording label followed by its directory. Repeat for additional recordings.')
    selection.add_argument('--sources', type=Path,
                           help='JSON object mapping recording names to directories relative to the sources file.')
    parser.add_argument('--output', type=Path, default=ROOT/'var/blink-real-replay-20261118')
    args = parser.parse_args()
    try:
        sources = recording_sources(args.recording, args.sources)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.json').exists():
        raise FileExistsError('Choose a fresh output; existing evidence is not overwritten')
    rows, examples, records = [], {}, []
    for key, directory in sources:
        records.append(replay(key, directory, output, rows, examples))
    with (output/'windows.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output/'diagnostic_examples.npz', **examples)
    metric = {'schema': 'ningsi-real-blink-replay-v1', 'records': records,
        'method': 'Matched Studio DC removal/preprocess-v2/live_artifacts/QC, 4s window/2s hop/up to6s context; new path adds raw uncalibrated blink evidence; no correction without marked personal calibration',
        'clock_method': 'Recorded packet clock, or existing BioMulti estimator on consistent SQLite snapshot; timing diagnostic shown separately; no live freshness/hardware-delay claim',
        'implementation_sha256': {name:digest(ROOT/name) for name in (
            'scripts/replay_real_blink_control.py','ningsi/src/ningsi/signal/blinks.py',
            'ningsi/src/ningsi/signal/live_artifacts.py','ningsi/src/ningsi/signal/preprocess.py',
            'ningsi/src/ningsi/signal/artifacts.py','ningsi-studio/src/ningsi_studio/core/live_source.py')},
        'limitations': ['No explicit blink events found; this replay does not use a stored personal calibration profile.',
            'No EOG/video or clean EEG truth: precision/recall, recovery RMSE and attention accuracy cannot be measured.',
            'Automatic prefix candidates are exploratory only, never used as validated labels or applied to task data.',
            'Software-inferred units, unknown electrode mapping/reference and failed/aborted sessions remain explicit.',
            'Overlapping window counts are not additive seconds; this replay produces no new baseline, attention score or training rows.',
            'blink_signal_fault denotes >1000uV deviation from window median, not proven amplifier saturation.',
            'Capture historical counts and matched Studio counts use distinct pipelines and are not before/after improvements.'],
        'originals_unchanged': all(r['inputs_unchanged'] for r in records),
        'personal_correction_applied': False, 'training_rows_created': 0}
    write_json(output/'metrics.json', metric)
    print(str(output/'metrics.json'))


if __name__ == '__main__':
    main()
