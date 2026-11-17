"""Exercise the packaged showcase over loopback without a device or private data."""

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18770')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    address = urlparse(args.url)
    if address.scheme != 'http' or address.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Only local HTTP verification is supported')
    if args.output.exists():
        raise FileExistsError('Verification record already exists')
    checks = []

    def request(path, body=None, binary=False):
        req = Request(args.url.rstrip('/') + path,
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=30) as response:
            data = response.read()
        return data if binary else json.loads(data)

    def passed(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    meta = request('/api/showcase/metadata')
    evidence = request('/api/showcase/evidence')
    passed('frozen evidence explicitly labelled as a development snapshot',
           evidence['evidence_origin'] == 'bundled_development_snapshot' and bool(evidence['snapshot_at']))
    passed('all requirement groups available', len(evidence['requirements']) == 8)
    passed('fixed theta alpha beta definitions', set(meta['bands_hz']) == {'theta', 'alpha', 'beta'})
    clean = request('/api/showcase/analyze', {'mode': 'demo', 'artifact': 'clean'})
    passed('clean simulation has independent baseline and valid scores',
           clean['source']['simulated'] and clean['baseline']['valid'] and bool(clean['latest']['scores']))
    mixed = request('/api/showcase/analyze', {'mode': 'demo', 'artifact': 'mixed'})
    passed('artifact windows are rejected without fake scores',
           not mixed['latest']['usable'] and not mixed['latest']['scores']
           and any(v is None for v in mixed['traces']['channels'][0]['cleaned_uv']))
    row = [12 * math.sin(2 * math.pi * 10 * i / 250) + 6 * math.sin(2 * math.pi * 20 * i / 250)
           for i in range(250 * 8)]
    raw = request('/api/showcase/analyze', {'mode': 'raw', 'samples': [row], 'rate': 250, 'scale_to_uv': 1})
    passed('imported data is unverified and has no score without independent baseline',
           raw['source']['kind'] == 'imported' and not raw['source']['real_hardware_verified']
           and not raw['baseline']['valid'] and not raw['latest']['scores'])
    model = request('/api/models/train', {'subjects': 6, 'windows_per_state': 4})
    passed('real classifier training returns held-out splits', all(model.get(key) for key in ('train', 'validation', 'test')))
    created = request('/api/showcase/demo-session', {})
    session = created['session']
    uuid = session['uuid']
    deadline = time.monotonic() + 90
    while session['status'] not in ('done', 'failed', 'cancelled') and time.monotonic() < deadline:
        time.sleep(.5)
        session = request(f'/api/sessions/{uuid}')
    passed('packaged complete simulation finishes', session['status'] == 'done' and session['device'] == 'sim-bsense')
    report = request(f'/api/sessions/{uuid}/report')
    passed('report contains EEG scales behavior assessment and advice',
           all(report.get(key) for key in ('indicators', 'scales', 'behavior', 'assessment'))
           and bool(report['assessment'].get('advice')))
    html = request(f'/api/sessions/{uuid}/artifacts/report_html', binary=True)
    passed('standalone HTML report downloads', b'<html' in html.lower())
    archive = request(f'/api/sessions/{uuid}/export.zip', binary=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        passed('session ZIP has readable report and valid CRC', zipped.testzip() is None
               and any(name.endswith('.html') for name in zipped.namelist()))
    heatmap = request(f'/api/sessions/{uuid}/heatmap')
    passed('saved heatmap available', bool(heatmap.get('cells')))
    for period in ('week', 'month'):
        params = urlencode({'participant': session['participant'], 'period': period, 'field': 'focus'})
        passed(f'{period} focus aggregation', request('/api/reports/trend?' + params)['sessions'] >= 1)
    stress = request('/api/reports/trend?' + urlencode({'participant': session['participant'], 'period': 'month', 'field': 'stress'}))
    passed('simulation is excluded from field pressure self-report trends', stress['sessions'] == 0)
    for asset in ('views/showcase.js', 'views/showcase-analysis.js', 'views/showcase-session.js'):
        passed(asset + ' served', len(request('/js/' + asset, binary=True)) > 100)
    passed('showcase stylesheet served', len(request('/css/showcase.css', binary=True)) > 100)
    result = {'verified_at': datetime.now(timezone.utc).isoformat(), 'url': args.url,
              'passed': True, 'checks': checks, 'check_count': len(checks),
              'source': 'packaged Windows EXE; explicit synthetic fixtures',
              'hardware_or_human_validity_tested': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
