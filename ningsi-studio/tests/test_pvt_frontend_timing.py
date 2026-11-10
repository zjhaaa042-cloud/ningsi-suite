"""Execute the real flow view with controlled SSE events, keys and timers."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node is required for frontend timing tests')
class PvtFrontendTimingTests(unittest.TestCase):
    def test_false_start_settlement_cancels_only_matching_trial(self):
        program = NODE_STUB + r'''
import assert from 'node:assert/strict';
ClassList.prototype.add = function (...names) { names.forEach(n => this.toggle(n, true)); };
ClassList.prototype.remove = function (...names) { names.forEach(n => this.toggle(n, false)); };
StubNode.prototype.append = function (...nodes) { this.children.push(...nodes); };
StubNode.prototype.getBoundingClientRect = () => ({top: 0});
document.body = new StubNode('body');
document.querySelector = () => null;
document.getElementById = () => null;
globalThis.MutationObserver = class { observe() {} disconnect() {} };
let clock = 10000, serial = 0;
Date.now = () => clock;
globalThis.performance = {now: () => clock};
const timers = new Map(), keys = [], cleanups = [], posted = [];
globalThis.window = {
  innerHeight: 1000, location: {search: ''}, scrollTo() {},
  addEventListener: (type, callback) => { if (type === 'keydown') keys.push(callback); },
  removeEventListener() {},
  setTimeout: callback => { timers.set(++serial, callback); return serial; },
  clearTimeout: id => timers.delete(id),
  setInterval: () => ++serial, clearInterval() {},
};
let events;
globalThis.EventSource = class {
  constructor() { events = this; this.handlers = new Map(); }
  addEventListener(type, callback) { this.handlers.set(type, callback); }
  removeEventListener() {} close() {}
};
const {api} = await import(API_URI);
api.config = async () => ({data: {phases: []}});
api.getSession = async () => ({data: {uuid: 'test', status: 'running', phase: 'pvt',
  time_scale: 1, runs: [], phases: []}});
api.submitPvtTrial = async (uuid, body) => { posted.push(body); return {data: {accepted: true}}; };
const {render} = await import(FLOW_URI);
const store = {state: {}, setState(patch) { Object.assign(this.state, patch); }};
render(new StubNode('main'), {params: new URLSearchParams('session=test'), store,
  signal: {aborted: false}, onCleanup: f => cleanups.push(f), navigate() {}});
await Promise.resolve(); await Promise.resolve();
const emit = (type, data) => events.handlers.get(type)({data: JSON.stringify({type, data})});
const press = (repeat=false) => keys.forEach(f => f({code:'Space', key:' ', repeat,
  target: {tagName: 'BODY'}, preventDefault() {}}));
const expire = () => { clock += 4000; const tasks = [...timers.values()]; timers.clear(); tasks.forEach(f => f()); };
emit('behavior_request', {task: 'pvt', trials: 30});
press(true);
assert.equal(posted.length, 0, 'held key must not create false starts');
press();
assert.equal(posted.length, 1);
assert.equal(posted[0].index, 0);
assert.equal(posted[0].false_start, true);
clock += 1500; press();
assert.equal(posted.length, 1, 'same waiting period submits only one false start');
emit('trial', {task:'pvt', index:0, phase:'main', total:30, response_window:1});
emit('notice', {task:'pvt', index:0, false_start:true, level:'info', message:'settled'});
expire(); await Promise.resolve();
assert.equal(posted.length, 1, 'settled false start must not also submit a timeout');
emit('trial', {task:'pvt', index:1, phase:'main', total:30, response_window:1});
emit('notice', {task:'pvt', index:0, false_start:true, level:'info', message:'late notice'});
expire(); await Promise.resolve();
assert.equal(posted.length, 2, 'old notice must not cancel the next trial');
assert.equal(posted[1].index, 1);
assert.equal(posted[1].responded, false);
cleanups.forEach(f => f());
console.log('PVT frontend timing OK');
'''
        program = program.replace('API_URI', json.dumps((ROOT / 'web/js/api.js').as_uri()))
        program = program.replace('FLOW_URI', json.dumps((ROOT / 'web/js/views/flow.js').as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', program],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
