"""Render the training view with real SSE handlers and controlled API replies."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
import assert from 'node:assert/strict';
// Match the browser: replacing textContent also removes existing children.
Object.defineProperty(StubNode.prototype, 'textContent', {
  get() { return this._text || ''; },
  set(value) { this._text = String(value); this.children = []; },
});
ClassList.prototype.add = function (...names) { names.forEach(n => this.toggle(n, true)); };
StubNode.prototype.append = function (...nodes) { this.children.push(...nodes); };
document.createElementNS = (ns, tag) => new StubNode(tag);
document.querySelector = () => null;
let poll, events;
const cleanups = [];
globalThis.window = {
  location: {search: ''}, setTimeout, clearTimeout,
  setInterval: callback => { poll = callback; return 1; }, clearInterval() {},
};
globalThis.EventSource = class {
  constructor() { events = this; this.handlers = new Map(); }
  addEventListener(type, callback) { this.handlers.set(type, callback); }
  removeEventListener() {} close() {}
};
const {api} = await import(API_URI);
const saved = (seq, mean, n=2) => ({seq, index:seq+1, target:.5, hold_sec:6,
  stats:{mean, n, achieved:n>0, on_target_ratio:n>0?1:0, current_hold_sec:6, max_hold_sec:6}});
let response = {segments:[saved(0,.8)], summary:{}};
let session = {status:'running', phase:'training', runs:[]};
api.training = async () => ({data:response});
api.getSession = async () => ({data:session});
const {render} = await import(VIEW_URI);
const root = new StubNode('main');
const mount = () => render(root, {
  params:new URLSearchParams('session=test'), signal:{aborted:false},
  store:{state:{},setState(){}}, onCleanup:f=>cleanups.push(f), navigate(){},
});
const settle = async () => { for (let i=0;i<8;i++) await Promise.resolve(); };
const emit = (type,data) => events.handlers.get(type)({data:JSON.stringify({type,data})});
const all = node => [node,...(node.children||[]).flatMap(all)];
const card = title => all(root).find(n => n.classList?.contains('card') &&
  all(n).some(child => child.tagName==='H3' && child.textContent===title));
const field = label => {
  const status = card('达标状态');
  assert.ok(status, 'training status card must render');
  const caption = all(status).find(n=>n.tagName==='P' && n.textContent===label);
  assert.ok(caption, `missing status field: ${label}`);
  const row = all(status).find(n=>n.children?.includes(caption));
  return row.children[1].textContent;
};
'''


@unittest.skipUnless(shutil.which('node'), 'Node is required for training view tests')
class TrainingFrontendTests(unittest.TestCase):
    def run_view(self, scenario):
        program = (NODE_STUB + HARNESS + scenario + '\ncleanups.forEach(f=>f());')
        program = program.replace('API_URI', json.dumps((ROOT / 'web/js/api.js').as_uri()))
        program = program.replace('VIEW_URI', json.dumps((ROOT / 'web/js/views/training.js').as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', program],
                                capture_output=True, text=True, encoding='utf-8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_live_window_and_poll_never_reuse_saved_score(self):
        self.run_view(r'''
mount(); await settle();
emit('feedback', {segment:2,score:.65,target:.5,usable:true,achieved:false,current_hold_sec:2});
assert.equal(field('当前专注度'),'0.650');
emit('feedback', {segment:2,score:null,target:.5,usable:false,achieved:true,current_hold_sec:6});
assert.equal(field('当前专注度'),'—');
assert.equal(field('达标状态'),'—');
assert.equal(field('本段达标时间占比'),'—');
assert.equal(field('当前段'),'2');
assert.equal(field('当前连续保持 / 本段最长'),'0.0s / —');
poll(); await settle();
assert.equal(field('当前专注度'),'—','empty summary must not enable saved-result fallback');
assert.equal(field('达标状态'),'—');
emit('feedback', {segment:2,score:.7,target:.5,usable:true,achieved:false,current_hold_sec:2});
assert.equal(field('当前专注度'),'0.700','feedback must recover after a missing window');
poll(); await settle();
assert.equal(field('当前专注度'),'0.700','polling cannot erase a valid live score');
emit('feedback', {segment:2,score:0,target:.5,usable:true,achieved:false});
assert.equal(field('当前专注度'),'0.000','a genuine zero is a valid score');
assert.equal(field('达标状态'),'未达标');
emit('feedback', {segment:2,score:null,target:.5,usable:true,achieved:true});
assert.equal(field('当前专注度'),'—','a valid raw window without a score remains missing');
assert.equal(field('达标状态'),'—');
emit('feedback', {segment:2,score:.9,target:null,usable:false,achieved:true});
assert.equal(field('当前专注度'),'—','unusable flag overrides even a numeric score');
assert.equal(field('达标状态'),'—');
emit('segment', saved(1,.6));
emit('phase', {key:'training',state:'done'});
assert.equal(field('末段平均专注度'),'0.600');
assert.equal(field('达标状态'),'达标','completed training retains its saved result');
''')

    def test_completed_session_loads_saved_result_without_live_feedback(self):
        self.run_view(r'''
session = {status:'running',phase:'baseline_after',runs:[{phase:'training',status:'done'}]};
mount(); await settle();
assert.equal(field('末段平均专注度'),'0.800');
assert.equal(field('达标状态'),'达标');
''')

    def test_terminal_session_preserves_partial_saved_training(self):
        self.run_view(r'''
session = {status:'cancelled',phase:'cancelled',runs:[]};
mount(); await settle();
assert.equal(field('末段平均专注度'),'0.800');
assert.equal(field('达标状态'),'达标');
''')

    def test_empty_last_segment_cannot_fall_back_to_session_average(self):
        self.run_view(r'''
response = {segments:[saved(0,.8),saved(1,0,0)],
  summary:{mean_focus:.8,on_target_ratio:1,segments:[saved(0,.8),saved(1,0,0)]}};
session = {status:'done',phase:'done',runs:[]};
mount(); await settle();
assert.equal(field('末段平均专注度'),'—','empty historic segment is missing, even if its old mean is zero');
assert.equal(field('达标状态'),'—');
assert.equal(field('本段达标时间占比'),'—');
''')

    def test_saved_time_coverage_and_valid_time_success_are_separate(self):
        self.run_view(r'''
const segment = saved(0,.8);
Object.assign(segment.stats, {timing_spec:'neurofeedback-v3',valid_time_ratio:.08,
  on_target_valid_ratio:1,on_target_ratio:.08,achieved:false});
response = {segments:[segment],summary:{spec:'neurofeedback-v3'}};
session = {status:'done',phase:'done',runs:[]};
mount(); await settle();
assert.equal(field('本段达标时间占比'),'8%（整段时间）');
assert.equal(field('有效时间覆盖率'),'8%');
assert.equal(field('有效时间内达标率'),'100%');
assert.equal(field('达标状态'),'未达标');
''')

    def test_baseline_features_are_not_labeled_as_normalized_scores(self):
        self.run_view(r'''
response = {segments:[saved(0,.8)],summary:{spec:'neurofeedback-v3',baseline_before:{
  valid:true,n_windows:15,device:'test',srate:250,indices:{focus:{mean:-.2,n:15}}}}};
session = {status:'done',phase:'done',runs:[]};
mount(); await settle();
const text=all(card('训练前后基线对比')).map(n=>n.textContent).join(' ');
assert.ok(text.includes('基线对数特征'));
assert.ok(text.includes('不是0–1评分'));
''')
