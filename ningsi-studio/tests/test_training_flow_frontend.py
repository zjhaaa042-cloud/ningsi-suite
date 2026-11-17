"""Exercise training metrics, timer snapshots and replay in the actual flow view."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node is required for flow DOM tests')
class TrainingFlowFrontendTests(unittest.TestCase):
    def test_metrics_pause_snapshot_and_segment_replay(self):
        program = NODE_STUB + r'''
import assert from 'node:assert/strict';
Object.defineProperty(StubNode.prototype,'textContent',{
  get(){return this._text||'';},set(value){this._text=String(value);this.children=[];}
});
ClassList.prototype.add=function(...names){names.forEach(n=>this.toggle(n,true));};
ClassList.prototype.remove=function(...names){names.forEach(n=>this.toggle(n,false));};
StubNode.prototype.append=function(...nodes){this.children.push(...nodes);};
StubNode.prototype.contains=function(child){return this===child||this.children.some(n=>n===child||n.contains?.(child));};
StubNode.prototype.getBoundingClientRect=()=>({top:0});
document.body=new StubNode('body');document.querySelector=()=>null;document.getElementById=()=>null;
globalThis.MutationObserver=class{observe(){}disconnect(){}};
globalThis.window={innerHeight:1000,location:{search:''},scrollTo(){},addEventListener(){},removeEventListener(){},
  setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){}};
let events;const cleanups=[];
globalThis.EventSource=class{constructor(){events=this;this.handlers=new Map();}
  addEventListener(type,callback){this.handlers.set(type,callback);}removeEventListener(){}close(){}};
const {api}=await import(API_URI);
let session={uuid:'test',status:'running',phase:'training',protocol:'effective_time',runs:[],phases:[]};
api.getSession=async()=>({data:session});api.config=async()=>({data:{phases:[]}});
const {render}=await import(FLOW_URI);let root;
const mount=()=>{root=new StubNode('main');const store={state:{},setState(patch){Object.assign(this.state,patch);}};
  render(root,{params:new URLSearchParams('session=test'),store,signal:{aborted:false},onCleanup:f=>cleanups.push(f),navigate(){}});};
const settle=async()=>{for(let i=0;i<10;i++)await Promise.resolve();};
const emit=(type,data)=>events.handlers.get(type)({data:JSON.stringify({type,data})});
const all=node=>[node,...(node.children||[]).flatMap(all)];
const text=node=>typeof node==='string'?node:(node.textContent||'')+(node.children||[]).map(text).join(' ');
const card=title=>all(root).find(n=>n.classList?.contains('card')&&all(n).some(c=>c.tagName==='H3'&&c.textContent===title));
const progress={spec:'effective-time-v1',target_seconds:50,valid_seconds:2,elapsed_seconds:10,
  max_seconds:100,extension_seconds:0,quality_paused:true,completed:false,completion_reason:null};
const feedback={segment:1,score:null,usable:false,target:.5,hold_sec:6,current_hold_sec:6,
  feedback_mode:'quality',mode_label:'质量反馈',effective_time:progress,
  segment_stats:{duration_sec:10,n:1,valid_seconds:2,on_target_ratio:.2,
    on_target_valid_ratio:1,valid_time_ratio:.2,coverage_sufficient:false}};
mount();await settle();emit('feedback',feedback);
let current=card('当前训练反馈');assert.ok(current);
assert.ok(text(current).includes('有效时间内达标率 100%'));
assert.ok(text(current).includes('有效覆盖 20%'));
assert.ok(text(current).includes('有效反馈 2.0s / 实际 10.0s'));
assert.ok(text(current).includes('样本不足'));
assert.ok(text(current).includes('信号待恢复'));
assert.equal(Number(all(current).find(n=>n.tagName==='PROGRESS').value),0,'bad window breaks continuous hold');
const timerBefore=text(card('有效时长训练进度'));assert.ok(timerBefore.includes('质量暂停'));
session={...session,runtime:{latest_feedback:feedback,effective_time:{key:'training',phase:'training',segment:1,effective_time:progress}}};
mount();await settle();current=card('当前训练反馈');
assert.ok(text(current).includes('有效时间内达标率 100%'));
assert.ok(text(current).includes('有效覆盖 20%'));
assert.equal(text(card('有效时长训练进度')),timerBefore,'reload uses the same server-owned timer');
const segment={seq:0,duration_sec:10,target:.5,target_after:.5,stats:feedback.segment_stats};
emit('segment',segment);emit('segment',segment);
const primaryTable=all(card('训练分段')).find(n=>n.tagName==='TABLE');
assert.equal(all(primaryTable).filter(n=>n.tagName==='TR').length,2,'SSE replay cannot duplicate a saved segment');
assert.ok(text(card('训练分段')).includes('100%'));assert.ok(text(card('训练分段')).includes('20%'));
emit('effective_progress',{key:'training',effective_time:{...progress,completion_reason:'max_duration',elapsed_seconds:100}});
assert.ok(text(card('有效时长训练进度')).includes('数据不足'));
cleanups.forEach(f=>f());
'''
        program = program.replace('API_URI', json.dumps((ROOT / 'web/js/api.js').as_uri()))
        program = program.replace('FLOW_URI', json.dumps((ROOT / 'web/js/views/flow.js').as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', program],
                                capture_output=True, text=True, encoding='utf-8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
