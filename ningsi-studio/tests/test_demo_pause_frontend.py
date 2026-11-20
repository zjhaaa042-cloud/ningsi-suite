"""Exercise the participant's pause control and its timers through the actual flow view."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_readiness_frontend import DOM

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node is required for flow tests")
class DemoPauseFrontendTests(unittest.TestCase):
    def test_pause_resume_snapshots_and_participant_view(self):
        program = DOM + r'''
import assert from 'node:assert/strict';
const {render}=await import(FLOW_URI);
ClassList.prototype.add=function(...names){names.forEach(n=>this.toggle(n,true));};
ClassList.prototype.remove=function(...names){names.forEach(n=>this.toggle(n,false));};
StubNode.prototype.contains=function(n){return this===n||this.children.some(c=>c.contains?.(n));};
StubNode.prototype.getBoundingClientRect=()=>({top:0});
StubNode.prototype.querySelector=function(s){if(s==='button:last-child')return this.children.at(-1);return null;};
document.body=new StubNode('body');document.getElementById=()=>null;
globalThis.MutationObserver=class{observe(){}disconnect(){}};
let wall=10000;
Date.now=()=>wall;
Object.defineProperty(globalThis,'performance',{value:{now:()=>wall},configurable:true});
const intervals=new Map();
Object.assign(window,{innerHeight:1000,scrollTo(){},addEventListener(){},removeEventListener(){},
  setInterval(fn,delay){intervals.set(delay,fn);return delay;},clearInterval(){},setTimeout(){return 1;},clearTimeout(){}});
let events;
globalThis.EventSource=class {constructor(){events=this;this.handlers=new Map();}
  addEventListener(k,f){this.handlers.set(k,f);}removeEventListener(){}close(){}};
const state=(paused,paused_seconds,revision)=>({available:true,paused,paused_seconds,revision});
let snap=state(false,0,0), source='sim',scale=.05;
let posted=[],cancelled=0;
api.config=async()=>({data:{}});
api.getSession=async()=>({data:{uuid:'demo',status:'running',phase:'qc',time_scale:scale,
  phases:[{key:'qc',label:'设备质检',duration_sec:600}],runs:[{phase:'qc',status:'running'}],
  runtime:{source_kind:source,demo_pause:snap}}});
api.cancelSession=async()=>{cancelled++;};
api.setDemoPaused=async(id,paused)=>{posted.push([id,paused]);return {data:snap};};
const mount=async()=>{const ctx=makeCtx();ctx.params=new URLSearchParams('session=demo&subject=1');
  const root=new StubNode('main');render(root,ctx);await Promise.resolve();await Promise.resolve();return root;};
const emit=(type,data)=>events.handlers.get(type)({data:JSON.stringify({type,data})});
const controls=root=>all(root).find(n=>n.classList?.contains('flow-demo-controls'));
const timing=root=>text(all(root).find(n=>n.classList?.contains('step__meta-item')&&n.classList.contains('mono')));
const turn=()=>intervals.get(1000)();
let root=await mount();
assert.ok(document.body.classList.contains('subject-mode'));
assert.equal(controls(root).hidden,false);
wall+=2000;turn();assert.match(timing(root),/已用 2s/);
snap=state(true,0,1);
await click(byText(root,'暂停演示'));
assert.deepEqual(posted,[['demo',true]]);assert.equal(cancelled,0);
assert.ok(byText(root,'继续演示'));assert.match(text(controls(root)),/演示已暂停/);
const frozen=timing(root);wall+=5000;turn();assert.equal(timing(root),frozen);
emit('demo_pause',state(false,0,0));assert.ok(byText(root,'继续演示'),'ignore older reconnect replay');
snap=state(false,5,2);await click(byText(root,'继续演示'));
assert.deepEqual(posted,[['demo',true],['demo',false]]);assert.equal(cancelled,0);
wall+=1000;turn();assert.match(timing(root),/已用 3s/);
snap=state(true,5,3);emit('demo_pause',snap);assert.ok(byText(root,'继续演示'));
wall+=6000;snap=state(true,11,3);
root=await mount();assert.ok(byText(root,'继续演示'),'paused session survives reload');
turn();const restored=timing(root);wall+=5000;turn();assert.equal(timing(root),restored);
events.onerror({});snap=state(false,16,4);events.onopen();await Promise.resolve();await Promise.resolve();
assert.ok(byText(root,'暂停演示'),'reconnect uses the server state even without a new event');
emit('finished',{status:'done'});assert.equal(controls(root).hidden,true);
source='lsl';root=await mount();assert.equal(controls(root).hidden,true);
source='sim';scale=.2;root=await mount();assert.equal(controls(root).hidden,true);
'''
        program = program.replace("API_URI", json.dumps((ROOT / "web/js/api.js").as_uri()))
        program = program.replace("FLOW_URI", json.dumps((ROOT / "web/js/views/flow.js").as_uri()))
        process = subprocess.run(["node", "--input-type=module", "-e", program], capture_output=True,
                                 text=True, encoding="utf-8", timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
