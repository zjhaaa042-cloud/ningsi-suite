"""Heatmap coverage, filtering and live replay retain the actual evidence denominator."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]
DOM = NODE_STUB + r'''
StubNode.prototype.append=function(...children){this.children.push(...children);};
StubNode.prototype.replaceChildren=function(...children){this.children=[];this.append(...children);};
StubNode.prototype.getBoundingClientRect=()=>({width:720,height:200});
Object.defineProperty(StubNode.prototype,'textContent',{
  get(){return this._text||'';},set(value){this._text=String(value);this.children=[];}
});
document.createElementNS=(_,tag)=>new StubNode(tag);
document.querySelector=()=>null;
globalThis.window={location:{search:'',origin:'http://localhost'},setInterval:()=>1,clearInterval(){},setTimeout,clearTimeout};
const all=node=>!node||typeof node==='string'?[]:[node,...(node.children||[]).flatMap(all)];
const text=node=>typeof node==='string'?node:(node?.textContent||'')+(node?.children||[]).map(text).join('');
const byClass=(node,name)=>all(node).find(item=>item.classList?.contains(name));
const byButton=(node,label)=>all(node).find(item=>item.tagName==='BUTTON'&&text(item)===label);
const action=async(node,event='click')=>{for(const fn of node.listeners[event]||[])await fn({target:node});};
const flush=()=>new Promise(resolve=>setTimeout(resolve,5));
const cellsIn=node=>all(node).filter(item=>item.classList?.contains('heatmap__cell'));
const {createStateHeatmap,summarizeHeatmap,mergeHeatmapCells}=await import(MAP_URI);
const {drawHeatmap}=await import(CHART_URI);
const scored=(i,t=i*2)=>({window_id:`task:${i}`,t,index:2,score:.6,status:'scored',signal_usable:true,color:'#6E9E8A',label:'中等'});
const pending=i=>({window_id:`baseline:${i}`,t:i*2,index:-1,score:null,status:'baseline_pending',signal_usable:true,
  color:'#79aaa2',label:'信号合格 · 待基线',reason_codes:['baseline_pending'],reason_labels:['正在建立个人基线']});
const limited=i=>({window_id:`task:${i}`,t:i*2,index:-1,score:null,status:'signal_limited',signal_usable:false,
  color:'#d9ab71',label:'信号受限',reason_codes:['amplitude'],reason_labels:['幅值超限']});
const legends=[{status:'baseline_pending',label:'信号合格 · 待基线',color:'#79aaa2'},{status:'signal_limited',label:'信号受限',color:'#d9ab71'}];
const scoreLegend=Array.from({length:5},(_,i)=>({label:`档${i}`,range:`${i/5}–${(i+1)/5}`,color:'#6E9E8A'}))
  .concat([{label:'低质量缺失',range:'—',color:'#6b6b6b'}]);
'''


@unittest.skipUnless(shutil.which("node"), "Node required for native DOM interaction tests")
class HeatmapFrontendTests(unittest.TestCase):
    def node(self, body):
        program = DOM + body
        for name, path in {"MAP_URI": "web/js/views/state-heatmap.js", "CHART_URI": "web/js/charts.js",
                           "LIVE_URI": "web/js/views/live.js", "API_URI": "web/js/api.js"}.items():
            program = program.replace(name, json.dumps((ROOT / path).as_uri()))
        result = subprocess.run(["node", "--input-type=module", "-e", program],
                                capture_output=True, text=True, encoding="utf-8", timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_signal_quality_is_separate_from_personal_baseline_and_score(self):
        self.node(r'''
import assert from 'node:assert/strict';
const data=[pending(1),pending(2),scored(3),limited(4),{t:10,index:-1,score:null}];
const summary=summarizeHeatmap(data);
assert.deepEqual({total:summary.total,signal:summary.signal,scored:summary.scored,unknown:summary.unknown},
  {total:5,signal:3,scored:1,unknown:1});
assert.equal(summary.counts.baseline_pending,2);assert.equal(summary.counts.signal_limited,1);
const map=createStateHeatmap();map.update({cells:data,availability_legend:legends,legend:scoreLegend});
assert.ok(text(map.node).includes('20%'));assert.ok(text(map.node).includes('3 窗已确认'));
assert.ok(text(map.node).includes('信号合格 · 待基线'));assert.ok(!text(map.node).includes('低质量缺失'));
assert.equal(cellsIn(map.node).length,5);
assert.ok(text(map.node).includes('未采集的行为阶段不补窗'));
assert.ok(text(map.node).includes('窗口时间（分析步进）'));
''')

    def test_no_scored_windows_has_useful_summary_without_a_large_gray_wall(self):
        self.node(r'''
import assert from 'node:assert/strict';
const map=createStateHeatmap();map.update({cells:Array.from({length:700},(_,i)=>pending(i)),availability_legend:legends});
assert.ok(text(map.node).includes('已收到合格信号，状态评分等待个人基线'));
assert.equal(byClass(map.node,'state-map__details').open,false);
assert.ok(cellsIn(map.node).length<=120);assert.ok(text(map.node).includes('0 / 700 窗'));
map.update({cells:[limited(1)],availability_legend:legends});
assert.ok(text(map.node).includes('暂未形成可评分片段'));assert.ok(text(map.node).includes('幅值超限 · 1 窗'));
map.update({cells:[{t:1,index:-1,score:null}]});
assert.ok(text(map.node).includes('历史记录未保存完整的未评分原因'));
''')

    def test_filter_keeps_full_denominator_and_original_time(self):
        self.node(r'''
import assert from 'node:assert/strict';
const map=createStateHeatmap();map.update({cells:[pending(0),limited(1),scored(5,101),scored(9,205)],availability_legend:legends});
const select=all(map.node).find(node=>node.tagName==='SELECT');select.value='scored';await action(select,'change');
assert.equal(cellsIn(map.node).length,2);assert.ok(text(map.node).includes('50%'));
assert.ok(text(map.node).includes('2 / 4 窗'));assert.ok(text(map.node).includes('101.0s — 205.0s'));
assert.ok(cellsIn(map.node)[0].attributes['aria-label'].includes('t=101.0s'));
map.update({cells:[pending(0),limited(1)],availability_legend:legends});
assert.equal(cellsIn(map.node).length,0);assert.ok(text(map.node).includes('此会话没有可评分片段'));
''')

    def test_long_sessions_are_paged_without_truncating_the_coverage_denominator(self):
        self.node(r'''
import assert from 'node:assert/strict';
const rows=Array.from({length:1500},(_,i)=>scored(i));const map=createStateHeatmap();map.update({cells:rows});
assert.equal(cellsIn(map.node).length,60);assert.ok(text(map.node).includes('1441–1500 / 1500 窗'));
await action(byButton(map.node,'上一页'));
assert.equal(cellsIn(map.node).length,120);assert.ok(text(map.node).includes('1321–1440 / 1500 窗'));
map.update({cells:[...rows,scored(1500)]});
assert.ok(text(map.node).includes('1321–1440 / 1501 窗'));assert.ok(text(map.node).includes('1501 / 1501 窗'));
await action(byButton(map.node,'最近片段'));assert.ok(text(map.node).includes('1441–1501 / 1501 窗'));
''')

    def test_replayed_events_and_old_snapshots_are_deduplicated(self):
        self.node(r'''
import assert from 'node:assert/strict';
const first=scored(1,2),second=scored(2,4);
let rows=mergeHeatmapCells([first],[first,second]);assert.equal(rows.length,2);
rows=mergeHeatmapCells([{t:2,index:-1,score:null}],[first]);assert.equal(rows.length,1);assert.equal(rows[0].status,'scored');
rows=mergeHeatmapCells([first],[{...first,window_id:'training:1'}]);assert.equal(rows.length,2);
const summary=summarizeHeatmap([{...first,status:'signal_limited',score:.9,signal_usable:false},
  {...second,score:NaN,status:'scored',signal_usable:null}, {...second,signal_usable:false}]);
assert.equal(summary.scored,0);assert.equal(summary.counts.unscored,2);
''')

    def test_null_or_nonfinite_scores_never_render_as_a_state_band(self):
        self.node(r'''
import assert from 'node:assert/strict';
const root=new StubNode('div');drawHeatmap(root,[pending(1),limited(2),{t:3,index:2,score:NaN,status:'scored'},null],
  {availabilityLegend:legends,legend:scoreLegend,missingColor:'#ccc'});
assert.equal(cellsIn(root).length,4);assert.ok(cellsIn(root).every(node=>node.classList.contains('heatmap__cell--missing')));
assert.ok(cellsIn(root)[0].attributes.title.includes('正在建立个人基线'));
assert.ok(cellsIn(root).every(node=>node.attributes['aria-label'].includes('不输出状态分数')));
assert.ok(!text(root).includes('低质量缺失'));
''')

    def test_live_snapshot_and_training_feedback_keep_all_recorded_windows(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {api}=await import(API_URI);const {render}=await import(LIVE_URI);
let resolveHeat;api.sessionHeatmap=()=>new Promise(resolve=>{resolveHeat=resolve;});
api.getSession=async()=>({data:{session:{uuid:'s',status:'running'}}});
api.sessionLive=async()=>({data:{status:'running',runtime_alive:true}});
const sources=[];globalThis.EventSource=class{constructor(url){this.url=url;this.handlers={};sources.push(this);}
addEventListener(type,fn){this.handlers[type]=fn;}removeEventListener(){}close(){}};
const cleanups=[],ctx={params:new URLSearchParams('session=s'),signal:{aborted:false},navigate(){},
store:{state:{config:{}},setState(patch){Object.assign(this.state,patch);},subscribe(){return()=>{};}},onCleanup(fn){cleanups.push(fn);}};
const root=new StubNode('main');render(root,ctx);await flush();
const events=sources.find(source=>source.url.endsWith('/events'));
const emit=(type,payload)=>events.handlers[type]({data:JSON.stringify({data:payload})});
emit('window',{index:1,t_end:2,usable:true,heatmap_cell:pending(1)});
emit('feedback',{heatmap_cell:scored(2,4)});
resolveHeat({data:{cells:[pending(1)],availability_legend:legends}});await flush();
assert.ok(text(root).includes('1 / 2 窗'));assert.ok(text(root).includes('50%'));
emit('feedback',{heatmap_cell:scored(2,4)});assert.ok(text(root).includes('1 / 2 窗'));
for(let i=3;i<=605;i++)emit('feedback',{heatmap_cell:scored(i,i*2)});
assert.ok(text(root).includes('604 / 605 窗'));assert.ok(cellsIn(root).length<=120);
const countBox=all(root).find(node=>node.children?.some(child=>child.textContent==='已接收窗'));
assert.equal(text(countBox),'已接收窗605');
ctx.signal.aborted=true;cleanups.forEach(fn=>fn());
''')

    def test_live_window_explanation_survives_summaries_and_clears_at_phase_change(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {api}=await import(API_URI);const {render}=await import(LIVE_URI);
const record={index:2,phase:'monitor',t_end:4,usable:true,baseline_valid:true,scores:{focus:.73},
  focus_basis:{window_log_ratio:Math.log(2),baseline_log_ratio:Math.log(1.5)},reasons:[]};
api.getSession=async()=>({data:{session:{uuid:'s',status:'running',phase:'monitor'}}});
api.sessionLive=async()=>({data:{status:'running',runtime_alive:true,phase:'monitor',analysis_window:record}});
api.sessionHeatmap=async()=>({data:{cells:[]}});
const sources=[];globalThis.EventSource=class{constructor(url){this.url=url;this.handlers={};sources.push(this);}
  addEventListener(type,fn){this.handlers[type]=fn;}removeEventListener(){}close(){}};
const cleanups=[],ctx={params:new URLSearchParams('session=s'),signal:{aborted:false},navigate(){},
store:{state:{config:{phases:[{key:'monitor',label:'任务态监测'},{key:'training',label:'反馈训练'}]}},setState(patch){Object.assign(this.state,patch);},subscribe(){return()=>{};}},onCleanup(fn){cleanups.push(fn);}};
const root=new StubNode('main');render(root,ctx);await flush();
const stream=sources.find(source=>source.url.includes('/signal?'));
stream.onopen?.();
assert.ok(text(root).includes('等待新数据'));
stream.handlers.signal({data:JSON.stringify({samples:1,realtime:true,channels:[]})});
assert.ok(text(root).includes('数据流正在更新'));
assert.ok(text(root).includes('任务态监测 · 第 2 窗'));
assert.ok(text(root).includes('本窗 β/θ 2.000 · 睁眼基线 β/θ 1.500'));
const events=sources.find(source=>source.url.endsWith('/events'));
const emit=(type,payload)=>events.handlers[type]({data:JSON.stringify({data:payload})});
emit('monitor',{summary:{indicators:{focus:{mean:.55,n:4}}}});
assert.ok(text(root).includes('监测阶段均值（4 个评分窗） · 专注相关指数 0.55'));
assert.ok(text(root).includes('第 2 窗'));
emit('window',{...record,index:3,t_end:6,usable:false,scores:{},reasons:['clock_unsynchronized']});
assert.ok(text(root).includes('时标待核对'));
assert.ok(text(root).includes('数据时间戳未同步'));
assert.ok(text(root).includes('个人基线 · 已建立'));
assert.ok(!text(root).includes('本窗 β/θ'));
emit('window',record);
assert.ok(text(root).includes('第 3 窗'));
emit('phase',{key:'training',state:'running'});
assert.ok(text(root).includes('等待当前阶段的分析窗'));
assert.ok(!text(root).includes('暂停评分 · 数据时间戳未同步'));
emit('feedback',{analysis_window:{...record,index:4,phase:'training',t_end:8}});
assert.ok(text(root).includes('第 4 窗'));
assert.ok(text(root).includes('本窗 β/θ'));
ctx.signal.aborted=true;cleanups.forEach(fn=>fn());
''')

    def test_history_snapshot_updates_received_window_count_without_sse(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {api}=await import(API_URI);const {render}=await import(LIVE_URI);
let resolveHeat;api.sessionHeatmap=()=>new Promise(resolve=>{resolveHeat=resolve;});
api.getSession=async()=>({data:{session:{uuid:'history',status:'done'}}});
api.sessionLive=async()=>({data:{status:'done',runtime_alive:false}});
globalThis.EventSource=class{addEventListener(){}removeEventListener(){}close(){}};
const cleanups=[],ctx={params:new URLSearchParams('session=history'),signal:{aborted:false},navigate(){},
store:{state:{config:{}},setState(patch){Object.assign(this.state,patch);},subscribe(){return()=>{};}},onCleanup(fn){cleanups.push(fn);}};
const root=new StubNode('main');render(root,ctx);await flush();
const count=()=>text(all(root).find(node=>node.children?.some(child=>child.textContent==='已接收窗')));
assert.equal(count(),'已接收窗0');
resolveHeat({data:{cells:Array.from({length:45},(_,i)=>limited(i)),availability_legend:legends}});await flush();
assert.equal(count(),'已接收窗45');assert.ok(text(root).includes('0 / 45 窗'));
ctx.signal.aborted=true;cleanups.forEach(fn=>fn());
''')

    def test_persisted_failure_reason_survives_both_request_orders(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {api}=await import(API_URI);const {render}=await import(LIVE_URI);
globalThis.EventSource=class{addEventListener(){}removeEventListener(){}close(){}};
for(const detailFirst of [true,false]){
  let resolveDetail,resolveLive;
  api.getSession=()=>new Promise(resolve=>{resolveDetail=resolve;});
  api.sessionLive=()=>new Promise(resolve=>{resolveLive=resolve;});
  api.sessionHeatmap=async()=>({data:{cells:[]}});
  const cleanups=[],ctx={params:new URLSearchParams('session=failed'),signal:{aborted:false},navigate(){},
    store:{state:{config:{}},setState(patch){Object.assign(this.state,patch);},subscribe(){return()=>{};}},onCleanup(fn){cleanups.push(fn);}};
  const root=new StubNode('main');render(root,ctx);
  const detail={data:{session:{uuid:'failed',status:'failed',error:'设备连接已中断'}}};
  const live={data:{status:'failed',runtime_alive:false}};
  if(detailFirst){resolveDetail(detail);await flush();resolveLive(live);}
  else{resolveLive(live);await flush();resolveDetail(detail);}
  await flush();
  assert.ok(text(root).includes('会话运行失败：设备连接已中断'));
  assert.ok(!text(root).includes('会话运行失败：未知原因'));
  assert.equal(ctx.store.state.currentSession.error,'设备连接已中断');
  ctx.signal.aborted=true;cleanups.forEach(fn=>fn());
}
''')
