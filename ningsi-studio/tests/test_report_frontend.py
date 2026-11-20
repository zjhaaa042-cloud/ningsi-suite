"""The page must display the saved report, including protocol-specific evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node required for report rendering regression')
class ReportFrontendTests(unittest.TestCase):
    def test_saved_markdown_is_used_and_legacy_json_still_has_a_fallback(self):
        program = NODE_STUB + r'''
import assert from 'node:assert/strict';
document.querySelector=()=>null;
globalThis.window={location:{search:''}};
StubNode.prototype.append=function(...nodes){this.children.push(...nodes);};
Object.defineProperty(StubNode.prototype,'childNodes',{get(){return this.children;}});
const {api}=await import(API_URI);
let report={participant:'p01',extras:{},report_markdown:'# 后端报告原文\n\n训练时间统计与独立测评分别保留。'};
api.report=async()=>({data:report});
api.getSession=async()=>({data:{uuid:'test',status:'done'}});
api.artifacts=async()=>({data:{items:[]}});
const {render}=await import(VIEW_URI);
const all=node=>[node,...(node.children||[]).flatMap(all)];
const text=node=>typeof node==='string'?node:(node.textContent||'')+(node.children||[]).map(text).join('');
const ctx={params:new URLSearchParams('session=test'),signal:{aborted:false},
  store:{state:{},setState(){}},navigate(){},onCleanup(){}};
const revealSaved=root=>{
  const panel=all(root).find(n=>n.tagName==='DETAILS'&&text(n.children[0])==='查看完整报告与评估证据');
  assert.ok(panel&&!panel.open); panel.open=true; panel.listeners.toggle[0]();
};
const first=new StubNode('main');
await render(first,ctx);
assert.ok(!text(first).includes('后端报告原文'));
revealSaved(first);
assert.ok(all(first).some(n=>n.tagName==='H1'&&text(n)==='后端报告原文'));
assert.ok(text(first).includes('训练时间统计与独立测评分别保留'));
report={participant:'p01',extras:{},indicators:{summary:{}}};
const legacy=new StubNode('main');
await render(legacy,ctx);
revealSaved(legacy);
assert.ok(all(legacy).some(n=>n.tagName==='H2'&&text(n)==='一、状态评分'),'old JSON-only reports retain a readable fallback');
report={participant:'p01',extras:{protocol:'effective_time',effective_time:{acquisition_complete:false,
  collection_status:'insufficient'}},indicators:{summary:{}}};
const insufficient=new StubNode('main');await render(insufficient,ctx);
assert.ok(text(insufficient).includes('本协议未采集独立任务态评分'));
assert.ok(text(insufficient).includes('采集不足：流程已结束'));
assert.ok(text(insufficient).includes('不能据此判断训练效果'));
assert.ok(!text(insufficient).includes('指标为本次有效脑电窗口的均值'));
report={participant:'p01',extras:{protocol:'short',assessment_eeg_collection:{
  baseline_open:{target_seconds:60,valid_seconds:26,elapsed_seconds:180,max_seconds:180,
    completed:false,completion_reason:'max_duration',status_reason:'insufficient_effective_time',
    shortfall_seconds:34,valid_ratio:.32,min_valid_ratio:.6},
  monitor:{target_seconds:24,valid_seconds:24,elapsed_seconds:40,max_seconds:120,
    completed:false,completion_reason:'baseline_invalid',shortfall_seconds:0}}},
  indicators:{summary:{}},report_markdown:'# 保存的报告'};
const assessment=new StubNode('main');await render(assessment,ctx);
revealSaved(assessment);
assert.ok(text(assessment).includes('评估脑电采集结果'));
assert.ok(text(assessment).includes('有效脑电 26.0s / 目标 1:00.0'));
assert.ok(text(assessment).includes('用时 3:00.0 / 采集上限 3:00.0'));
assert.ok(text(assessment).includes('还差 34.0s 有效脑电'));
assert.ok(text(assessment).includes('信号合格比例 32.0% / 要求 60.0%'));
assert.ok(text(assessment).includes('睁眼基线未通过，任务态脑电指数不可用'));
assert.ok(!text(assessment).includes('还差 0.0s'));
assert.ok(text(assessment).includes('保存的报告'));
report={participant:'p01',extras:{state_observations:{}},behavior:{pvt:null,sart:null},
  indicators:{summary:{}}};
const sparse=new StubNode('main');await render(sparse,ctx);
revealSaved(sparse);
assert.ok(!text(sparse).includes('报告加载失败'),'partial legacy observations and null behavior must not hide the report');
assert.ok(all(sparse).some(n=>n.tagName==='H2'&&text(n)==='一、状态评分'));
report={partial:true,status:'failed',phase:'failed',error:'模拟连接中断',
  runs:[{phase:'qc',status:'done'}]};
const ended=new StubNode('main');await render(ended,ctx);
assert.ok(text(ended).includes('本次会话无完整报告'));
assert.ok(text(ended).includes('模拟连接中断')&&text(ended).includes('阶段执行情况'));
assert.ok(!text(ended).includes('刷新报告')&&!text(ended).includes('导出 zip'));
report={...report,status:'running',phase:'qc',error:null};
const running=new StubNode('main');await render(running,ctx);
assert.ok(text(running).includes('刷新报告'));
'''
        program = program.replace('API_URI', json.dumps((ROOT/'web/js/api.js').as_uri()))
        program = program.replace('VIEW_URI', json.dumps((ROOT/'web/js/views/report.js').as_uri()))
        result = subprocess.run(['node','--input-type=module','-e',program],
                                capture_output=True,text=True,encoding='utf-8',timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
