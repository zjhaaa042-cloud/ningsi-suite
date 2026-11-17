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
const first=new StubNode('main');
await render(first,ctx);
assert.ok(all(first).some(n=>n.tagName==='H1'&&text(n)==='后端报告原文'));
assert.ok(text(first).includes('训练时间统计与独立测评分别保留'));
report={participant:'p01',extras:{},indicators:{summary:{}}};
const legacy=new StubNode('main');
await render(legacy,ctx);
assert.ok(all(legacy).some(n=>n.tagName==='H2'&&text(n)==='一、状态评分'),'old JSON-only reports retain a readable fallback');
report={participant:'p01',extras:{protocol:'effective_time',effective_time:{acquisition_complete:false,
  collection_status:'insufficient'}},indicators:{summary:{}}};
const insufficient=new StubNode('main');await render(insufficient,ctx);
assert.ok(text(insufficient).includes('本协议未采集独立任务态评分'));
assert.ok(text(insufficient).includes('采集不足：流程已结束'));
assert.ok(text(insufficient).includes('不能据此判断训练效果'));
assert.ok(!text(insufficient).includes('指标为本次有效脑电窗口的均值'));
'''
        program = program.replace('API_URI', json.dumps((ROOT/'web/js/api.js').as_uri()))
        program = program.replace('VIEW_URI', json.dumps((ROOT/'web/js/views/report.js').as_uri()))
        result = subprocess.run(['node','--input-type=module','-e',program],
                                capture_output=True,text=True,encoding='utf-8',timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
