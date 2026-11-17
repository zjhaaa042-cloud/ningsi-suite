"""History must show the actual cohort size and its protocol/source label."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node required for history rendering')
class HistoryFrontendTests(unittest.TestCase):
    def test_comparison_count_and_context_are_not_total_session_count(self):
        program = NODE_STUB + r'''
import assert from 'node:assert/strict';
document.createElementNS=(_,tag)=>new StubNode(tag);
document.querySelector=()=>null;
StubNode.prototype.append=function(...nodes){this.children.push(...nodes);};
globalThis.window={location:{search:''}};
const {api}=await import(API_URI);
const records=[
  {uuid:'test',participant:'p01',status:'done',protocol_label:'完整协议',device:'sim-bsense',
    time_scale:.03,has_artifacts:true,has_report:true,has_training:true},
  {uuid:'cancel',status:'cancelled',has_artifacts:false,has_report:false,has_training:false},
  {uuid:'partial',status:'cancelled',has_artifacts:true,has_report:false,has_training:true},
  {uuid:'short',status:'done',has_artifacts:true,has_report:true,has_training:false},
];
api.listSessions=async()=>({data:{items:records,total:records.length}});
api.reportsTrend=async()=>({data:{sessions:4,comparable:1,field:'focus',period:'week',
  points:[{period:'2026-W47',mean:.4,n:1}],comparison_group:{protocol_label:'完整协议',
  training_mode:'quick',source_kind:'sim',time_scale:.03}}});
const {render}=await import(VIEW_URI);
const root=new StubNode('main');
const ctx={signal:{aborted:false},store:{state:{subjects:[{public_id:'p01'}]},setState(){}},
  reload(){},navigate(){}};
await render(root,ctx);
const text=node=>typeof node==='string'?node:(node.textContent||'')+(node.children||[]).map(text).join('');
assert.ok(text(root).includes('参与聚合 1 次 / 已完成 4 次'));
assert.ok(!text(root).includes('参与聚合会话 4 次'));
assert.ok(text(root).includes('当前分组：完整协议 · 简短训练 · 模拟数据 · 时间倍率 0.03'));
assert.ok(text(root).includes('模拟 · sim-bsense'));
const all=node=>[node,...(node.children||[]).flatMap(all)];
const links=all(root).filter(node=>node.tagName==='A');
const labelsFor=id=>links.filter(node=>node.attributes['aria-label']?.includes('会话 '+id+' '))
  .map(node=>node.textContent);
assert.deepEqual(labelsFor('cancel'),['阶段记录']);
assert.deepEqual(labelsFor('partial'),['导出 zip','阶段记录','训练']);
assert.deepEqual(labelsFor('short'),['导出 zip','报告']);
assert.deepEqual(labelsFor('test'),['导出 zip','报告','训练']);
'''
        program = program.replace('API_URI', json.dumps((ROOT / 'web/js/api.js').as_uri()))
        program = program.replace('VIEW_URI', json.dumps((ROOT / 'web/js/views/history.js').as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', program],
                                capture_output=True, text=True, encoding='utf-8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
