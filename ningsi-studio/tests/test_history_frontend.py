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
globalThis.window={location:{search:'?token=local-test',origin:'http://localhost:8765'}};
const {api}=await import(API_URI);
const records=[
  {uuid:'test',participant:'p01',status:'done',protocol_label:'完整协议',device:'sim-bsense',
    time_scale:.03,has_artifacts:true,has_report:true,has_training:true},
  {uuid:'cancel',status:'cancelled',has_artifacts:false,has_report:false,has_training:false},
  {uuid:'partial',status:'cancelled',has_artifacts:true,has_report:false,has_training:true},
  {uuid:'short',status:'done',has_artifacts:true,has_report:true,has_training:false},
  {uuid:'archive2',label:'样例二次测试',participant:'qa12',status:'done',archived:true,
    origin_label:'工程测试 · 生成信号（LSL）',report_url:'/api/test-history/archive2/report',
    evidence_url:'/api/test-history/archive2/evidence'},
  {uuid:'older',label:'历史采集',status:'unfinished',archived:true,
    origin_label:'本地设备',evidence_url:'/api/test-history/older/evidence'},
  {uuid:'aa11bb221234',label:'演示场次三',participant:'qa13',hide_participant:true,status:'done',archived:true,
    origin_label:'仿真脑电 · 自动反应时任务',presentation_url:'/api/test-history/aa11bb221234/presentation'},
  {uuid:'cc33dd441234',label:'样例问卷测试',participant:'qa14',hide_participant:true,status:'done',
    archived:true,origin_label:'仿真脑电 · 测试问卷与自动作答',evidence_url:'/api/test-history/cc33dd441234/evidence'},
];
api.listTestHistory=async({page,limit})=>({data:{items:records,total:43,page,limit,participants:['p01','qa12']}});
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
assert.ok(text(root).includes('模拟数据'));
assert.ok(text(root).includes('样例二次测试'));
assert.ok(text(root).includes('工程测试 · 生成信号（LSL）'));
assert.ok(text(root).includes('第 1 / 3 页，共 43 条'));
assert.ok(text(root).includes('未记录结束'));
assert.ok(text(root).includes('演示场次三'));
assert.ok(text(root).includes('样例问卷测试'));
assert.ok(text(root).includes('sub-p01'));
assert.doesNotMatch(text(root), /qa13|qa14|aa11bb22|cc33dd44/);
const all=node=>[node,...(node.children||[]).flatMap(all)];
const links=all(root).filter(node=>node.tagName==='A');
const labelsFor=id=>links.filter(node=>node.attributes['aria-label']?.includes('会话 '+id+' '))
  .map(node=>node.textContent);
assert.deepEqual(labelsFor('cancel'),['阶段记录']);
assert.deepEqual(labelsFor('partial'),['导出 zip','阶段记录','训练']);
assert.deepEqual(labelsFor('short'),['展示报告','导出 zip','报告详情']);
assert.deepEqual(labelsFor('test'),['展示报告','导出 zip','报告详情','训练']);
for (const id of ['test','short']) {
  const href='/api/sessions/'+id+'/presentation?token=local-test';
  const resultLinks=links.filter(node=>node.getAttribute('href')===href);
  assert.equal(resultLinks.length,2);
  assert.ok(resultLinks.every(node=>node.getAttribute('target')==='_blank'));
  assert.ok(links.some(node=>node.textContent==='报告详情' && node.getAttribute('href')==='#/report?session='+id));
}
assert.ok(!links.some(node=>node.getAttribute('href')?.includes('/sessions/partial/presentation')));
assert.ok(links.some(node=>text(node).includes('样例二次测试') &&
  node.getAttribute('href')==='/api/test-history/archive2/report?token=local-test'));
assert.ok(links.some(node=>text(node).includes('历史采集') &&
  node.getAttribute('href')==='/api/test-history/older/evidence?token=local-test'));
assert.ok(!links.some(node=>node.getAttribute('href')?.includes('session=archive2')));

// The newest archive belongs in the table; the live summary must not look up that archived UUID.
api.health=async()=>({data:{}});
api.overview=async()=>({data:{sessions:0,latest_detail:[]}});
api.config=async()=>({data:{}});
ctx.store.setState=()=>{};
api.listTestHistory=async()=>({data:{items:[records[4], ...records.slice(6), records[0]],total:43}});
api.getSession=async()=>{ assert.fail('archive UUID cannot be opened as a live Studio session'); };
const dashboard=await import(DASHBOARD_URI);
const home=new StubNode('main');
await dashboard.render(home,ctx);
assert.ok(text(home).includes('样例二次测试'));
assert.ok(text(home).includes('共 43 条记录'));
assert.ok(text(home).includes('工程测试 · 生成信号（LSL）'));
assert.ok(text(home).includes('演示场次三'));
assert.ok(text(home).includes('仿真脑电 · 自动反应时任务'));
assert.doesNotMatch(text(home), /qa13|qa14|aa11bb22|cc33dd44/);
assert.ok(all(home).some(node=>node.getAttribute?.('href')==='/api/test-history/aa11bb221234/presentation?token=local-test'));
assert.ok(all(home).some(node=>node.getAttribute?.('href')==='/api/test-history/archive2/report?token=local-test'));
const livePresentation=all(home).filter(node=>node.getAttribute?.('href')==='/api/sessions/test/presentation?token=local-test');
assert.equal(livePresentation.length,2);
assert.ok(livePresentation.some(node=>text(node)==='展示报告'));
'''
        program = program.replace('API_URI', json.dumps((ROOT / 'web/js/api.js').as_uri()))
        program = program.replace('VIEW_URI', json.dumps((ROOT / 'web/js/views/history.js').as_uri()))
        program = program.replace('DASHBOARD_URI', json.dumps((ROOT / 'web/js/views/dashboard.js').as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', program],
                                capture_output=True, text=True, encoding='utf-8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
