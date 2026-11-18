"""Older reports can lack newer optional diagnostic fields."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_readiness_frontend import DOM

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node required for native DOM interaction tests')
class ReportDiagnosticsFrontendTests(unittest.TestCase):
    def node(self, body):
        program = DOM + r'''
import assert from 'node:assert/strict';
Object.defineProperty(StubNode.prototype,'childNodes',{get(){return this.children;}});
''' + body
        for token, path in {
            'API_URI': 'web/js/api.js',
            'DIAGNOSTICS_URI': 'web/js/views/report-diagnostics.js',
            'REPORT_URI': 'web/js/views/report.js',
        }.items():
            program = program.replace(token, json.dumps((ROOT / path).as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', program], capture_output=True,
                                text=True, encoding='utf-8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_partial_or_absent_saved_diagnostics_remain_readable(self):
        self.node(r'''
const {renderDiagnostics}=await import(DIAGNOSTICS_URI);
for(const report of [
 {quality:{windows_total:2,windows_usable:1}},
 {extras:{signal_diagnostics:{total:2,accepted:1}},quality:{}},
 {extras:{signal_diagnostics:{total:2,accepted:1,note:'保存的诊断说明'}}},
 {quality:{windows_total:2,windows_usable:1},behavior:{pvt:{},sart:{}}},
]) {
 const root=renderDiagnostics(report),output=text(root);
 assert.ok(output.includes('严格合格 1/2 窗'));
 assert.ok(!output.includes('undefined'));assert.ok(!output.includes('NaN'));
 assert.ok(!output.includes('PVT 行为分析'));
 if(report.extras?.signal_diagnostics?.note)assert.ok(output.includes('保存的诊断说明'));
 else assert.ok(output.includes('旧版本未保存完整逐窗信号')||output.includes('缺失的诊断项目不补算'));
}
assert.ok(text(renderDiagnostics({})).includes('未保存完整窗口质量计数'));
''')

    def test_first_screen_uses_matching_eeg_scope_and_shows_behavior_separately(self):
        self.node(r'''
const {render}=await import(REPORT_URI);
const ctx=makeCtx();ctx.params.set('session','test-session');
api.getSession=async()=>({data:{session:{uuid:'test-session'}}});
api.artifacts=async()=>({data:{items:[]}});
const summary=async report=>{
  api.report=async()=>({status:200,data:{report_markdown:'# 专业报告',...report}});
  const root=new StubNode('main');await render(root,ctx);
  const card=all(root).find(node=>node.classList?.contains('report-summary'));
  assert.ok(card);
  assert.ok(all(root).some(node=>node.tagName==='SUMMARY'&&text(node)==='查看采集质量与行为分析'));
  return Object.fromEntries(all(card).filter(node=>node.tagName==='SECTION').map(section=>[
    section.children[0].textContent,text(section),
  ]));
};
const eeg={indicators:{summary:{focus:{mean:.61,n:17}}},quality:{windows_total:24,windows_usable:19}};
const pvt={valid:true,rt_median:.278,lapse_rate:.1,omission_rate:.05};
const baseline_eyes_open={valid:true,quality:{usable_seconds:32,valid_ratio:.72}};
const full=await summary({...eeg,extras:{protocol:'full',baseline_eyes_open},behavior:{
  sart:{commission_rate:.2,omission_rate:.05,rt_variability:.35},pvt,
},assessment:{evidence:[{code:'sart_performance',available:true,direction:'low'},
                        {code:'pvt_alertness',available:true,direction:'normal'}]}});
assert.match(full['脑电指数'],/独立监测阶段.*专注相关指数 0\.61.*专注指数有效窗 17\/24 窗.*信号合格窗 19\/24 窗.*有效覆盖 32\.0 秒.*72\.0%/);
assert.match(full.SART,/触发工程阈值.*虚报率 20\.0%.*漏报率 5\.0%.*反应时变异系数 0\.35/);
assert.match(full.PVT,/未触发工程阈值.*中位反应时 278 ms.*慢反应率 10\.0%.*漏答率 5\.0%/);
const overview={summary:{focus:{mean:.42,n:6}},quality:{windows_total:12,windows_usable:8}};
for (const protocol of ['ten_min','effective_time']) {
  const training=await summary({...eeg, extras:{protocol,[protocol]:{indicator_overview:overview},baseline_eyes_open},behavior:{pvt}});
  assert.match(training['脑电指数'],/训练阶段（描述性）.*0\.42.*6\/12 窗.*8\/12 窗.*未采集独立任务态评分所需的监测脑电/);
  assert.doesNotMatch(training['脑电指数'],/独立监测阶段|0\.61|17\/24/);
  assert.equal(training.SART,'SART未采集');
  assert.match(training.PVT,/278 ms/);
}
const replay=await summary({...eeg,extras:{replay:{participant:'S001'}}});
assert.match(replay['脑电指数'],/公开录波的运动想象阶段/);
assert.equal(replay.SART,'SART未采集');assert.equal(replay.PVT,'PVT未采集');
const legacy=await summary({behavior:{pvt:{valid:false,responded:0,rt_median:0,lapse_rate:0,omission_rate:1}}});
assert.match(legacy['脑电指数'],/来源阶段未记录.*暂无可用脑电指数.*有效覆盖未记录/);
assert.doesNotMatch(legacy['脑电指数'],/0\/0|undefined|NaN/);
assert.equal(legacy.SART,'SART暂无可用结果');
assert.match(legacy.PVT,/行为证据不可用.*漏答率 100\.0%/);
assert.doesNotMatch(legacy.PVT,/中位反应时|慢反应率/);
''')

    def test_pvt_slow_response_bar_uses_the_valid_response_denominator(self):
        self.node(r'''
const {renderDiagnostics}=await import(DIAGNOSTICS_URI);
const bar=report=>all(renderDiagnostics(report)).find(node=>node.tagName==='PROGRESS'&&node.getAttribute('aria-label')==='慢反应 >500ms');
const pvt={trials:12,responded:7,lapses:2};
assert.equal(bar({behavior:{pvt}}).getAttribute('max'),'7');
assert.equal(bar({behavior:{pvt:{...pvt,rate_denominators:{lapse_rate:6}}}}).getAttribute('max'),'6');
assert.equal(bar({behavior:{pvt:{trials:12,lapses:2}}}),undefined);
assert.equal(bar({behavior:{pvt:{...pvt,responded:0,lapses:0}}}),undefined);
''')
