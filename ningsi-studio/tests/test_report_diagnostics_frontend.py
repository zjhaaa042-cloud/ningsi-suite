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

    def test_summary_cards_use_saved_results_before_technical_disclosures(self):
        self.node(r'''
const {render}=await import(REPORT_URI);
const ctx=makeCtx();ctx.params.set('session','x');
api.getSession=async()=>({data:{uuid:'x'}});
api.artifacts=async()=>({data:{items:[]}});
api.report=async()=>({data:{participant:'qa-hidden',extras:{source_kind:'sim',
  report_context:{title:'演示报告',hide_participant:true}},quality:{windows_total:20,windows_usable:19},
  scales:{SAS:{answered:20,name:'焦虑自评量表'},SDS:{answered:17,name:'抑郁自评量表'}},
  behavior:{sart:{trials:180,commission_errors:0,omission_errors:1},pvt:{trials:72,responded:70,rt_median:.198}},
  assessment:{conclusion:'个人状态结论'},report_markdown:'# 原始记录'}});
const root=new StubNode('main');await render(root,ctx);
const cards=all(root).filter(n=>n.classList?.contains('report-metric')).map(text);
assert.equal(cards.length,4);assert.match(cards[0],/95%.*20.*19/);assert.match(cards[1],/37 题.*20 题.*17 题/);
assert.match(cards[2],/180 次.*误按 0 次.*漏按 1 次/);assert.match(cards[3],/198 ms.*72 次.*70 次/);
assert.doesNotMatch(text(root),/个人状态结论|qa-hidden|原始记录/);
const panel=all(root).find(n=>n.tagName==='DETAILS'&&text(n.children[0])==='查看采集质量与行为分析');
assert.ok(!panel.open);assert.equal(panel.children.length,1);
panel.open=true;panel.listeners.toggle[0]();assert.equal(panel.children.length,2);
panel.open=false;panel.listeners.toggle[0]();panel.open=true;panel.listeners.toggle[0]();assert.equal(panel.children.length,2);
''')

    def test_report_preview_preserves_origin_response_mode_and_zero_coverage(self):
        self.node(r'''
const {render}=await import(REPORT_URI);
const ctx=makeCtx(); ctx.params.set('session','archive-id');
api.getSession=async()=>({data:{uuid:'archive-id'}});
let fileExists=true;
api.artifacts=async()=>({data:{items:[{kind:'report_json',exists:fileExists}]}});
const show=async extras=>{
  api.report=async()=>({status:200,data:{extras,report_markdown:'# 已保存记录',
    assessment:{conclusion:'受试者测评结论',advice:['受试者训练建议']},behavior:{pvt:{trials:21,rt_median:.3}}}});
  const root=new StubNode('main'); await render(root,ctx); return root;
};
const base={protocol:'ten_min',source_kind:'lsl',ten_min:{indicator_overview:{quality:{windows_total:100}},
  segments:[{measured_focus_updates:0,stats:{valid_seconds:0}},{measured_focus_updates:0,stats:{valid_seconds:0}}]}};
const replay=await show({...base,report_context:{title:'录波测试',source_label:'本地录波回放 · 自动反应时任务',
  signal_origin:'local_replay',response_mode:'automated',recording_ref:'原始采集 #3'}});
assert.match(text(replay),/录波测试.*本地录波回放 · 自动反应时任务.*原始采集 #3/);
assert.match(text(replay),/100 次训练更新中，0 次显示专注指数/);
assert.match(text(replay),/由程序自动作答/);
assert.doesNotMatch(text(replay),/受试者测评结论|受试者训练建议/);
const link=all(replay).find(n=>n.tagName==='A'&&text(n)==='打开展示报告');
assert.equal(link.getAttribute('href'),'/api/sessions/archive-id/presentation');
fileExists=false;
const generated=await show({...base,device:'ningsi-sim-eeg'});
assert.match(text(generated),/数据来源　生成信号/);
assert.doesNotMatch(text(generated),/本地录波回放|由程序自动作答|受试者训练建议|打开展示报告/);
const ordinary=await show({...base,device:'BioMulti'});
assert.match(text(ordinary),/数据来源　LSL 输入/);
assert.doesNotMatch(text(ordinary),/由程序自动作答|本地录波回放/);
''')
