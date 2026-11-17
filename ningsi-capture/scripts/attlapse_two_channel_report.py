"""Build the bilingual-keyed results grid and the Chinese offline study report."""

import argparse
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GATES = {"strict": "双通道及差分严格质控", "bipolar_strict": "仅差分做频带伪迹筛查", "physical": "仅物理质控"}
WINDOWS = {"pre12_2": "10 秒", "pre18_2": "16 秒", "pre32_2": "30 秒"}
FEATURES = {"bipolar6": "差分 6 维", "bilateral15": "双侧 15 维", "bilateral_variability18": "双侧与波动 18 维"}
MODELS = {"logistic": "逻辑回归", "lda": "收缩 LDA", "rbf": "RBF 核岭回归"}
FROZEN_SCOPE = "pipeline/physical/pre18_2/bipolar6/"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def num(value):
    return "—" if value is None else f"{value:.3f}"


def interval(bounds):
    return "—" if bounds is None else "–".join(num(v) for v in bounds)


def table(headers, rows):
    def cells(tag, values):
        return "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values)
    head = "<thead><tr>" + cells("th", headers) + "</tr></thead>"
    body = "".join("<tr>" + cells("td", row) + "</tr>" for row in rows)
    return f'<div class="overflow"><table>{head}<tbody>{body}</tbody></table></div>'


def forest(reports, diagnostics, path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.0, 4.0), layout="constrained")
    for rate, offset, color in ((512, .12, "#255e91"), (250, -.12, "#aa4f2e")):
        data = []
        for family in ("logistic", "rbf"):
            m = reports[rate]["results"][FROZEN_SCOPE + family]["metrics"]
            data.append((m["macro_auc"], m["ci95"]["macro_auc"]))
        for key in ("flag_only_logistic", "time_only_logistic", "time_and_bipolar6_logistic"):
            m = diagnostics[rate][key]
            data.append((m["macro_auc"], m["participant_bootstrap_ci95"]["macro_auc"]))
        for i, (point, (lo, hi)) in enumerate(data):
            ax.errorbar(point, 4-i+offset, xerr=[[point-lo], [hi-point]], fmt="o",
                        color=color, capsize=3, lw=1.5, label=f"{rate} Hz" if i == 0 else None)
    ax.axvline(.5, color="#777777", linestyle="--", lw=1, zorder=0)
    ax.set(yticks=range(4, -1, -1), yticklabels=["EEG / logistic", "EEG / RBF", "Artifact flags only",
                                                  "Probe order only", "Order + EEG / logistic"],
           xlim=(.30, .83), ylim=(-.65, 5.05), xlabel="Mean within-person AUROC (95% participant bootstrap interval)")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.grid(axis="x", alpha=.15)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def build(root):
    reports = {hz: read(root / folder / "report.json") for hz, folder in ((512, "evaluation"), (250, "evaluation250"))}
    diagnostic = {int(row["rate_hz"]): row for row in read(root / "artifact_time_diagnostic.json")["results"]}
    coverage_rows, search_rows, diagnostic_rows, feature_rows, paired_rows = [], [], [], [], []
    grid = []
    for hz, report in reports.items():
        results = report["results"]
        for gate, title in GATES.items():
            m = results["selected/" + gate]["metrics"]
            search_rows.append([hz, title, m["scored_labelled"], m["scored_participants"],
                                m["both_classes_scored_participants"], num(m["macro_auc"]), interval(m["ci95"]["macro_auc"]),
                                num(m["coverage_adjusted_macro_auc"]), num(m["pooled_auc"])])
        for window, title in WINDOWS.items():
            counts = [results[f"pipeline/{gate}/{window}/bipolar6/logistic"]["metrics"]["scored_labelled"] for gate in GATES]
            coverage_rows.append([hz, title, *counts])
        for key, record in results.items():
            if not key.startswith("pipeline/"):
                continue
            _, gate, window, features, family = key.split("/")
            m = record["metrics"]
            grid.append(dict(hz=hz, quality=gate, window=window, features=features, model=family,
                             probes=m["scored_labelled"], participants=m["scored_participants"],
                             participants_with_both=m["both_classes_scored_participants"], macro_auc=m["macro_auc"],
                             macro_ci_low=m["ci95"]["macro_auc"][0], macro_ci_high=m["ci95"]["macro_auc"][1],
                             coverage_adjusted=m["coverage_adjusted_macro_auc"], pooled_auc=m["pooled_auc"],
                             balanced_accuracy=m["balanced_accuracy"], probe_coverage=m["probe_coverage"]))
        for family, fname in MODELS.items():
            vals = []
            for feature in FEATURES:
                m = results[f"pipeline/physical/pre18_2/{feature}/{family}"]["metrics"]
                vals.append(num(m["macro_auc"]))
            feature_rows.append([hz, fname, *vals])
        for key, title in (("flag_only_logistic", "仅眼动及肌电代理计数"), ("later_probe_higher", "仅探针序号，越晚越高"),
                           ("bipolar6_only_logistic", "仅差分 6 维"), ("time_and_bipolar6_logistic", "探针序号与差分 6 维")):
            m = diagnostic[hz][key]
            diagnostic_rows.append([hz, title, m["scored_labelled"], num(m["macro_auc"]),
                                    interval(m["participant_bootstrap_ci95"]["macro_auc"]),
                                    num(m["coverage_adjusted_macro_auc"])])
        pair = diagnostic[hz]["time_plus_bipolar6_minus_time_only"]["macro_auc"]
        paired_rows.append(f'{hz} Hz 的配对差为 {pair["difference"]:+.3f}（95% 区间 {pair["ci95"][0]:+.3f} 至 {pair["ci95"][1]:+.3f}）')
    if len(grid) != 162:
        raise ValueError("Expected 81 completed pipelines at each of two sample rates")
    with (root / "comparison_grid.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    chart_path = root / "within_person_comparison.png"
    forest(reports, diagnostic, chart_path)
    encoded = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    qt = table(["采样率", "质控", "探针 / 420", "出分人数", "双类人数", "按人 AUROC", "95% 区间", "覆盖调整", "合并 AUROC"], search_rows)
    ct = table(["采样率", "回看长度", *GATES.values()], coverage_rows)
    ft = table(["采样率", "分类模型", *FEATURES.values()], feature_rows)
    dt = table(["采样率", "输入", "探针 / 420", "按人 AUROC", "95% 区间", "覆盖调整"], diagnostic_rows)
    pair_results = read(root / "paired_candidate_comparisons.json")["rates"]
    contrasts = []
    for hz, result in pair_results.items():
        for row in result["comparisons"]:
            _, _, window, features, model = row["comparator"].split("/")
            lo, hi = row["paired_ci95"]
            contrasts.append([hz, f"{WINDOWS[window]} · {FEATURES[features]} · {MODELS[model]}",
                              row["common_count"], row["common_people_with_both_classes"],
                              f'{row["target_minus_comparator"]:+.3f}', f"{lo:+.3f} 至 {hi:+.3f}"])
    pt = table(["采样率", "比较对象", "共同探针", "双类人数", "按人 AUROC 差", "95% 配对区间"], contrasts)
    top512 = num(reports[512]["results"][FROZEN_SCOPE + "logistic"]["metrics"]["coverage_adjusted_macro_auc"])
    top250 = num(reports[250]["results"][FROZEN_SCOPE + "rbf"]["metrics"]["coverage_adjusted_macro_auc"])
    parts = []
    for hz in reports:
        for gate, title in GATES.items():
            rows = [[WINDOWS[g["window"]], FEATURES[g["features"]], MODELS[g["model"]], g["probes"],
                     g["participants_with_both"], num(g["macro_auc"]), interval([g["macro_ci_low"],g["macro_ci_high"]]),
                     num(g["coverage_adjusted"]), num(g["pooled_auc"]), num(g["balanced_accuracy"])]
                    for g in grid if g["hz"] == hz and g["quality"] == gate]
            parts.append(f"<details><summary>{hz} Hz · {title} · 27 组</summary>" +
                         table(["窗口", "特征", "模型", "探针", "双类人数", "按人 AUROC", "95% 区间",
                                "覆盖调整", "合并 AUROC", "平衡准确率"], rows) + "</details>")
    body = f"""
<header><p>ATTLAPSE 健康对照 · 凝思双通道 EEG</p><h1>两路额极脑电能区分探针前的走神吗？</h1>
<p>28 人 · 420 次自评 · 每个采样率 81 组方案 · 按人留出评估</p></header>
<main><p class="lead">最值得带入本地复核的窗口是提示前 18 至 2 秒，使用 Fp1−Fp2 的 6 个频谱特征。
逻辑回归和 RBF 核岭回归都可用作后台对照。它们的按人区分能力有限，实验暂未证明脑电比单用探针顺序提供更多信息。</p>
<h2>同一人走神时，得分是否比专注时高？</h2>
<p>对每人分别把走神探针与专注探针两两比较，再对有两类可评分记录的人等权平均。随机排序的 AUROC 为 0.5。
图中使用同一个 16 秒窗和同一批通过物理质控的探针。512 Hz 有 280 次，250 Hz 有 278 次，均有 21 人可作双类比较。
两种采样率来自同一份记录。圆点是均值，线段是按受试者重采样的 95% 区间。</p>
<img src="data:image/png;base64,{encoded}" alt="16秒物理质控窗口下的按人 AUROC 及95%区间。两种采样率的时间基线均接近0.70。">
<h2>窗口、特征和模型的差异</h2><p>下表固定 16 秒窗、仅物理质控，仅在训练组选择各模型的正则强度。
6 维包括 θ、α、β 相对功率及三个对数频带比。15 维增加左右各频带功率和左右对数功率差，18 维再加入各子窗间三个频带的波动。数值为按人 AUROC。</p>{ft}
<p>在 512 Hz，16 秒差分 6 维加逻辑回归的覆盖调整分为 {top512}，在仅物理质控的 27 组中居首。250 Hz 对应的最高值为 {top250}，出现在相同窗口、相同特征的 RBF 模型。
这一比较使用已经看过的外层结果，存在择优偏差。两个候选的独立复核应使用新采集的受试者。</p>
<h2>在共同探针上检验窗口与特征差异</h2><p>下表以 16 秒、差分 6 维、RBF 为参照，正数表示参照方案按人 AUROC 更高。
把窗口从 10 秒加长到 16 秒，在同一批探针上平均提高约 0.11。与 30 秒窗相比，512 Hz 的区间跨零。换用逻辑回归、15 维、18 维的配对区间在两种采样率下都跨零。这些都是事后比较，未作多重比较校正。</p>{pt}
<h2>换用质控规则，能给多少探针打分？</h2><p>每个回看窗均在提示出现前 2 秒截止，基于 4 秒子窗、2 秒步长计算。同一窗口和规则下，各特征集共享完全相同的可用信号。
严格规则同时检查两路输入及差分的幅度、平线、通道跨度和眼动、肌电频带代理。中间规则仅在差分上保留频带代理检查。仅物理质控保留物理完整性条件，允许更多带伪迹的信号参与分析。</p>{ct}
<h2>让训练数据自行选择方案后，泛化表现如何？</h2><p>外层五折按人留出，内层四折选择窗口、特征、模型及参数。
同一人的全部探针只属于一个外层测试折。三套质控规则分别选择，不跨规则挑分。不同外层折可选中不同的窗口，所以下表的探针数可以与上表任一固定窗口不同。</p>{qt}
<p>“覆盖调整”是本实验的选型指标。对每个人，两条记录都有分数的正负配对按实际排序计分，其余配对计 0.5 分，再平均全部 28 人。拒评越多，指标越接近 0.5。合并 AUROC 将各人的探针合到一起，混入个体差异，不能直接回答同一个人的状态是否可区分。</p>
<h2>探针顺序与伪迹对照</h2><p>源实验的 15 次探针位于固定试次。事后对照在相同 16 秒物理质控样本、相同外层折上完成。探针序号越大得分越高的对照无需训练；其余三行用固定正则强度 0.1 的逻辑回归。
伪迹输入是两路 EEG 自身被标记为眼动或肌电的窗口比例。结果接近随机，仍无法排除较细微的眼动、肌电或参考电极差异。</p>{dt}
<p>比较“探针序号与脑电”和“仅探针序号”的按人 AUROC，{'，'.join(paired_rows)}。在这组特征及模型下，时间基线之外的增量仍未得到验证。</p>
<h2>本地采集如何使用这些结果</h2><p>保留独立随机探针、原始回答和现有实时质控。在本地验证时可锁定 16 秒差分 6 维及逻辑回归、RBF 两种候选，在后台同步记录分数与拒评原因。
同时记录任务经过时间和探针序号，在同一批新受试者、同一批探针上比较时间、脑电及两者合并。按人 AUROC、实际覆盖率及眼动代理分层应同时报告。</p>
<p>公开健康对照共有 272 次专注和 148 次走神，走神包括 95 次自发和 53 次有意走神。本地问卷的标签定义和任务时序有所不同。
公开文件内 Fp1、Fp2 为 512 Hz，EEGReference 字段为 n/a。本地设备诊断为两通道、标称 250 Hz。重采样复核只模拟采样率；实际电极位置、参考、前端滤波、接触和现场干扰需要用本地数据确认。</p>
<h2>结果与复核</h2><p>95% 百分位区间基于已生成的折外分数，对 28 名受试者整人重采样 2,000 次。各组合的区间未按多重比较校正。
每折只用训练人员估计均值、尺度和权重。250 Hz 使用录制起点对齐的 129 点单侧抗混叠 FIR，截止频率 102 Hz，名义延迟 125 ms。先前版本的 10 秒严格窗准入在 420 条记录上全部复现，合格探针的 6 维特征最大绝对差约 1.31 × 10<sup>−13</sup>。</p>
<p><a href="comparison_grid.csv">完整 162 行结果 CSV</a> · <a href="evaluation/report.json">512 Hz 评估</a> · <a href="evaluation250/report.json">250 Hz 评估</a> · <a href="artifact_time_diagnostic.json">事后对照</a> · <a href="paired_candidate_comparisons.json">候选配对差</a> · <a href="evaluation/historical_raw_common.json">旧版共同探针对比</a> · <a href="evaluation250/shadow_comparison_candidates.json">250 Hz 后台复核候选权重</a> · <a href="data_provenance.json">数据溯源</a></p>{''.join(parts)}
<p>数据由 Gabriela Horáková、Sébastien Weibel、Ugo Tissot、Bich-Thuy Pham、Madalina Elena Costache 与 Anne Bonnefond 发布，版本 2.0.1，CC BY 4.0。<a href="https://doi.org/10.5281/zenodo.23592484">ATTLAPSE 数据记录</a></p></main>"""
    css = """*{box-sizing:border-box}body{margin:0;color:#203447;background:#fff;font:16px/1.7 "Microsoft YaHei",sans-serif}
    main,header{max-width:1000px;margin:auto;padding:28px 32px}header{border-bottom:3px solid #246a72}
    h1{font-size:clamp(24px,4vw,38px);margin:8px 0}h2{margin:36px 0 10px;font-size:23px}
    p{margin:10px 0}header p{color:#546b72}.lead{font-size:19px;line-height:1.8}img{width:100%;max-width:840px}
    .overflow{overflow-x:auto}table{border-collapse:collapse;margin:10px 0;font-size:14px;min-width:68%;font-variant-numeric:tabular-nums}
    th,td{padding:7px 10px;border-bottom:1px solid #d6e0df;text-align:right;white-space:nowrap}th{background:#edf4f3}
    th:first-child,td:first-child{text-align:left}details{margin:12px 0}summary{color:#205571;cursor:pointer}
    a{color:#165b76}@media(max-width:600px){main,header{padding:18px 14px}}"""
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ATTLAPSE 双通道脑电比较</title><style>'
    (root / "comparison.html").write_text(page + css + "</style><body>" + body + "</body></html>", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
