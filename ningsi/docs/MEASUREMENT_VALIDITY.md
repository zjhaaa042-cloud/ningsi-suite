# 注意相关指标与测量有效性

当前 β/θ 仅作为专注相关脑电候选特征。`ln(β/θ)` 相对睁眼个人基线标准化，再映射为 0–1 显示分数；分数不是专注概率，也不是通用注意能力量表。算法可复算不代表其识别效度已经验证。

## 当前质量门控

- 分析窗口须完整覆盖 4 秒，空数据、NaN/Inf 与短窗不可用。
- 所有通道分别计算眼电（0.5–4 Hz）和肌电（30–45 Hz）代理比例；任一通道越界则整窗拒绝，保存各通道指标。保留原幅值、平坦、通道跨度和频带阈值。
- 这仍是粗筛，不能排除 β 频带内肌电或所有眼电污染。电极位置、参考方式仍需在真实采集协议中固定；当前通道平均不是额区定位指标。
- `baseline-v2` 保留至少 5 个可用窗，并增加至少 30 秒窗口时间区间并集及至少 60% 可用窗要求。30 秒是新的保守工程质量门槛，待验证，不能替代默认 120 秒睁眼/闭眼采集协议。
- 重叠、重复窗口不重复累计时长。Studio 真实源基线采用单调时钟记录覆盖时间，不能用加速演示的虚拟时间冒充真实采集时长；仍需设备数据时间戳与掉流检测保证输入新鲜度。
- 保存前后半段对数频带比值均值差供稳定性复核；没有证据支持通用稳定性阈值，因此不自动宣称“稳定性已验证”。窗口质量通过也不代表注意状态有效。

## PVT 计分 v2

保留主阈值 `RT > 500 ms` 与联合评估慢反应率 `>10%`，新增 `RT ≥355 ms` 对照统计，不用于替代主判断。刺激前抢答及 `0≤RT<100 ms` 从 RT 分布排除；负数、缺失或非有限的已作答 RT 标为无效数据。正常漏答与抢答分开统计。

- `lapse_rate`：超过 500 ms 的有效反应数 / 有效反应数。
- `lapse_rate_355ms`：达到 355 ms 的有效反应数 / 有效反应数，仅对照。
- `omission_rate`：正常漏答数 /（有效反应数 + 正常漏答数）。抢答不计入这个分母。
- 有效反应不足 5 次、有无效 RT，或 Studio 任务未完成时，行为证据不可用，不进入联合评估。
- 联合评估增加漏答率检查，复用现有 `omission_rate_high=0.10`，避免大量漏答但少量快速反应被解释为正常。该阈值在 PVT 上仍待验证。
- 数值统计为零不代表证据正常；消费方必须先检查 `valid`、`completed`（Studio）与 `unavailable_reasons`。

当前刺激按预生成起始时刻调度，属于定时刺激变体；完整模式 180 秒，短模式 120 秒。尚未验证它与原始 PVT-B 的等效性；355 ms 来源不能替代对浏览器延迟、超时、刺激协议的验证。

## 阈值与验证边界

联合评估版本为 `joint-assessment-v2`，记录 PVT 漏答判定及行为有效性门控的语义变化。SART/PVT 显式标记无效的结果不进入评估；历史版本报告保留原版本，不作新旧阈值效度等价声明。

保留专注偏低 0.40、预警 0.35/20 秒、解除 0.45/10 秒，以及现有 SART/PVT 判断阈值。评分斜率 1.1、标准差下限、质量阈值和时长门槛均为工程参数，状态为 `pending_validation`。结果表述为“触发/未触发当前工程阈值”，不能声称落在经验证常模范围。

下一步须使用真实被试、固定设备/导联、同步行为与走神探测，按被试及会话划分数据，标定并独立评估误报率、漏报率与检测延迟。仿真模型结果只验证链路，不能证明真人识别能力。

## 研究依据

- Putman et al. (2014), EEG theta/beta ratio as a potential biomarker for attentional control…：https://doi.org/10.3758/s13415-013-0238-7
- van Son et al. (2019), Frontal EEG theta/beta ratio during mind wandering episodes：https://doi.org/10.1016/j.biopsycho.2018.11.003
- Can the Spontaneous Electroencephalography Theta/Beta Power Ratio and Alpha Oscillation Measure Individuals’ Attentional Control? (2024)：https://pmc.ncbi.nlm.nih.gov/articles/PMC10968630/
- Whitham et al. (2007), Scalp electrical recording during paralysis：https://doi.org/10.1016/j.clinph.2007.04.027
- Basner et al. (2011), Validity and Sensitivity of a Brief Psychomotor Vigilance Test：https://doi.org/10.1016/j.actaastro.2011.07.015
