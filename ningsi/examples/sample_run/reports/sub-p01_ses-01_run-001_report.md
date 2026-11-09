# 凝思状态评估报告（sub-p01 / ses-01 / run-001）

- 指标口径：indicator-v1；基线口径：baseline-v1；处理链：welch-v1
- 软件版本：0.1.0

## 一、状态评分
- 专注度：0.44（波动 0.43，n=35）
- 放松度：0.56（波动 0.43，n=35）
- 认知负荷：0.49（波动 0.39，n=35）
- 相对基线的频带 z 值：alpha +0.24、beta -0.50、gamma -0.07、low -0.00、theta +0.34

## 二、采集质量
- 4 秒窗 35/35 可用（可用窗比例 100%）

## 三、量表结果
- 焦虑自评量表（SAS）：粗分 45，标准分 56，轻度；反向题 [5, 9, 13, 17, 19]，版本 zung-cn-v1
- 抑郁自评量表（SDS）：粗分 50，标准分 63，中度；反向题 [2, 5, 6, 11, 12, 14, 16, 17, 18, 20]，版本 zung-cn-v1

## 四、行为任务
- PVT-B：69 试次，中位反应时 0.339s，慢反应率 1.5%，抢答 0 次
- SART：180 试次（No-Go 20），正确率 100%，虚报率 0.0%，漏报率 0.0%，反应时 0.320±0.000s，序列集 5

## 五、联合评估结论
- 结论：压力相关指标偏高，建议休息与放松训练；另有维度与其他证据不一致，建议复测
- 专注维度：正常范围（依据 eeg_focus, sart_performance, pvt_alertness）
- 压力维度：部分提示偏离（依据 eeg_load, eeg_relax, sas_standard_score, sds_standard_score）
- 一致性：脑电 vs 量表 不一致；脑电 vs 行为 无冲突

## 六、改善建议
- 在安静环境、固定佩戴位置下复测一次，优先核对伪迹比例与基线质量。
- 安排 5–10 分钟腹式呼吸或闭眼放松，并观察放松度评分是否回升。
- 本结论仅用于研究与自我调节参考，不构成医疗诊断，也不得用于处罚或自动上岗决策。

## 七、证据回填
- [eeg_focus]（eeg/可用）专注度评分 0.44（可用窗比例 100%）；引用 {"indicator_spec": "indicator-v1", "baseline_spec": "baseline-v1"}
- [eeg_load]（eeg/可用）认知负荷评分 0.49；引用 {"indicator_spec": "indicator-v1"}
- [eeg_relax]（eeg/可用）放松度评分 0.56；引用 {"indicator_spec": "indicator-v1"}
- [sas_standard_score]（scales/可用）焦虑自评量表标准分 56（轻度）；引用 {"scale_version": "zung-cn-v1", "reverse_items": [5, 9, 13, 17, 19]}
- [sds_standard_score]（scales/可用）抑郁自评量表标准分 63（中度）；引用 {"scale_version": "zung-cn-v1", "reverse_items": [2, 5, 6, 11, 12, 14, 16, 17, 18, 20]}
- [sart_performance]（behavior/可用）SART 正确率 100%、虚报率 0.0%、漏报率 0.0%、反应时变异 0.00；引用 {"sequence_set_id": 5, "seed": 4157852885, "spec": "joint-assessment-v1"}
- [pvt_alertness]（behavior/可用）PVT-B 中位反应时 0.339s、慢反应率 1.5%；引用 {"task": "pvt-b"}

> 边界说明：本结论仅用于研究与自我调节参考，不构成医疗诊断，也不得用于处罚或自动上岗决策。