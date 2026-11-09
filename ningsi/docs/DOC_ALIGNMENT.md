# 文档口径修订清单（两份交付 docx ↔ 本项目代码）

起因：`A09-凝思-项目概要介绍-最终版.docx` 与 `A09-凝思-项目详细方案-最终版.docx` 中
有四处口径与代码不一致，另有若干承诺原先没有实现。本文件给出**逐条替换建议**，
使"文档—代码—答辩口径"三者一致。

## 一、必须统一的四处口径

| # | 文档原文位置 | 现在的问题 | 建议替换为 | 代码依据 |
|---|---|---|---|---|
| 1 | 概要「三 功能简介」/ 详细方案 8.4.1 基线时长 | 概要写"闭眼 3 分钟、睁眼 2 分钟"，详细方案 8.4.1 另写一套，凝思会话内基线为睁眼、闭眼各 2 分钟 | 已在 v2 文档中统一为：「静息基线：睁眼 2 分钟 + 闭眼 2 分钟，按 4 秒窗、2 秒步长逐窗质检，两段各自独立建基线，任务态以睁眼基线为参照；每段至少 5 个合格窗才建立基线」 | `config.WELCH`、`signal/baseline.py::build_baseline`（min_windows=5） |
| 2 | 详细方案 8.3.1 频谱估计 | 原文写 Welch PSD（4 s 窗 / 2 s 分段 / 50% 重叠 / 汉宁窗）——**这一条现已实现**，但需注明是自研实现 | 保留参数描述，补一句「自研零依赖实现，位于 `signal/spectrum.py`，不依赖 SciPy」 | `signal/spectrum.py::welch_psd`；`tests/test_signal.py::SpectrumTest` |
| 3 | 详细方案 7.2.13 / 8.x 硬件范围 | 原文写"硬件无关，OpenBCI / Muse / Emotiv 均可" | 改为：「通过 LSL 抽象层接入，凡发布标准 EEG 流的设备均可适配；**当前已完成 BSense-R → BioMultiLite → LSL 链路验证**，其他厂商需按其 SDK/驱动确认后逐台验收」 | `acquisition/lsl_source.py`；`docs/REQUIREMENT_COVERAGE.md` 第一节 |
| 4 | 全文产品名与许可 | "凝思"在概要仅出现 1 次，其余用"系统"；`bsense-lsl` 无开源许可证 | 首次出现处统一写「凝思（Ningsi）」；许可表述改为「自研代码 + NumPy 等开源依赖，上游采集工程与其设备厂商的许可另行确认」 | `README.md` 第 5 节 |

## 二、原先"只在文档里"的能力：现在的状态

| 文档承诺 | 修订前 | 现在 | 文档应如何写 |
|---|---|---|---|
| SAS / SDS 量表与联合评估 | 无实现 | **已实现**（题库、反向题、粗分×1.25、分级、JSONL 落盘、与脑电/行为联合判定） | 可保留承诺，并补「量表与脑电不一致时标注建议复测」的实现依据 |
| 专注度 / 放松度 / 认知负荷三指标 | 无实现 | **已实现**（β/θ、α/β、θ/α 相对个体基线 z → 0–1 评分） | 把公式口径改为与代码一致：比值型指标，而非"相对功率之差" |
| 神经反馈训练闭环与自适应 | 无实现 | **已实现**（分段统计、目标 0.05 步长自适应、保持时长 6–20 s、训练前后同口径基线对比） | 补一句「初始目标取自当次基线中位数对应的评分」 |
| 状态热力图 | 无实现 | **已实现**（五档着色 + 低质量缺失色块 + SVG 输出） | 可直接引用，并说明缺失窗不参与指标与计时 |
| 历史记录与周/月趋势 | 无实现 | **已实现**（JSONL 历史 + 周/月聚合 + 跨设备可比性保护） | 补「设备/采样率/通道变化时明确标注不可比」 |
| 状态实时预警 | 只有质量提示 | **已实现**（低专注 <0.35/20 s、高负荷 ≥0.75/30 s、质量不足，伪迹窗停表） | 把阈值、持续与解除条件写进正文（与 `config.ALERTS` 一致） |
| 评估报告（评分/分析/建议/证据） | 无实现 | **已实现**（Markdown + JSON + 证据回填 + 边界声明） | 可把报告章节结构照搬进方案正文 |
| 模型训练与部署 | 无实现 | **部分实现**（逻辑回归基线 + 被试级 6:2:2 划分 + AUC + JSON 保存） | 保留"传统机器学习为主"的定位，深度模型写为下一阶段 |
| 压力与情绪识别 | 无实现 | **未实现**（仅"压力相关维度指标偏高"的提示，不含情绪分类） | 降级为路线图：写"可操作定义与标签采集方案已给出，模型训练属下一阶段" |
| 跨厂商即插即用 | 无实现 | **部分实现**（LSL 抽象层） | 按上面第 3 条改写 |

## 三、建议补充到文档的"证据索引"

在详细方案第 9 章或附录加一张对照表，让评委可以直接核对：

| 文档章节 | 代码位置 | 自动化用例 |
|---|---|---|
| 6.3.1 工作流与处理链 | `src/ningsi/signal/preprocess.py` | `tests/test_preprocess_model.py::PreprocessTest` |
| 6.4 模型训练与部署 | `src/ningsi/models/logistic.py` | `ModelTest`（含留出被试评估与保存/加载） |
| 6.6 状态热力图 | `src/ningsi/monitoring/heatmap.py` | `HeatmapHistoryTest` |
| 6.7 指标公式与量纲 | `src/ningsi/signal/indicators.py`、`config.py` | `tests/test_signal.py::IndicatorTest` |
| 7.2.3 量表填写 | `src/ningsi/scales/` | `tests/test_scales_behavior.py::ScaleTest` |
| 7.2.5 专注力行为任务 | `src/ningsi/behavior/sart.py`、`pvt.py` | `BehaviorTest` |
| 7.2.10 评估报告 | `src/ningsi/assessment/report.py` | `ReportTest` |
| 7.2.11 历史与趋势 | `src/ningsi/monitoring/history.py` | `HeatmapHistoryTest::test_history_aggregation_and_comparability` |
| 7.2.12 / 7.2.17 预警 | `src/ningsi/monitoring/alerts.py` | `AlertTest` |
| 8.6 自适应训练目标 | `src/ningsi/training/neurofeedback.py` | `TrainingTest` |
| 9 系统测试 | `tests/`（52 个用例） | `ningsi self-test` 或 `scripts/verify.ps1` |

## 四、还需要补的交付材料

1. **项目简介 PPT**：建议按概要介绍的七节结构做 10–12 页，直接使用 `reports/*.svg`（热力图、趋势）与界面截图。
2. **演示视频**：`ningsi ui` 录制一次完整会话（质检 → 量表 → 基线 → 行为任务 → 指标/热力图 → 报告 → 训练 → 趋势），
   再用 `ningsi demo` 展示一条命令跑完整链路。
3. **产品使用说明文档**：已完成，见 `docs/PRODUCT_GUIDE.md`。
