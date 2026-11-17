# 成熟工具复用与目标设备验证方案

核对日期：2026-11-17。本文给出开发选型与现场实施方案，未新增真人实验结果，未改变生产滤波、质量门槛或冻结模型。官方文档会更新，真正运行时应另保存安装版本与依赖清单。

## 本轮选型

| 环节 | 采用方式 | 可以解决的事情 | 尚不能证明的事情 |
|---|---|---|---|
| LSL 原始与事件留证 | 厂商源继续供 Capture／Studio 使用；现场并行使用 LabRecorder 保存选定 EEG 与事件流的 XDF | 使用现成录制器保存多流原始样本、时间戳和同步信息，形成独立于本产品的记录 | 厂商流内部每点时标正确、物理显示／按键延迟已经测准 |
| 离线信号审计 | 在独立工具环境中使用 MNE-Python；项目工具 `scripts/audit_field_signal.py` 调用其 Welch PSD，配套依赖在 `scripts/requirements-field-audit.txt` | 核对原始工频占比、采样与时标诊断，比较项目与成熟库的计算行为 | 滤后曲线平滑就等于得到真实干净脑电，或模型已经有效 |
| XDF 读取 | 使用 pyxdf，保留读取选项、流选择与原始／同步后时间戳的区别 | 复用现成 XDF 读取和时钟映射工具 | 将离线时间平滑结果直接当作在线真实时标 |
| 传统模型比较 | scikit-learn 的 Pipeline、按参与者分组的交叉验证及 LR／SVM／随机森林作为离线对照 | 复用标准训练与模型选择组件，减少手工标准化和划分错误 | 依靠更换分类器解决无合格数据、无两类标签或目标设备域偏移 |
| 实时接口扩展 | MNE-LSL 先用于回放与对照；当前 pylsl／BioMulti 路径继续作为生产输入 | 使用现成实时缓冲与回放能力，比较窗口、事件与丢流行为 | 新库会自动保留既有模型的处理与时标契约 |
| 反馈训练验证 | 采用 CRED-nf 的实验设计与报告结构 | 明确对照、盲法、反馈学习和行为迁移证据 | 一个训练前后分数差即可证明强化训练有效 |

LabRecorder 是 LSL 官方录制项目，支持把选定流录入 XDF；LSL 文档说明 XDF 可保留原始时间戳及测得的时钟偏移，便于以后重新选择映射。这里将它用于并行留证，不替代 Capture 已有任务事件与研究导出。[LabRecorder 官方项目](https://github.com/labstreaminglayer/App-LabRecorder)、[LSL 时间同步说明](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html)。

Capture 0.4.1 已新增默认关闭的可选 LSL Marker 出口：在「外部同步」显式勾选，或通过 `gui --lsl-markers` 预选。出口使用原事件 `stamp.lsl`，仅允许 LSL 本地时钟；事件固定白名单不包含答案、分数、参与者或完整载荷。已有合成回环与异常隔离测试，不代表实际 LabRecorder/XDF 多流录制已验收。任务事件仍完整保存于本机数据库及导出，使用者需另选择 EEG／Marker 流、录制并打开 XDF 核查。[Capture 操作说明](../ningsi-capture/README.md)。

pyxdf 提供已有的 XDF 读取实现，应记录实际使用的读取参数与版本；不要自行另写 XDF 解析器。[pyxdf 官方项目](https://github.com/xdf-modules/pyxdf)。

## 先处理当前实采问题

本地历史证据中，完整记录复算为 573 窗中 3 窗严格合格、有效基线为否、正式探针配对为 0；后续约 32 秒片段中，原始 50 Hz 附近功率占比中位数约 95%，15 窗中 0 窗合格。它们是具体历史记录，不能外推成所有设备或所有佩戴情况的表现。出处为本机 `var/preprocessing-0.4.0/P001-export-check.json` 与 `live-review.json`，原始记录不进入源码或发行包。

**工程判断：应先排查采集与佩戴链路。** 50 Hz 占比只说明该频段在本次分母内占主导，不直接定位是接触、参考／地、环境耦合、单位、设备设置还是其他原因。按每次只改变一项的方式记录：原佩戴 → 重新确认参考与接触 → 调整线缆和周围供电设备位置 → 按厂商允许的安全供电方式复测。每一步均保存相同长度的原始片段和配置，比较原始 PSD、幅值、饱和、短时跳变和有效覆盖。

不通过提高幅值阈值、隐去工频、改变单位倍率或删除困难片段来增加通过率。陷波后的 50 Hz 降低可以是滤波器预期行为，不能单独证明信号恢复。真实静息、自然任务与标记眨眼／咬牙／转头片段分别保留，才能检查漏检和误拒。

具体现场顺序与记录字段见 [现场验收清单](FIELD_ACCEPTANCE_CHECKLIST.txt)。

用户补充的 BSense-R 手册已逐页核对，规格与实际配置的适用边界见 [设备接入说明](BSENSE_R_DEVICE_GUIDE.md)。特别注意厂商显示滤波不改变存储导出的原始数据、蓝牙笑脸不表示电极接触、CSV 的 µV/FP1/FP2 不等于 LSL 单位/通道索引。手册还提示 LSL 开启后不会自动本地保存，应确认实际录制器已生成文件。

## 时间同步：复用工具，但保留物理边界

本项目 BioMulti 流有分包发布语义。XDF／LSL 的时钟同步处理针对已经发布的时间戳与时钟偏移；它不能凭空还原厂商未提供的设备采样时刻。LSL 官方也区分在线最新偏移校正、规则流去抖和离线段内拟合，说明没有额外桥接时不能普遍同步非 LSL 时钟。[LSL 时间同步说明](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html)。

本项目据此采用以下约束：

1. 原始源时间戳、接收时刻、所选分析时标与调整量分别保留。
2. LabRecorder 与本产品同时记录时，按明确流名／source_id 选择 EEG 和事件；模拟流保留模拟标记。
3. `proc_dejitter`、离线规则化或 MNE-LSL 的处理选项不得直接开启到现有冻结模型路径。先复算同一记录，检查间断、重置与事件对齐，再形成新的处理版本。
4. BioMulti 原生 REC 双录仅在当前厂商版本支持并确已生成对应文件时用于核对设备样点与分包；原生设备时钟和普通在线 LSL 的适用范围继续分开，不默认推流已同时留存 REC。
5. 事件发布时刻只作为软件时标。要报告物理刺激到显示、按键到回传的端到端延迟，应另用同步触发或光电等测量方案。未测量时保留“物理延迟未验证”。

MNE-LSL 有 `StreamLSL`、`EpochsStream` 和 `PlayerLSL`，可作为现成缓冲、事件取窗与回放对照。回放能验证代码行为，不能复现真实蓝牙／USB／设备时钟的全部延迟。[MNE-LSL 官方介绍与回放教程](https://mne.tools/mne-lsl/stable/generated/tutorials/00_introduction.html)。

## MNE 离线审计的接入与产物

独立工具环境避免扩大轻量生产程序的依赖。工作区可使用：

```powershell
python -m venv var/tool-envs/field-audit
var/tool-envs/field-audit/Scripts/python.exe -m pip install -r scripts/requirements-field-audit.txt
var/tool-envs/field-audit/Scripts/python.exe scripts/audit_field_signal.py --help
var/tool-envs/field-audit/Scripts/python.exe scripts/audit_field_signal.py --export "D:/实验/完整采集导出" --output "D:/实验/独立审计_新目录"
```

具体输入和输出参数以该工具 `--help` 为准。运行后应保存：工具版本与依赖、输入清单哈希、选中通道与采样率、单位来源、工频功率分母与频段、时标异常、已有质量统计及未完成项。对缺失单位、非有限值、长度冲突和不可确认输入应给出失败或待确认，不能猜测倍率。

只需要一段新的现场信号作诊断时，可先使用下面的短时采集工具。它复用 Capture 的精确发现、原始记录、LSL 接收与导出，不改厂商程序；默认不发布 Marker，仅创建 `pilot`、`signal-check-only-v1` 记录，没有任务、答案或研究协议完成标记，不进入训练／测试。单位只使用流声明或已有本机软件核对依据，缺失时保持 `raw_only`；电极与参考不猜测。

```powershell
.venv/Scripts/python.exe scripts/capture_field_signal.py --device "BioMulti Lite EEG-00a659" --seconds 30 --data "var/field-signal-new" --export "var/field-signal-export-new"
```

`--data` 和 `--export` 都须使用新目录。收到首批原始样本后才开始计时；缺源、同名冲突、采集停止、无原始数据或无窗口历史均应失败并保留已写记录。记录完成仍不证明物理佩戴、模型有效或信号质量已通过。

使用 MNE 构建 EEG `RawArray` 时要遵守其 SI 单位语义：项目内部 µV 数组进入 MNE EEG 时须显式换算到 V；如果直接对 µV 数组调用 PSD 数值函数，报告的单位则对应 µV²/Hz，不能标成 V²/Hz。MNE 的数据接口明确区分默认 SI 单位与显式请求的 µV 返回单位。[MNE RawArray／数据单位说明](https://mne.tools/stable/generated/mne.io.RawArray.html)。

这份独立审计不自动重写导出、基线、掩码、模型或训练资格。比较时显式对齐 Welch 的采样率、窗口、段长、重叠、去均值、频率范围和积分方法；参数不同时结果差异不直接视为生产实现错误。MNE 处理参数与生产参数有差异的试验应另存结果。[MNE Welch PSD 官方接口](https://mne.tools/stable/generated/mne.time_frequency.psd_array_welch.html)。

## 双前额输入下的伪迹方案选择

| 方法 | 官方工具的能力 | 本项目当前选择 | 进入下一阶段的条件 |
|---|---|---|---|
| 质量门控与污染标记 | MNE 注释／Epochs 可保留并排除坏片段；项目已有幅值、频谱、时标与伪迹证据 | 立即保留，增加现场人工动作标记作对照 | 报告每类污染的误拒与漏检、自然任务覆盖和失败原因 |
| EOG 参考校正／ICA 成分核对 | MNE 支持专门 EOG／ECG 参考及 ICA 诊断 | 有真实同步 EOG 时才做参考对照；不把 Fp1／Fp2 当作已确认 EOG | 参考类型、单位和同步明确，并验证神经频带保留；新版本重新训练／匹配模型 |
| autoreject | 学习峰峰值阈值，按坏传感器情况拒绝或插值修复 epoch | 可做离线阈值敏感性对照；不在双通道生产路径自动插值 | 有充分独立校准 epoch、可用空间邻居和正确电极位置；只在开发数据选择参数 |
| ASR／clean_rawdata | 对相对干净校准数据的协方差／子空间变化识别异常段 | 暂不作为双前额低质量记录的自动恢复开关 | 更充分空间信息、可验证的干净校准段及保留神经信号的独立证据 |

MNE 文档明确提示：用前额 EEG 构造虚拟 EOG 可能同时移除前额脑活动。当前只有两条前额 EEG，差分后空间秩还会更低；因此不据“能运行 ICA”推断能可靠分离脑源与眼源。这是基于输入维度和本项目目标的适用性判断。[MNE ICA 与虚拟 EOG 的风险说明](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html)。

autoreject 的局部修复依赖坏传感器比例与插值策略，并由数据估计参数；不能照搬多通道示例的插值数到两条前额导联。当前优先只将其作为离线比较工具。[autoreject 官方修复示例](https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html)。

EEGLAB 的 ASR 流程以校准数据及多通道子空间比较识别异常；强噪声覆盖几乎整段时，校准本身可能不可靠。官方建议查看清理前后差异；在本项目采用 ASR 前还须核对具体实现版本和使用许可。此处没有将其加入生产依赖。[EEGLAB clean_rawdata 官方说明](https://eeglab.org/plugins/clean_rawdata/)。

## 分类模型复用与独立验证

已有 NumPy 逻辑回归和冻结权重继续提供可追溯的历史行为。新对照可复用 `Pipeline(StandardScaler(), LogisticRegression/SVC)` 与 `RandomForestClassifier`，用 `StratifiedGroupKFold` 或适当的 `GroupKFold` 按参与者分组。`StratifiedGroupKFold` 尽量保留类别比例且参与者组不重叠，但组数／类别分布不足时不保证每折都能满足同样比例；应先检查每折人数与两类标签覆盖。[scikit-learn 分组分层交叉验证](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)。

所有需要拟合的标准化、特征选择、超参数、概率校准和阈值选择都放在开发数据内。Pipeline 可以帮助防止折外预处理泄漏，但不能阻止错误分组或滤波历史跨界；这些边界仍需项目显式检查。[scikit-learn 数据泄漏与 Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)。

固定比较表：

| 模型臂 | 可用输入 | 主要问题 |
|---|---|---|
| B0 | 固定相对个人基线指数 | 现有工程指标在独立人群上能解释多少变化 |
| B1 | 固定六项 EEG 特征，LR／SVM／随机森林 | 现有特征下采用成熟学习器是否有可重复增益 |
| C0 | 同探针的时间、任务序号、行为及允许使用的个人历史 | 无 EEG 也能达到什么程度 |
| C1 | C0 + 与其严格配对的合格 EEG | EEG 是否提供额外信息，而不是只增加一个复杂模型 |

压力、情绪效价、专注分别使用各自的标签定义，不拼成互斥三分类。报告按人 AUROC／平衡准确率、覆盖率、参与者区间和预测明细；单类标签时标明不可计算。跨天复测和新参与者测试分别报告，已看过的测试者不再当下一轮未见测试者。EEGNet 等深度基线待合格样本与固定对照齐备后再比较。

## 反馈训练效果：采用 CRED-nf 的结构

训练实施前固定主要行为结局、对照条件、随机分配、盲法范围、排除规则、样本量依据与随访安排。报告实际反馈信号、目标／阈值、被暂停的时长和神经调节学习，再独立报告行为变化；两类结果不能互相替代。[CRED-nf 原始共识论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7296848/)。

本项目可执行的首轮方案是比较“实时反馈”与时长、界面接触和任务练习匹配的固定引导；若采用假反馈／配对回放，需在实施前明确研究告知、盲法和实施条件。前后行为任务考虑练习与疲劳，所有比较预先固定。先做流程试采，再开展足够人数的正式比较；完成前只报告训练过程和描述性前后变化，不声明疗效。

## BrainFlow 的采用边界

BrainFlow 为其支持清单内的设备提供统一采集接口，另有 Synthetic／Playback 模式可用于开发。核对日的官方支持清单中未检出 BioMulti 名称；不能据“支持多种脑电设备”推断它已经支持当前 BioMulti。当前保留厂商 LSL／原生 REC，未来增加明确受支持的设备时再做适配。[BrainFlow 官方支持清单](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html)。

新增适配至少要复核设备型号、通道索引／名称、物理单位、原始／滤后数据定义、参考、采样率、数据包与时间戳。更换驱动、设备或参考后，不直接复用旧个人模型；按模型包所保存契约重新确认或重新训练。

## 完成状态的判定

- **本轮可完成：** 正式设备档案、开始前模型适配提示、独立审计工具、源码／打包验收、采集与实验记录模板。
- **拿到现场数据后完成：** 电极接触与参考确认、独立留证、时标核对、合格基线和正式任务配对、掉线恢复与重戴检查。
- **独立研究后才能确认：** 三类目标状态的泛化、EEG 的增量、训练的可重复效果。

采用第三方库时，版本、许可证、参数和输入单位随产物记录；安装成熟库不自动改变以上完成状态。

## 局部伪迹与有效时间的采用边界

新增只读工具 `scripts/audit_artifact_segments.py` 检查 Capture 导出的窗口掩码、单位、采样点、校验值和重叠一致性。掩码是通道联合结果，不能反推出某个单独通道干净；被不同重叠窗口标记的坏点按并集排除，并按当前有限滤波支撑区增加保护。短片段不拼接成完整窗口，也不提升为生产评分、标签或训练资格。

三段既有现场数据在保守保护下未提供额外完整 4 秒窗口。当前产品因此采用“质量暂停、有限补采、覆盖与表现分列”，而没有默认启用 ICA、ASR 或插补。后续采用成熟纠伪方法仍需同设备的带参考采集、固定参数和独立验证，不能用软件可运行替代恢复效果证据。
