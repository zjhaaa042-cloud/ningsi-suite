# 双前额脑电压力与情绪效价候选模型

2026 年 11 月 15 日完成公开真人脑电训练和按人留出测试。两组权重位于 `src/ningsi/models/frontal_affect.json`，由 Studio 研究接口加载。运行时只接受已确认的 Fp1、Fp2 双通道、250 Hz、通过严格质量检查的 4 秒窗。导出的连续分数供研究记录，尚不足以对本地使用者给出个人分类或心理压力预警。

## 情绪效价的标签复核与优化对照

重查发布记录后，在 OpenNeuro 1.0.3 找回了当前 1.0.7 文件树中缺少的事件表。两个版本对应的行为文件和视频波形指针一致。完整刺激编号可独立核对部分受试者的视频条件。第 53、54 人行为表中的刺激顺序与对方公布的顺序吻合，修正分析排除这两人。原模型的训练与得分保存在下文，供历史复算。[早期事件版本](https://github.com/OpenNeuroDatasets/ds005540/tree/1.0.3) · [当前数据](https://openneuro.org/datasets/ds005540/versions/1.0.7)

对双额极候选，实验在相同合格时段加入分导联五频带微分熵、空间差异、个体历史基线和同人高低效价排序训练，并在开发受试者内部按人选型。另用 64 导作空间信息对照。模型的整体预测时点为 30 秒视频结束时。独立受试者实验仍未证实比已部署模型有稳定提升。各预测、权重、受试者清单和置信区间见[情绪专项结果](../../var/state-optimization-20261115/emotion/RESULTS.md)和[三项状态识别汇总](PUBLIC_STATE_OPTIMIZATION.md)。

先按下方历史复算命令生成 `evaluation/trials.json`。然后在引擎源码目录运行以下命令，使用同一个 `$cache`；需 NumPy、SciPy，下载时需联网。两套程序都保持原留出人的身份，已有留出结果不构成新盲测。

```powershell
$out = "D:\ningsi\var\state-optimization-20261115\emotion"
python scripts/fetch_emoeeg_events.py "$out/events"
python scripts/resolve_emoeeg_video_order.py "$out/events" "$cache/emoeeg" "$out"
python scripts/fetch_valence_topography.py "$out/topography"
python scripts/validate_valence_mechanisms.py "$cache" "$out"
```

`corrected_holdout.csv` 的 `previous_deployed` 为已部署三频段，`old` 为修正标签重新训练的同三频段模型，`fp` 和 `all64` 为两个蒙太奇经开发数据选出的新候选。`refit_corrected_three_bands.json` 保存重训的参数和模型指纹，`research_candidates.json` 保存候选参数，`results.json` 提供同试次配对差异。官网发布的 53 份全头微分熵数据已逐份核对 1.0.7 的 SHA-256。

## 已部署权重的历史数据与标签

- **压力**采用 BCI Lab GU、Rajdeep Ghosh 等人的 SAM40 v1，许可为 CC BY 4.0。取 40 名参与者完成算术、镜像识别、Stroop 任务时的原始电极数据及其实际 1–10 分自评，每人 9 次，共 360 次。自评 1–5 分为较低，6–10 分为较高，分界取量尺中点。两组分别 233 和 127 次。未收集自评的静息段排除。训练从 128 Hz Emotiv 32 通道记录中取 Fp1/Fp2。源归档 MD5 为 `318b8b0c4e5a22c6e903534a842ce683`，与发布方一致。[数据](https://doi.org/10.6084/m9.figshare.14562090.v1) · [实验论文](https://doi.org/10.1016/j.dib.2021.107772)
- **情绪**采用 Xin Xu 等人发布的 EmoEEG-MC，OpenNeuro 标记为 CC0。使用作者重排后视频段的 Fp1/Fp2 波形和 `score_8` 效价自评，低于或高于 0–7 分量尺中点 3.5 分别为负、正向。原实验在每对区块中随机决定视频和想象的先后，行为表未提供逐块条件字段。仅使用两个条件内刺激顺序相同的前 53 名完整参与者。对每段视频同时核对两条可能与其对应的自评，两条都在中点同一侧时，才可确定视频自评所属的类别。1,113 次视频中 320 次负向、368 次正向，425 次无法确定或触及中点。该子集主要覆盖两种诱导方式下效价方向一致的情形。原数据已使用多通道 ICA 清理及全头平均参考，本机参考方式和现场伪迹水平可能不同。[数据](https://openneuro.org/datasets/ds005540/versions/1.0.7) · [区块安排与评分字段](https://github.com/ncclab-sustech/EmoEEG-MC/blob/main/Readme.md) · [论文](https://doi.org/10.1038/s41597-025-05349-2)

## 特征和验证

公开源波形经多相重采样转为 250 Hz。沿用产品逐通道去均值、0.5 Hz 高通、50/60 Hz 陷波、45 Hz 低通的处理链，取 6 秒上下文的最后 4 秒进行 Welch 分析和严格窗质控。公开源数值按微伏解释，衍生数组自身没有单位字段。每试次等间距取 5 个时点。特征为双通道平均谱的 θ、α、β 对数相对功率。线性逻辑回归以人和试次加权、训练时两类等权，L2 固定 0.05，分类阈值固定 0.5。

先用固定随机种子 20261115 从所有来源参与者抽出约四分之一作为测试组。其余人用于训练。标准化及权重拟合只使用开发数据，交付权重未用测试组重新拟合。每个试次取可用窗分数的平均。下表置信区间按人重采样 2,000 次，描述公开队列内的不确定性。

| 指标 | 压力 | 情绪效价 |
|---|---:|---:|
| 留出参与者 | 10 | 14 |
| 有可评分试次的人数 | 9 | 14 |
| 全部原始留出试次 | 90 | 294 |
| 二分类自评可确定的试次 | 90 | 172 |
| 同时有明确标签和合格 EEG 的试次 | 51（占原始 56.7%） | 165（占原始 56.1%） |
| 阳性／阴性可评分试次 | 21／30 | 90／75 |
| 合格窗／有标签试次的全部抽样窗 | 105／450 | 588／860 |
| 普通准确率／开发组多数类基线 | 49.0%／58.8% | 52.7%／54.5% |
| 平衡准确率及 95% 区间 | 50.2%［34.3%, 66.3%］ | 53.6%［46.3%, 61.4%］ |
| 合并试次 AUROC | 0.437 | 0.522 |
| 个体内平均 AUROC 及 95% 区间 | 0.571［0.456, 0.713］，5 人有两类 | 0.577［0.504, 0.643］，13 人有两类 |
| 合并单窗 AUROC | 0.423 | 0.500 |

SAM40 同批测试试次中，仅按任务类型给出开发组较高压力标签频率的基线 AUROC 为 0.521。固定任务后脑电 AUROC 分别为算术 0.471、镜像 0.485、Stroop 0.311。情绪模型的开发人群五折个体内 AUROC 为 0.395，留出组为 0.577，方向不稳定。这些结果尚不能支持对个人压力、情绪或短时变化的准确识别。

## 左右分通道与额极 alpha 不对称对照

在原开发人群内做五折按人验证。对照候选增加 Fp1、Fp2 各自的 θ、α、β 对数相对功率，以及右减左的对数 alpha 绝对功率。两种模型使用完全相同的参与者、试次、合格窗口、L2 和阈值，逐窗检查旧三维特征与冻结提取结果一致。

| 开发人群配对比较 | 压力 | 情绪效价 |
|---|---:|---:|
| 有合格数据的开发参与者／试次／窗口 | 27／159／340 | 39／494／1,994 |
| 原模型 → 加入左右信息后的个体内平均 AUROC | 0.647 → 0.614 | 0.395 → 0.381 |
| 按人配对 AUROC 差及 95% 区间 | −0.033［−0.111, 0.035］ | −0.014［−0.060, 0.033］ |
| 平衡准确率变化及 95% 区间 | −1.5 个百分点［−8.5, 5.6］ | −2.5 个百分点［−5.9, 0.7］ |

未观察到改进，产品继续使用三维研究候选。扩展特征的比较到开发人群为止。

## 已部署权重的历史复算

训练脚本需要 NumPy、SciPy 和 xlrd。运行时只需要 NumPy。下载上面的 SAM40 归档并核验 MD5，在缓存根目录解压 `Data/raw_data`、`Data/scales.xls`、`Data/Coordinates.locs`。将 OpenNeuro 发布的 README.md 保存在缓存根目录，训练时所用 README.md 的 SHA-256 为 `379cfc97fce08b4c54d3dc26e95c35d504ca130666bbcf02eff59638f0ff7e3a`。在引擎源码目录运行下面的命令，示例缓存根目录为 `D:\ningsi\var\affect-public`。

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:OPENBLAS_NUM_THREADS = "1"
$cache = "D:\ningsi\var\affect-public"
python scripts/fetch_emoeeg_affect.py "$cache/emoeeg"
python scripts/train_public_affect.py --sam "$cache/Data" --emo "$cache/emoeeg" --readme "$cache/EmoEEG-readme.md" --output "$cache/evaluation" --workers 3
python scripts/compare_frontal_affect.py "$cache"
```

`evaluation` 内的 `trials.json` 保存逐试次标签、特征与拒评计数，`*_heldout.csv` 保存可评分留出试次的真实类别和预测。冻结权重和独立验证指标在 `frontal_affect.json`，配对开发组结果在 `development_comparison.json` 和 `*_development_comparison.csv`。
