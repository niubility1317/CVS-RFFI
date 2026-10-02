# CVS 固定身份网络：mid／low urban 拼接增强

状态：PLANNED，尚未启动或产生本次测试结果。网络固定为 `residual_fusion`，164225 参数。按用户“具体性能看测试集不是源域”的补充，使用既有clean测试较优的residual_fusion（四seed均值78.4543%）。此为BENCHMARK_INFORMED_FIXED_DESIGN，历史测试表现已影响用户指定结构，不宣称目标无接触的结构筛选或新的盲测。具体性能以本次冻结权重后的测试集准确率和 Macro-F1 判断，源域指标仅记录训练情况，不用目标测试进行后续调参、换权重或选择性重跑。

本次仅训练 `practical_mid`／`practical_low_urban`。E1 至 79 只做 clean CE；E80 至 90 在两场景之间均匀抽样，扰动概率 0.60；E91 至 200 仍只抽取两场景，概率 0.80。同一个 physical ID／epoch 对应固定随机增强，augmentation_seed 与 receiver_seed 都为 2027。high 的训练配置已移除。

每个 clean batch 生成同尺寸 satellite batch（未抽中扰动的样本保留），拼接成一个 batch 前向，再优化 clean CE＋0.68×satellite CE。拼接不增加 optimizer step 数；E80 后满批前向尺寸为 256，末批为 56。无一致性损失、域骨干、DAOT、RC4、PL、MixStyle、EMA 或其他辅助目标。

固定原物理划分：L／U／V=6300／56700／27000，U 不参与训练，source RX1、3、4、6、8，day1、2、3；ManySig 六类、equalized1、中心256点、单位RMS、25MHz。4个模型种子2026092701至2026092704，从零初始化，无历史权重、optimizer、teacher、EMA或原型继承。历史记录只提供设计与已固定对照证据。

AdamW，lr=0.0002，wd=0.0001，cosine至0.000001，batch128且保留末批。每轮50步、200轮，单模型10000步。完整FP32，cuDNN和matmul TF32关闭，无梯度裁剪；模型与loader RNG明确记录。本次相比历史practical基准还存在结构、显式loader及backend口径差异，不将差值单独归因于增强场景变化。

每行 E200 最终权重保存且源冻结标记核实后，由独立预测进程读取既有 VALIDATED_ONCE capsule；先固定全部4行预测，独立 scorer 再连接 truth。测试保留 clean 和3个既有 practical 场景（high、mid、low urban），即168000条clean及相同物理ID的一个satellite观测，卫星场景为互不重叠子集。high 是未用于训练增强的场景，不新建数据、不更换物理ID。它们不是每个样本各生成三个完整LEO视图。

报告4seed均值±样本标准差、每seed和RX的accuracy／Macro-F1及完整混淆矩阵，固定历史CVS-CE／CVCNN-CE作为描述性配对对照；负结果保留。既有benchmark已有公开结果，不称新盲测。无Phase2、SFT或新类注册。

保留完整文本日志、step JSONL、epoch JSONL／CSV，显示实际参数、clean／sat CE及权重、梯度及使用参数、LR、实际增强状态、源V／最差源RX、耗时与峰值显存；另在可丢弃clone上测资源，clone不回流训练。未启用项false，未测项null／N/A。

唯一launch owner，独占新run／输出，每GPU最多2训练任务且至少12GB空闲。不停止、重启或热改其他健康任务。失败保留产物，无性能停机或自动重试。初始checkpoint往返无query smoke通过后直接继续。

[逐行配置与测试预登记](experiment.json) · [只读实时preflight](evidence/preflight.json) · [聚焦验证](evidence/local_validation.json)。正式启动和完成状态以独立PID／CWD／argv／GPU／日志及artifact读回补充。


## 启动读回：VERIFIED

发布commit `3fb08ffb148bff03309dfeb167d013bae778033b`，dispatcher PID 1520025。4个worker在GPU0至3持续运行，源训练均达到E12，实际164225参数、50步/轮、target_access=false；checkpoint从零往返smoke已通过、物理角色EXACT_MATCH，日志持续增长，无failure。E80前satellite_ce=null／augmentation_active=false符合预登记。

| 模型seed | GPU | 源PID | 读回epoch |
|---:|---:|---:|---:|
| 2026092701 | 0 | 1520050 | 12 |
| 2026092702 | 1 | 1520060 | 12 |
| 2026092703 | 2 | 1520071 | 12 |
| 2026092704 | 3 | 1520082 | 12 |

[独立启动证据](evidence/launch_readback.json)。训练、预测和评分尚未完成，不能据源域指标判断测试性能。唯一dispatcher将继续E200→冻结→4行clean/satellite预测→独立truth-last评分。


## 纯 clean 基准已完成，无需补跑

用户询问是否已有无星地增强训练、clean测试的residual_fusion：已完成。原run `20261001-phase1-cvs-residual-identity-manysig-m8-r01` 四行配置全部 augmentation=false、domain_backbone=false、extra_losses=[]、E200；冻结clean测试run `20261001-phase1-cvs-selected-clean-manysig-m20-r01`，每seed168000包。4个总体混淆矩阵独立重算：accuracy=78.4543%±0.8436%，Macro-F1=78.2067%±0.9517%。保留既有产物，不重复训练或测试。

[基准配置与独立重算证据](evidence/pure_clean_baseline_reference.json) · [原报告](../20261001-phase1-cvs-selected-clean-manysig-m20-r01/report.md)。后续本run完成后报告与该基准的clean配对差值；历史无增强实验没有对应星地目标测试，本次不追溯追加其卫星测试。新run显式backend／loader与旧run可能不同，差值不单独证明增强场景因果贡献。


## 最终测试：VERIFIED

4/4 从零训练完成 E200，固定末轮权重完成 clean／satellite 预测，全部4行固定后独立 truth-last 评分完成160条指标记录。全40000步、800轮文本／JSONL／CSV一致性与预算／增强状态核验通过；160个总体和RX混淆矩阵的accuracy／Macro-F1独立重算通过，RX合并、三场景合并与总体完全一致。

以下为四个固定seed的均值±样本标准差，单位为%。**具体识别性能由测试集判断；源V不作为最终性能结论。**

| 测试场景 | 本次残差CVS Accuracy | 本次 Macro-F1 | 历史原生 CVS-CE Accuracy | 历史 CVCNN-CE Accuracy |
|---|---:|---:|---:|---:|
| Clean | 79.05 ± 0.81 | 78.61 ± 0.95 | 76.62 ± 1.54 | 74.27 ± 2.02 |
| 星地总体 | 65.68 ± 0.79 | 65.28 ± 0.85 | 64.15 ± 0.98 | 62.55 ± 1.10 |
| High | 75.46 ± 1.04 | 75.04 ± 1.13 | 73.76 ± 1.38 | 72.96 ± 1.35 |
| Mid | 72.94 ± 0.93 | 72.40 ± 0.99 | 71.49 ± 1.10 | 70.32 ± 1.32 |
| Low urban | 48.57 ± 0.40 | 48.06 ± 0.47 | 47.11 ± 0.51 | 44.29 ± 0.70 |

纯clean训练残差CVS基准为 **78.4543% ± 0.8436%**（Macro-F1 78.2067% ± 0.9517%）。本次相对其clean准确率配对差值为 **+0.5914 ± 0.8243个百分点**，2/4种子为正。该基准已完成，无需补跑；它没有历史卫星测试，本次不追溯追加。

| 模型seed | Clean | 星地总体 | High | Mid | Low urban |
|---:|---:|---:|---:|---:|---:|
| 2026092701 | 80.08% | 66.62% | 76.70% | 73.99% | 49.09% |
| 2026092702 | 78.55% | 65.85% | 75.65% | 73.24% | 48.58% |
| 2026092703 | 79.30% | 64.70% | 74.16% | 71.76% | 48.12% |
| 2026092704 | 78.26% | 65.56% | 75.34% | 72.79% | 48.48% |

| 接收机物理ID | Clean | 星地总体 | High | Mid | Low urban |
|---|---:|---:|---:|---:|---:|
| 1-1 | 74.19 ± 3.48 | 58.89 ± 2.43 | 66.87 ± 3.13 | 64.99 ± 2.85 | 44.69 ± 1.41 |
| 14-7 | 72.11 ± 3.76 | 56.33 ± 3.12 | 63.68 ± 4.41 | 62.30 ± 3.60 | 43.10 ± 1.39 |
| 2-1 | 86.89 ± 1.05 | 71.88 ± 1.59 | 82.58 ± 2.25 | 80.21 ± 1.92 | 52.46 ± 0.64 |
| 20-1 | 76.81 ± 3.17 | 66.06 ± 0.41 | 76.90 ± 0.96 | 73.27 ± 0.52 | 48.02 ± 0.36 |
| 7-14 | 87.43 ± 0.70 | 72.67 ± 1.95 | 84.32 ± 2.62 | 80.04 ± 2.40 | 53.67 ± 0.84 |
| 7-7 | 78.26 ± 1.69 | 67.97 ± 0.69 | 78.77 ± 0.93 | 76.27 ± 0.91 | 48.72 ± 0.31 |
| 8-8 | 77.63 ± 2.37 | 65.97 ± 2.53 | 75.06 ± 3.00 | 73.43 ± 3.13 | 49.36 ± 1.49 |

各RX的Macro-F1、完整混淆矩阵及逐seed配对差值均保存在附件，不只报告总体。训练只用mid／low urban，High是本次未用于训练增强的场景；卫星总体按样本合并，并非三场景指标的简单平均。

| 实测成本 | 四seed均值 |
|---|---:|
| 总参数 | 164225.000 |
| CE实际使用参数 | 164225.000 |
| Conv／Linear MAC/样本 | 9708836.000 |
| 模型常驻字节 | 657552.000 |
| 合成训练batch128 ms | 26.324 |
| 推理batch1 ms | 4.720 |
| 推理batch128 ms | 4.480 |

硬件RTX3090、Torch2.1.0+cu121、完整FP32；副本resource profile不回流模型。MAC不包含FFT／归一化／物理特征等，实测训练batch128是可丢弃副本上的合成普通CE；正式E80后拼接batch256及信道生成成本由完整源日志耗时记录，不能把副本单batch成本当作正式增强训练成本。星载设备／新增传输字节N/A。

本次是用户指定的既有测试较优结构＋固定两场景增强实验，历史benchmark已曝光；不是新的盲测或目标无接触的架构筛选。本轮测试不回流参数、结构、权重或种子选择，全部种子及负结果保留。历史CVS／CVCNN使用三场景增强，网络、显式loader和backend口径还存在差异；无增强残差基准也存在backend／loader口径差异。因此以上差值评价整套固定配置，不单独证明“两场景增强”或残差头的因果收益。无Phase2，适应／新类／H为N/A。

[完整训练与产物审计](evidence/final/completion_audit.json) · [完整测试混淆矩阵](evidence/final/test_scores.json) · [所有分层指标CSV](evidence/final/test_metrics.csv) · [逐种子配对比较](evidence/final/paired_comparisons.csv)。


完成状态：ANALYZED／VERIFIED。完整40000步、800个epoch与160个原定测试记录核实，4个训练及预测进程均自然完成。后续用户授权的两组固定权重、clean＋完整六环境新测试见[六环境报告](../20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/report.md)，其环境样本数及realization与本报告原三子集不同。
