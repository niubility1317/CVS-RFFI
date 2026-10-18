# P67：提高 U 时间确认覆盖，没有形成新增 urban 收益

**两行 E30、clean＋全部六种固定 practical 的预测、独立 truth-last 评分和完整分析均已完成，无技术缺项。固定方案按先行源域判据关闭，不延长 E50。**

单个 seed2026092701 中，AS_BANK 相对匹配 NOAS_BANK 的 clean 为＋0.041071 个百分点，满足 clean 下降不超过0.1个绝对百分点的阶段要求；相对原始 NoAs30 仅＋0.007143pp。该微小正差不构成可重复双提升证据。

新增时间规则相对原 As30 的 urban 三场景均值仅＋0.012103pp，mid＋0.023214pp，low−0.017262pp，同时 clean−0.086310pp。完整 As 的既有大幅收益不能归到时间规则。正式 low urban=51.226786%，未达60%；本轮没有证据证明缓解了某一类低质量 urban 退化。

## 已检验的唯一假设与实际基线

本轮检验：把原 U 时间确认从同一 batch 内不同样本的近邻预测一致，改为同一物理 ID 在相邻 epoch 的预测一致，能否在保持完整 practical 时提高识别。它是已有 epoch_bank 的当前 F／As 用途，不是新算法或 IQ 泛化增强 A_g。旧 reference E200／无 As 用途如实保留。PCGrad、As KL-only-off、正负噪声配对等已有实际训练，不换名重开；另一对话的 L ordinary→raw 未改动。

起点 fc7f941b6721f34ad77ae063dfaeedd98422e8a5；实际训练代码 71b3c3e340005d93433c4d4b3f51d26f4d464d60，覆盖后处理 296c9242fefb387e3d1289e733f1415ded289e29。F＋CROSS_RX 保持 SIG_TIME_X3／统一 Sinc／CENTER_UNITS／joint160／默认 cosine30，900722参数、206状态键。结构固定，参数按原流程联合更新；原头最终直接预测，没有替换、冻结后重训或另拟合诊断分类器。

完整 L6300／hidden U56700／V27000、RX1/3/4/6/8、day1/2/3、六TX、split392005。无 held RX 或 L_dev，U TX 真值始终隐藏。对应 P43 B00/B01 的 source-only scratch20／final_only／4440 更新 student＋ownEMA 是唯一父；实际角色 EXACT_MATCH、无未知祖先、无目标选模。两条新学生各追加10轮，累计 E30／6660；AdamW、cos10、缓存按原段重置。

真实 native args 与各自原 E30 只有 run_id、output_dir、pseudo_temporal_mode 三项差异；唯一训练项为 batch_neighbor→epoch_bank。min_streak2、min_conf.8、原 confidence/domain/strong agreement、raw EMA teacher／strong student .015、U CE／熵及归一化均保持。物理键 (rx_i,day_i,eq_i,sig_i,base_index)，本地首轮无确认，同轮重复不增长。不是新增门控或利用 U 标签。

原六次 L 前向768／step、总 L 1.25、RDrop、ordinary128 跨 RX SupCon .1、全部 fullAs64 标签 CE 保持，无新增神经前向或教师训练。IQ为2×256／25MHz，原逐包 RMS／center=false 在 practical 前，内部 CENTER_UNITS 不变。实际 leo_practical __file__ 为 pinned20261015_r02/experiments/adv3b02_xuc/code/leo_practical，非顶层 code。

As保持 low suburban／mid urban／low urban 各1/3；train fc2.462GHz，完整 residual/post_sync、EQ=false 且 G=1，quality adaptation／phase tracking／IQ compensation／AGC 开启、reference_snr_db30／quality reference20／reflect。硬件按RX/day/eq稳定，非TX定义，信道跨epoch/step刷新。正式固定 received IQ／fc2.45GHz／全部六 practical 未改变。新增噪声比不称总SNR；没有弱化信道、删除困难包或反演未知 TX 硬件。

## 全部正式结果

四条件均为同 seed、同最终学生默认头、每视图168000包。NOAS30／AS30 为原 P43 已冻结预测及评分，未重新推理或重启 scorer。新两行分别 own20＋10，不能当成四次新 scratch30 训练。

准确率（%）：

| 场景 | NoAs30 | 原 As30 | NoAs bank | As bank |
|---|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.213690 | 85.254762 |
| practical_high | 42.685119 | 81.405357 | 42.667857 | 81.417857 |
| practical_mid | 40.995238 | 78.661310 | 40.996429 | 78.695238 |
| practical_low_suburban | 39.472619 | 75.861905 | 39.520833 | 75.901190 |
| practical_high_urban | 41.439881 | 77.767262 | 41.376190 | 77.797619 |
| practical_mid_urban | 35.132738 | 62.806548 | 35.090476 | 62.829762 |
| practical_low_urban | 30.139881 | 51.244048 | 30.172024 | 51.226786 |

Macro-F1（%）：

| 场景 | NoAs30 | 原 As30 | NoAs bank | As bank |
|---|---:|---:|---:|---:|
| clean | 84.979636 | 85.044474 | 84.940917 | 84.954476 |
| practical_high | 39.311416 | 81.084230 | 39.287914 | 81.104903 |
| practical_mid | 37.954813 | 78.359706 | 37.954633 | 78.398297 |
| practical_low_suburban | 36.758731 | 75.572741 | 36.786837 | 75.615817 |
| practical_high_urban | 38.481030 | 77.423208 | 38.402216 | 77.459846 |
| practical_mid_urban | 33.446603 | 62.450463 | 33.396415 | 62.479265 |
| practical_low_urban | 29.184589 | 50.887035 | 29.205580 | 50.874327 |

最差 RX 准确率（%，括号为实际 RX 标识）：

| 场景 | NoAs30 | 原 As30 | NoAs bank | As bank |
|---|---:|---:|---:|---:|
| clean | 78.450000 (14-7) | 78.362500 (1-1) | 78.383333 (14-7) | 78.404167 (1-1) |
| practical_high | 34.304167 (14-7) | 71.908333 (14-7) | 34.016667 (14-7) | 71.804167 (14-7) |
| practical_mid | 33.520833 (14-7) | 70.212500 (14-7) | 33.412500 (14-7) | 70.191667 (14-7) |
| practical_low_suburban | 32.054167 (14-7) | 67.741667 (14-7) | 31.916667 (14-7) | 67.700000 (14-7) |
| practical_high_urban | 34.000000 (14-7) | 69.108333 (14-7) | 33.712500 (14-7) | 68.887500 (14-7) |
| practical_mid_urban | 29.495833 (14-7) | 56.458333 (14-7) | 29.212500 (14-7) | 56.387500 (14-7) |
| practical_low_urban | 25.750000 (14-7) | 46.379167 (14-7) | 25.725000 (14-7) | 46.337500 (14-7) |

| 比较 | clean Δpp | mid urban Δpp | low urban Δpp | urban 均值 Δpp |
|---|---:|---:|---:|---:|
| AS30−NOAS30 | 0.093452 | 27.673810 | 21.104167 | 28.368452 |
| AS_BANK−NOAS_BANK | 0.041071 | 27.739286 | 21.054762 | 28.405159 |
| AS_BANK−AS30 | -0.086310 | 0.023214 | -0.017262 | 0.012103 |
| AS_BANK−NOAS30 | 0.007143 | 27.697024 | 21.086905 | 28.380556 |
| NOAS_BANK−NOAS30 | -0.033929 | -0.042262 | 0.032143 | -0.024603 |

AS_BANK 的 urban 均值63.951389%，原As30为63.939286%。三个 suburban 与自身 clean 差分别3.836905／6.559524／9.353571pp，均未满足“各不超过2pp”的探索定义。不能靠降低 clean 缩小差距。

四格交互 AS_BANK−NOAS_BANK−AS30＋NOAS30：clean−0.052381pp、mid urban＋0.065476pp、low urban−0.049405pp，urban均值＋0.036706pp。仅一个模型 seed，没有 SD／CI 或跨seed复现；所有小于0.1pp的正差只作初步观察。

## 错误配对及退化定位

| AS_BANK 的参照 | 场景 | 修复 | 新增错误 | 净正确数 |
|---|---|---:|---:|---:|
| 原 As30 | clean | 137 | 282 | −145 |
| 原 As30 | mid urban | 384 | 345 | ＋39 |
| 原 As30 | low urban | 453 | 482 | −29 |
| NOAS_BANK | clean | 3805 | 3736 | ＋69 |
| NOAS_BANK | mid urban | 53305 | 6703 | ＋46602 |
| NOAS_BANK | low urban | 44196 | 8824 | ＋35372 |

相对原 As30，mid／low Macro-F1 仅＋0.028802／−0.012708pp；最差RX分别−0.070833／−0.041667pp。low仅7个RX中2个、4个day中1个正差。相对 NOAS_BANK，mid／low 全RX／day均提高，但这主要是完整 As 的既有收益；clean 仍有RX下降，不能把总体正差称为全部RX保持。

源同一 V1800×2 draw：AS_BANK−AS30 的 mid修复10／新增3，净＋7/3600＝＋0.194444pp；low修复4／新增6，净−2/3600＝−0.055556pp。源 pooled midlow 交互约0；两个draw不是两个模型seed。

| 源分层 | mid n | mid Δpp | low n | low Δpp |
|---|---:|---:|---:|---:|
| LOS | 1899 | ＋0.263296 | 1060 | 0 |
| PARTIAL | 367 | ＋0.272480 | 561 | −0.356506 |
| BLOCKED | 1334 | ＋0.074963 | 1979 | 0 |
| locked | 2205 | ＋0.226757 | 1367 | −0.073153 |
| degraded | 715 | 0 | 667 | −0.149925 |
| unlocked | 680 | ＋0.294118 | 1566 | 0 |
| 新增噪声比<3dB | 383 | ＋0.261097 | 1127 | 0 |
| 3至<8dB | 718 | ＋0.139276 | 794 | −0.125945 |
| 8至<15dB | 293 | 0 | 303 | 0 |
| ≥15dB | 2206 | ＋0.226655 | 1376 | −0.072674 |

low净损失落在PARTIAL；BLOCKED及unlocked各为4修复／4新错，没有净改善。MISSING为空，不当成零性能；极小交叉层保留但不解释比例。没有重新移除某项物理损伤的反事实实验，不能将这些微小状态关联解释为新增噪声、多径、CFO、IQ或边界的因果改善。正式逐包状态未提供，状态表仅源域。

## 训练覆盖、父正确集合与曲线

| 条件 | U 观察次数 | 最终接受次数 | 最终接受率 |
|---|---:|---:|---:|
| NOAS30 | 567000 | 8963 | 1.580776% |
| AS30 | 567000 | 8726 | 1.538977% |
| NOAS_BANK | 567000 | 466144 | 82.212346% |
| AS_BANK | 567000 | 456028 | 80.428219% |

这些是567000次重复U观察，不是等量独立物理样本；unique U=56700。两新行首个本地epoch时间确认及最终接受均为0；AS_BANK从E22最终接受87.222222%至E30的90.922399%。最终bank含54902个键，54246个streak10。时间门累计通过490978次，与最终接受456028不同；U真值未开放，采用率不代表伪标签精度。提高覆盖约52倍仍没有新增urban收益，否定本轮固定假设，不能据此否定所有半监督方法。

完整四条尾段各2220步／10轮已解析；新两行567000个U转移逐条回放，首轮空bank／同轮不增长／原mask返回均核对。每行L前向1704960，As行142080完整观察；无额外NN，默认头联合更新、Sinc原单位优化保持。训练和验证曲线包含全部已记录轮次，预算耗尽不等于收敛。

补齐真实 P43 B01 As20 **student model** 的固定clean正确集合，复用先前合法源预测、无新NN/IQ/CP：1765/1800包；每satellite pooled分母3530＝1765×2draw。

| 模型 | mid urban保持数／3530 | 保持率 | low urban保持数／3530 | 保持率 |
|---|---:|---:|---:|---:|
| 实际父As20 | 2573 | 72.889518% | 2059 | 58.328612% |
| 原As30（既有读出） | 2608 | 73.881020% | 2112 | 59.830028% |
| AS_BANK | 2613 | 74.022663% | 2111 | 59.801700% |

AS_BANK在这1765个clean正确样本中保留1764个。它相对原As30的low保持数仍少1；此表仅源集合保持率，不是整体或正式low准确率。正式另报固定NoAs30 clean正确集合的保持率，不能混称父集合。

物理覆盖的直接证据为P67保存的268个RX×day×scene×state×lock格／142080观察。另一次独立CPU分析固定对照P66 MASS_UNIFORM：完整2220个L batch／284160次L呈现同序，原raw输入、同一已验证源角色／TX映射、实际channel配置／sessions／状态边际／named seed及namespace相同，As64选位相同；因此复用P66的515个TX×RX×scene×state×lock格／142080计数并核边际。该联合表是经相同确定性流核实后复用，**不是P67直接序列化的计数**；没有从U或ID数值推断TX，也没有重新物理重放。参见[覆盖证据](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/coverage_reuse_r01/complete.json)。

## 源关闭、正式阶段与实际代价

源四条件全FAIL：AS_BANK fullVclean对matchedNoAsBank −0.118519pp，对原NoAs30 −0.111111pp，midlow对原As30仅＋0.069444pp，low−0.055556pp。源决定函数不读取formal；主owner先保存关闭决定后才取新formal分析。NoAs行已先自动评分，但其新正式分数未被用于该决定。As行正式评分在源决定之后完成。正式单seed阶段通过不改变source关闭，不扩E50、不扫阈值／streak／窗口或换seed救跑。

两新tail合计20model-epochs／4440更新，nativeNN6325924＋源读出25200＋正式预测2352004＋共享CPsmoke8＝**8703136新增远端NN**。本地真实模型NN0，额外学生／教师NN0。每行原U仍567000、teacher567000计入native；没有另训教师。

对应两个历史E20父各4797722NN，原NoAs／As两个历史E30共60model-E／18298572已知NN（含其父、源及正式读出），本轮不重复收费。四条件唯一训练成本为80model-E／17760更新，已知远端NN合计27001708；这不是整个项目的全部历史成本。P66物理覆盖与父集合的既有证据仅复用，不冒称本轮新训练。

新NoAs／As native墙钟891.240831／1102.053856秒，和1993.294688秒是行耗时之和；主pipeline从提交到完整CPU分析完成为1208.785354秒，约20.15分钟。新formal预测各48.860505／48.287968秒，源读出1.845621／1.836371秒。覆盖补充1.056636秒、父集合补充0.053237秒，均0NN；不把行耗时之和当并行墙钟，不将遥测开销全归因于方法。

## 可复查交付与边界

- 实现：[配置](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_temporal_bank67/design.py)、[原生时间记忆观察](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_temporal_bank67/temporal.py)、[训练接线](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_temporal_bank67/observer.py)、[入口与自动评价](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_temporal_bank67/dispatch.py)、[完整分析](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_temporal_bank67_analysis_20261018.py)、[覆盖归属](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_temporal_bank67_coverage_20261018.py)。实际来源与参数见[runtime_configuration.json](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01/runtime_configuration.json)。
- 运行命令：python -u experiments/cvs_temporal_bank67/runner.py；CWD PROJECT/releases/cvs_FcrossRX_temporal_bank_E30_20261018_r01，普通N607既有CVS-RFFI Python3.10.19／Torch2.1.0+cu121／NumPy2.2.5，FP32／CPU2。训练code 71b3c3e340005d93433c4d4b3f51d26f4d464d60，覆盖code 296c9242fefb387e3d1289e733f1415ded289e29。精确私有运行路径保留本地运行记录。
- 28项负例＋10次原生helper toy、8分析toy、18文件compile PASS；一次限定P0/P1 PASS。首步两真实own20 student／EMA各2包合成smoke，共8NN，先于源IQ。单release archive SHA比对、remote19文件compile、PID/CWD/argv/GPU2/3／log增长及全部所属PID自然退出均VERIFIED。无真实训练技术失败、重启或选择性重跑。
- 本地最初缺参数／依赖路径、NaN比较误判与分析Python缺NumPy的调用问题均已修正或换用既有数值环境，发生在真实训练前，0NN；没有据此重启实验。覆盖后处理2项metadata toy PASS，无额外review或gate。
- 4行×7视图的504个混淆矩阵及84分区复算一致；所有新预测先固定后独立truth-last。完整124份取回文件、原日志和逐包源数组在[本地产物](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01)；weights／IQ／大logit数组保留原不可变远端，不入Git。
- [逐row逐seed指标](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv)、[全部配对错误](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv)、[源状态](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv)、[完整训练汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/training_summary.json)、[成本](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/costs.json)、[实际父集合](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/parent_cohort_r01/complete.json)。
- 四张科研图已逐张查看，均有同名PDF：[训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.png)、[时间确认与最终接受](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/bank_acceptance.png)、[源质量分层](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/source_quality.png)、[正式配对与交互](E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired.png)。

本轮技术产物缺项0；科学限制仍包括单model seed、没有新增物理反事实、正式状态未保存、源high urban及完整V27000 F1未提供、未保存逐步内部分支／融合门控轨迹。没有用新增分类器代替这些证据。当前已否定本轮固定时间确认假设；未证明新增A_g、可重复clean净正收益、low>60%或suburban接近clean，整体目标继续ACTIVE。
