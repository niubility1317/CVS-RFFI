# P66：观测质量分配未改善 clean—星地权衡

**本轮完整结束，主候选不晋级 E50。** 两个新增 E30 学生均完成 clean＋固定六 practical 预测、独立 truth-last 评分和全量日志分析，无缺失行。观测加权相对原 As30 的 urban 三场景均值增加0.408333个百分点、low增加0.482738、mid增加0.433333，但clean相对同预算NoAs30下降0.150595个百分点，超过0.1的阶段限制。相对完全匹配的同质量均匀控制，观测加权的urban均值反而低0.104365、low低0.163690、mid低0.127381，clean也低0.026786个百分点。

主候选正式low urban为51.726786%，未达到60%。只有seed2026092701一个模型种子，没有可重复双提升证据。源域判据已经在正式评分前固定为FAIL；不根据正式结果调q、换教师、扩种子或技术性重跑。E50预算授权保持有效，但本路线不符合自身晋级规则。

## 唯一假设、匹配矩阵与实际执行基线

检验额外U默认头KL的权重是否应随本次完整practical观测的新增噪声能量比变化。qT=clip((teacher pmax−.95)/.05,0,1)，qN=sigmoid((signal_to_added_noise_db−5)/5)，显式None→0。主候选q=qT×qN；控制q=qT×sum(qTqN)/sum(qT)，sum(qT)=0时两者零。全子集mean(q×KL)，T=1、教师stop-gradient，前五个续训epoch爬坡到.16。两行总q质量匹配，KL和梯度不要求相等。没有学生urban高置信门、状态删除或第二种新一致性损失。

两行独立从P43 B01 As20的student/ownEMA开始，各追加10轮，学生累计E30/6660更新。额外raw-U冻结教师仍为该同一As20的ema_model，外部holder不注册进学生/optimizer/EMA/checkpoint，不新增教师训练或部署参数。每个原U256/尾124取首min(128,n)，新增一次完整practical学生和一次raw教师前向。全部有标签As CE、六L前向768样本/step、L总尺度1.25、原RDrop、SupCon .1及旧U规则保留。原U规则实际接受uniform8714/weighted8719个；算法未变，不代表不同学习轨迹的mask必须逐位相同。

基座固定F＋CROSS_RX：SIG_TIME_X3、统一Sinc、CENTER_UNITS、joint160、默认cosine30，900722参数/206状态键，骨干与默认头按原方式联合训练。无新骨干、部署分支、分类头、独立诊断分类器或头重训。代码起点5ca93a7e8f9692971b892ec19a3bb68cc888454d；P66实际发布代码668a3f8bf39c9b1eaf4d4f0f76df40b3cc0f4fe0；实际父集合读出代码0913a798757ccb9f274c12d74abc843c322ea8bd。

数据角色为完整L6300/U56700（TX始终隐藏）/V27000、源RX1/3/4/6/8、day1/2/3、六TX、split392005；没有held源RX/Ldev。父为来源已核实的P43同seed scratch20/final_only/4440更新，无外部祖先或目标选模。尾段reset AdamW/cos10/cache，不冒充恢复连续optimizer轨迹。NoAs30/As30复用P43原冻结产物，学生预算/seed/data匹配；其额外前向成本不与新两行匹配，机制归因主要使用新两行直接比较。

实际导入的leo_practical来自PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical，而非根code默认实现。实际__file__及所有字段见[runtime_configuration.json](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01/runtime_configuration.json)。IQ为2×256、25MHz，practical前按原per-packet RMS归一化、center=false，网络内部原CENTER_UNITS保留。完整residual/post_sync，EQ=false即G=1；quality_adaptation_enabled、phase_tracking_enabled、iq_compensation_enabled和AGC均true；reference_snr_db=30、quality_reference_snr_db=20、reflect、AGC±30dB上限。训练fc2.462GHz、正式已固定fc2.45GHz均未改。L/U仅lowSub/midUrban/lowUrban各1/3，session按RX/day/eq稳定、无TX依赖，按epoch/step/sample刷新命名流。正式168000样本的clean及六practical IQ、SNR、信道和困难样本全部不变。

signal_to_added_noise_db是实际全窗信号分量与本次新增噪声分量的能量比，不是总SNR；原录音未知噪声也在signal内。quality_snr_db是白输入代理，lock和residual_scale同源，不能当独立质量证据叠乘。EQoff时噪声不经H，仍经过注入IQ处理。AGC_limited仅指增益上限，未发生逐样本幅度clipping。全窗含reflect人工历史；24样本数值延迟不等于整个无效区域。本轮未更改状态计算、噪声或边界处理。

## 源域预登记结果

主候选clean−NoAs30=−0.144444pp（V27000）；mid/low−As30=+0.402778pp；low−As30=+0.361111pp；mid/low−uniform=−0.041667pp；low−uniform=−0.111111pp。五项中只有low不低于As30通过，其余四项失败。[原始固定判定](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01/source_matrix_decision.json)没有读取正式分数。判定时间早于两行scoring完成约51秒；CPU分析复用判定，不重新计算资格。

额外U教学相对As30有小幅源收益；质量分配相对同质量均匀控制没有优势。mid两draw的weighted−uniform为−0.055556/+0.111111pp，low为−0.166667/−0.055556pp。每scene1800物理样本×2draw=3600观测，不是两个模型种子。

## 正式默认头结果

每格单位为%，每view样本数168000。全部4逻辑行×7视图先固定prediction，再由独立scorer连接truth；CPU复算504个混淆矩阵、84个分区关系。旧参照无重新预测或评分。

| 环境 | NoAs30 | As30 | 同质量均匀 | 观测加权 |
|---|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.123810 | 85.097024 |
| practical_high | 42.685119 | 81.405357 | 81.700000 | 81.690476 |
| practical_mid | 40.995238 | 78.661310 | 79.052976 | 79.035714 |
| practical_low_suburban | 39.472619 | 75.861905 | 76.323810 | 76.289286 |
| practical_high_urban | 41.439881 | 77.767262 | 78.098214 | 78.076190 |
| practical_mid_urban | 35.132738 | 62.806548 | 63.367262 | 63.239881 |
| practical_low_urban | 30.139881 | 51.244048 | 51.890476 | 51.726786 |

urban三场景均值：NoAs30=35.570833%，As30=63.939286%，uniform=64.451984%，weighted=64.347619%。weighted−NoAs30的urban总差为+28.776786pp，其中大部分来自已有完整As；不能把它全部归给新增质量分配。

Macro-F1（%）：

| 环境 | NoAs30 | As30 | 同质量均匀 | 观测加权 |
|---|---:|---:|---:|---:|
| clean | 84.979636 | 85.044474 | 84.819654 | 84.789563 |
| practical_high | 39.311416 | 81.084230 | 81.407082 | 81.398771 |
| practical_mid | 37.954813 | 78.359706 | 78.759826 | 78.748643 |
| practical_low_suburban | 36.758731 | 75.572741 | 76.012935 | 75.994022 |
| practical_high_urban | 38.481030 | 77.423208 | 77.759901 | 77.746667 |
| practical_mid_urban | 33.446603 | 62.450463 | 62.909950 | 62.824113 |
| practical_low_urban | 29.184589 | 50.887035 | 51.332060 | 51.249218 |

最差RX准确率（%，括号内为实际RX标签）：

| 环境 | NoAs30 | As30 | 同质量均匀 | 观测加权 |
|---|---:|---:|---:|---:|
| clean | 78.450000 (14-7) | 78.362500 (1-1) | 78.291667 (1-1) | 78.212500 (1-1) |
| practical_high | 34.304167 (14-7) | 71.908333 (14-7) | 72.300000 (14-7) | 72.262500 (14-7) |
| practical_mid | 33.520833 (14-7) | 70.212500 (14-7) | 70.879167 (14-7) | 70.858333 (14-7) |
| practical_low_suburban | 32.054167 (14-7) | 67.741667 (14-7) | 68.320833 (14-7) | 68.312500 (14-7) |
| practical_high_urban | 34.000000 (14-7) | 69.108333 (14-7) | 69.537500 (14-7) | 69.454167 (14-7) |
| practical_mid_urban | 29.495833 (14-7) | 56.458333 (14-7) | 57.137500 (14-7) | 57.004167 (14-7) |
| practical_low_urban | 25.750000 (14-7) | 46.379167 (14-7) | 47.162500 (14-7) | 46.995833 (14-7) |

相对NoAs30，uniform clean下降0.123810pp，weighted下降0.150595pp，均不满足下降≤0.1的阶段目标。相对As30，weighted clean下降0.244048pp。主候选low urban尚差8.273214pp才达到60%。

## 配对错误与分层一致性

下表均为weighted相对指定控制，逐view同一168000物理样本；修复指控制错而新模型对，新增错误反之。

| 对照 | 环境 | Δaccuracy/pp | 修复 | 新增错误 |
|---|---|---:|---:|---:|
| AS30 | clean | -0.244048 | 96 | 506 |
| NOAS30 | clean | -0.150595 | 3770 | 4023 |
| MASS_UNIFORM | clean | -0.026786 | 151 | 196 |
| AS30 | practical_mid_urban | 0.433333 | 1497 | 769 |
| NOAS30 | practical_mid_urban | 28.107143 | 54048 | 6828 |
| MASS_UNIFORM | practical_mid_urban | -0.127381 | 392 | 606 |
| AS30 | practical_low_urban | 0.482738 | 1717 | 906 |
| NOAS30 | practical_low_urban | 21.586905 | 45199 | 8933 |
| MASS_UNIFORM | practical_low_urban | -0.163690 | 551 | 826 |

weighted相对As30的mid/low改善覆盖7/7 RX及4/4日期，但clean在7/7 RX及4/4日期下降。weighted相对uniform的mid/low在7/7 RX及4/4日期全部下降；这是同一模型seed内分层一致性，不能当跨seed复现。逐RX/day/TX的完整数值与分母保存在CSV。

## urban退化关联与质量分配归因

两行额外U各284120次，正q255977次（90.094678%）；qTeacher零的28143包仍执行学生前向。q总质量uniform174663.746064776、weighted174663.746055250，差约9.53e−6；最大单步误差3.44e−6。weighted将质量相对uniform从lowUrban移出19.1648%、从BLOCKED移出45.3495%、从unlocked移出56.7247%；LOS/PARTIAL/locked分别增加25.6710%/13.1728%/24.6487%。全部19个实际scene×state×lock层都有正q，没有按质量删掉实际支持层。L的90个TX×RX×scene组合、U的45个RX×day×scene组合均有覆盖。

源mid相对As30：PARTIAL+1.090pp(n367)、BLOCKED+0.600(n1334)、degraded+0.839(n715)；源low：BLOCKED+0.404(n1979)、unlocked+0.383(n1566)。这些是额外教学与训练组织的状态关联。相对uniform，源low BLOCKED为−0.303pp、unlocked为−0.383pp；BLOCKED/unlocked交叉层n1565，仅修复6而新增12，净−6。mid BLOCKED/degraded为−0.152439pp(n656)。因此不能认为质量分配定向缓解了困难urban退化。

新增噪声比<3dB的源low层n1127，weighted−As为+0.532pp，weighted−uniform为−0.355pp；3至8dB层n794分别+0.378/−0.252pp。该比值不是总SNR。这些分层不识别多径、CFO、IQ或边界的因果贡献；正式逐包状态未提供。LOS/degraded、LOS/unlocked、BLOCKED/locked为空；PARTIAL/unlocked仅suburban6/mid2/low1次观测，小层和空层均保留而不作强解释。

## 实际父模型clean正确集合已补齐

另做零训练source-only读出，严格使用P43 B01 As20 checkpoint的model键（非ema_model）。其默认头在V1800的clean正确集合含1765个物理样本。每个satellite场景的3530分母=1765×2draw，不是3530个独立样本。以下保持率和正式准确率分开报告。

| 模型 | clean保持率 | mid urban保持率 | low urban保持率 |
|---|---:|---:|---:|
| PARENT20 | 1765/1765 = 100.000000% | 2573/3530 = 72.889518% | 2059/3530 = 58.328612% |
| NOAS30 | 1762/1765 = 99.830028% | 1546/3530 = 43.796034% | 1269/3530 = 35.949008% |
| AS30 | 1765/1765 = 100.000000% | 2608/3530 = 73.881020% | 2112/3530 = 59.830028% |
| MASS_UNIFORM | 1764/1765 = 99.943343% | 2622/3530 = 74.277620% | 2129/3530 = 60.311615% |
| OBSERVATION_WEIGHTED | 1764/1765 = 99.943343% | 2624/3530 = 74.334278% | 2125/3530 = 60.198300% |

weighted−uniform在mid/low分别+0.056657/−0.113314pp；weighted−As30为+0.453258/+0.368272pp。weighted和uniform均在父clean正确集合新增1个clean错误，As30无新增。源low子集保持率超过60%不等于整体或正式low准确率达到60%。该读出只补机制证据，不改变源FAIL或E50规则。

[父集合逐draw/RX/day/TX/state/lock及margin/30](E:/type10-7/local_artifacts/cvs_FcrossRX_parent20_cohort_20261018_r01/retrieved/cohort_r02/cohort_metrics.csv)；[父集合完整统计](E:/type10-7/local_artifacts/cvs_FcrossRX_parent20_cohort_20261018_r01/retrieved/cohort_r02/full_cohort_metrics.json)。原自动分析中的NoAs30正确集合仍明确标为参照集合，现有实际父集合是额外source证据，未伪造正式父集合。

## 实际成本和验证

新增训练为两条10轮尾段，共20模型轮/4440更新，各学生累计E30/6660。共享As20父训练20轮/4440只计一次，NN4797722、原墙钟1810.249311秒；同一EMA兼作冻结教师，不追加教师E20。新两行每行native3162962＋extraStudent284120＋frozenTeacher284120=3731202训练NN；合计7462404。每行L NN1704960、As142080、native U567000，全部L As CE保留。

P66新增source25200＋formal2352004＋smoke4，使远端合计9839612 NN。本次真实父source-only诊断再增加2零输入＋12600source=12602，总新增远端9852214 NN；CPU cohort分析新增NN/IQ/CP/更新均0。本地实际F合成3072 NN只执行一次、无真实IQ/CP/更新，单独列。若计共享两条历史NoAs30/As30，已知远端合计28150786 NN；历史共同smoke未分摊，不冒充全项目精确总成本。历史两参照60轮/13320更新，新尾段20轮/4440，唯一模型训练合计80轮/17760；父source读出为0轮。

实际P66矩阵墙钟1754.585893秒（29分14.586秒，从dispatcher发起至CPU分析complete）；两行训练墙钟1632.987102/1627.719365秒，之和3260.706467秒是并行行耗时之和。正式预测两行47.101255/49.027201秒；CPU全日志分析56.519828秒。父读出function wall2.529965秒，含smoke0.421874/source1.463905；其独立启动到产物完成3.840106秒。全部4个本run/诊断所属PID均已不存在；没有停止其他owner进程。没有技术失败或重复正式评分。

已执行35项协议负例、15项目标toy、4项流程检查、10项CPU分析toy；一次实际F两行合成3072 NN；P66真实CP smoke4 NN；父读出7项toy及真实CP smoke2 NN。各自一次限定P0/P1评审通过。发布包分别完成一次archive本地/远端SHA比较及23/25个Python文件编译。合成随机教师qT为0，额外梯度非零仅由toy证明；真实训练额外KL为正且native/Sinc/global梯度健康，但没有伪造隔离的逐loss梯度轨迹。部署状态始终206 keys/900722参数。6张PNG已逐一视觉检查，配套PDF存在。

## 文件、配置、命令与产物

实际代码与配置：[design.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_u_quality66/design.py)、[IQ信道](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_u_quality66/channel.py)、[教学目标](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_u_quality66/teaching.py)、[训练hook](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_u_quality66/model.py)、[计数observer](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_u_quality66/observer.py)、[调度器](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_u_quality66/dispatch.py)、[冻结预测与评分](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_u_quality66/evaluate.py)、[CPU分析](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_u_quality66_analysis_20261018.py)、[实际父读出](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_parent_cohort66/readout.py)。实际解析配置与来源见[runtime_configuration.json](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01/runtime_configuration.json)和[experiment.json](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01/experiment.json)。

实际执行入口（N607已验证CVS-RFFI环境内，CWD为对应PROJECT/releases目录）：

```text
CVS-RFFI/python -u experiments/cvs_u_quality66/runner.py
CVS-RFFI/python -u analysis/cvs_u_quality66_analysis_20261018.py --project PROJECT --out RUN/analysis_r01 --local-nn 3072 --code-commit 668a3f8bf39c9b1eaf4d4f0f76df40b3cc0f4fe0
CVS-RFFI/python -u experiments/cvs_parent_cohort66/runner.py
```

这里PROJECT、RUN和CVS-RFFI/python是报告中的既有远端路径别名；原始实际argv/CWD保存在未入Git的[主运行命令](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/launch_private.json)、[父读出命令](E:/type10-7/local_artifacts/cvs_FcrossRX_parent20_cohort_20261018_r01/launch_private.json)和[CPU父集合分析](E:/type10-7/local_artifacts/cvs_FcrossRX_parent20_cohort_20261018_r01/cohort_analysis_remote_r02.py)。不可变主根为PROJECT/runs/20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01，父读出根为PROJECT/runs/20261018-diagnostic-FcrossRX-parent20-cohort-s2026092701-m1-r01。原始IQ、checkpoint、完整U物理trace及逐view logits留远端；没有另行预测。已取回128个报告/日志/统计文件，压缩包49480408字节，无IQ/权重。

关键统计文件：

- [formal_metrics.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv)
- [formal_pair_summary.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv)
- [formal_paired_effects.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv)
- [source_paired_effects.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)
- [training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.csv)
- [training_coverage.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/training_coverage.csv)
- [training_U_coverage.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/training_U_coverage.csv)
- [new_tail_training.json](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/new_tail_training.json)

图表：

- [training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.pdf)
- [u_quality_training PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/u_quality_training.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/u_quality_training.pdf)
- [u_quality_physical_allocation PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/u_quality_physical_allocation.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/u_quality_physical_allocation.pdf)
- [practical_mid_urban_source_states PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/practical_mid_urban_source_states.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/practical_mid_urban_source_states.pdf)
- [practical_low_urban_source_states PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/practical_low_urban_source_states.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/practical_low_urban_source_states.pdf)
- [formal_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/formal_accuracy.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/analysis_r01/formal_accuracy.pdf)

两行完整训练日志：[uniform](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/logs/MASS_UNIFORM-s2026092701.log)、[weighted](E:/type10-7/local_artifacts/cvs_FcrossRX_U_quality_E30_20261018_r01/retrieved_r01/logs/OBSERVATION_WEIGHTED-s2026092701.log)。原始20个新增epoch/4440更新、双格式序列8880条、ordinary MSE和U前缀均全量核对，不使用抽样日志冒充完整审计。

本轮运行产物missing=[]。科学限制仍包括：单模型seed；未验证可重复双提升；未达到0.1pp/low60目标；source高urban与全V Macro-F1没有保存；正式逐包状态、实际As20父正式预测和隔离教学/内部门控完整轨迹缺失；没有固定接收状态的物理反事实识别。预算结束不等于收敛。这些限制不因本轮闭环而消失，整体研究目标仍未完成。

补充：主候选clean默认头平均margin/30相对NoAs30增加0.005642，但准确率下降0.150595pp。平均间隔不替代实际识别性能。父source短读出分配GPU7，在首次postcheck前完成，未捕获活跃GPU PID快照；其终态由严格CP smoke、完整预测、冻结state和12602计数核实。
