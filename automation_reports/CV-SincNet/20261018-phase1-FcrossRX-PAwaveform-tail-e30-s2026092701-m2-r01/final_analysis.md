# P60 结果：末十轮停止普通 PA 波形注入未改善星地性能

状态：**VERIFIED / ANALYZED**。两个新尾段训练、各 clean＋六 practical 固定预测、独立 truth-last 评分、四行汇总均已完成。analysis_r01/complete.json 为 ANALYZED，missing=[]；原 dispatcher、两个 worker 与零 NN 分析进程均已终止。不是仅训练完成。全局性能目标尚未完成。

**判断：完整 practical 增强的收益应保留；本次新增 PA 末段旁路不晋级。** 相对原 As30，clean +0.194643 个百分点，但 mid urban −0.104762、low urban −0.093452、urban 三场景均值 −0.088889；六种 practical 均小幅下降。预先固定的三项源条件全 FAIL，因此关闭本固定方案，不扫参数、不扩 E50 或种子。正式 clean 的上涨不改变源决定。

匹配 PAOFF As−PAOFF NoAs 为 clean +0.317857、mid +27.576786、low +20.962500、urban 均值 +28.228968 个百分点，单 seed 满足 clean 下降不超过 0.1 绝对百分点的阶段目标。大部分星地收益来自原完整 As，不能归于新 PA 处理。low urban 51.150595%，未达到 60%；只有 model seed2026092701，没有可重复双提升证据。

## 已执行设计与真实基线

代码/config 提交：23f794296a11e1a4f25837e78b60f36bc700b99e；启动前工作树已核实，父来源为原 P43 B00／B01 各自 scratch20 的 student＋own EMA。两个父 E1–20 均已使用普通 PA；本项只改变 E21–30，不能称全程无 PA。复用原 NoAs30、As30 全部预测和分数，未重训、重预测或重评分。新两行各继承 own20＋新10，累计 E30／6660 更新；新尾各 2220 更新。

固定 F＋CROSS_RX、SIG_TIME_X3／unified Sinc／joint160／默认 cosine30 分类头，网络与默认头按原流程联合更新，无替换头、额外诊断头或部署模块。原 6L 为 ordinary128×2、mixed128×2、raw128×2，共 768 个 L 前向/更新；L 总尺度1.25、ordinary128 SupCon0.1、同 IQ RDrop、旧 U 门控及 own EMA 保留。本轮没有附加一致性损失或新 Ag。

PA 是 ordinary 的末项。新实现仍在 clone 上执行完整原 simulate_pa，保留 Python／NumPy／Torch 原生随机消耗，最后返回 PA 前 prefix＋零实际注入强度。实际开关仍 true，base p_pa=.14，冻结 E13 的实际 p_pa=0.03508404675089638；不能把 enabled=true 解释为旁路未生效。As、raw、U 的处理不变。

原完整 As 训练场景为 practical_low_suburban／practical_mid_urban／practical_low_urban，各1/3，样本从 raw 经原 normalize_input 进入完整 residual，不先叠普通 PA。几何、遮挡、传播、噪声和接收补偿依赖未拆散。RX 硬件按15个 RX/day session 稳定，不由 TX 定义；新 As142080，无有标签置信度筛除。

实际导入：PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py；并非仓库另一个 code/ 实现。DataAugmentation/train_ssdg 同属该固定 runtime。IQ256、fs25MHz，原逐包 complex RMS、center=false 的 normalize_input 位置不变；训练fc2.462GHz，正式固定fc2.45GHz。实际 residual／post_sync／equalization_enabled=false（G=1）／reference_snr_db30／接收质量适配、相位跟踪与 IQ 补偿开启／reflect 边界，与各自原对照一致。residual 不能解释为已均衡。

L6300／隐藏 U56700／V27000；source RX1/3/4/6/8、day1/2/3、全部6TX、split392005；没有留出源 RX 或 L_dev。两个父 checkpoint 与角色 EXACT_MATCH、从零祖先与 fixed-final20 来源已核实，未继承 P59、E30、E50 或目标选模权重。严格加载 student/own EMA 后重置 AdamW／cos10／cache，lr.0002、WD.0001。复用未变 VALIDATED_ONCE 和正式 IQ，不新增数据验证链。

## 全部正式结果

准确率单位为 %；差值为绝对百分点。各视图168000次观察；七视图来自同一物理测试集合，不能按独立样本合并。所有结果均是冻结 checkpoint 内默认头，seed2026092701。

| 固定视图 | 原 NoAs30 | 原 As30 | PAOFF NoAs | PAOFF As | 匹配 As−NoAs（pp） | PAOFF As−原 As（pp） |
|---|---:|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.217857 | 85.535714 | 0.317857 | 0.194643 |
| practical_high | 42.685119 | 81.405357 | 42.744048 | 81.348810 | 38.604762 | -0.056548 |
| practical_mid | 40.995238 | 78.661310 | 41.082738 | 78.547619 | 37.464881 | -0.113690 |
| practical_low_suburban | 39.472619 | 75.861905 | 39.584524 | 75.726786 | 36.142262 | -0.135119 |
| practical_high_urban | 41.439881 | 77.767262 | 41.551190 | 77.698810 | 36.147619 | -0.068452 |
| practical_mid_urban | 35.132738 | 62.806548 | 35.125000 | 62.701786 | 27.576786 | -0.104762 |
| practical_low_urban | 30.139881 | 51.244048 | 30.188095 | 51.150595 | 20.962500 | -0.093452 |
| urban 三场景均值 | 35.570833 | 63.939286 | 35.621429 | 63.850397 | 28.228968 | -0.088889 |

下表每格为宏平均 F1%／最差 RX 准确率%。完整 RX/day/TX、逐类 CM、间隔与保持率保存在 formal_metrics.csv／formal_confusions.json，不只报告总体准确率。

| 视图 | 原 NoAs30 | 原 As30 | PAOFF NoAs | PAOFF As |
|---|---:|---:|---:|---:|
| clean | 84.979636 / 78.450000 | 85.044474 / 78.362500 | 84.943290 / 77.850000 | 85.254226 / 78.837500 |
| practical_high | 39.311416 / 34.304167 | 81.084230 / 71.908333 | 39.210368 / 34.400000 | 81.031007 / 70.895833 |
| practical_mid | 37.954813 / 33.520833 | 78.359706 / 70.212500 | 37.958896 / 33.591667 | 78.243040 / 69.145833 |
| practical_low_suburban | 36.758731 / 32.054167 | 75.572741 / 67.741667 | 36.831615 / 32.050000 | 75.425993 / 66.791667 |
| practical_high_urban | 38.481030 / 34.000000 | 77.423208 / 69.108333 | 38.504552 / 34.091667 | 77.358414 / 68.016667 |
| practical_mid_urban | 33.446603 / 29.495833 | 62.450463 / 56.458333 | 33.503653 / 29.545833 | 62.336456 / 55.687500 |
| practical_low_urban | 29.184589 / 25.750000 | 50.887035 / 46.379167 | 29.383919 / 25.883333 | 50.779681 / 45.833333 |

相对原 As30，新方案 mid/low 最差 RX 由56.458333／46.379167降至55.687500／45.833333%，分别−0.770833／−0.545833个百分点。整体卫星小幅下降且弱 RX 没有改善。

## 收益归因与错误变化

| 比较 | clean Δpp | mid urban Δpp | low urban Δpp | urban 均值 Δpp |
|---|---:|---:|---:|---:|
| PAOFF As−PAOFF NoAs | +0.317857 | +27.576786 | +20.962500 | +28.228968 |
| PAOFF As−原 As30 | +0.194643 | −0.104762 | −0.093452 | −0.088889 |
| PAOFF NoAs−原 NoAs30 | −0.029762 | −0.007738 | +0.048214 | +0.050595 |
| PA×As 交互：两种 As 效应之差 | +0.224405 | −0.097024 | −0.141667 | −0.139484 |

| 新 PAOFF As 的对照 | 视图 | 修复旧错误 | 新增错误 | 净修复 |
|---|---|---:|---:|---:|
| 原 As30 | clean | 957 | 630 | 327 |
| 原 As30 | mid urban | 1081 | 1257 | −176 |
| 原 As30 | low urban | 1438 | 1595 | −157 |
| 原 As30 | urban 三视图合计 | 3277 | 3725 | −448 |
| PAOFF NoAs | clean | 4056 | 3522 | 534 |
| PAOFF NoAs | mid urban | 53335 | 7006 | 46329 |
| PAOFF NoAs | low urban | 44299 | 9082 | 35217 |
| PAOFF NoAs | urban 三视图合计 | 162683 | 20409 | 142274 |

urban 错误合计为三个视图504000次观察，并非504000个独立物理样本。四态交互显示 clean 有局部正变化，所有六种 practical 的交互均为负值。不能用相对 NoAs 的巨大差值宣称 PA 处理改善了星地。

默认头尺度固定30；mean margin/30 的 clean／mid／low：原 As30=.14612423/.05380980/.02807553，PAOFF As=.14463294/.05366503/.02782491。clean 准确率上涨并不等于平均间隔上涨。固定原 NoAs30 clean 正确集合143216个样本，PAOFF As 的 clean/mid/low 保持率97.479332/68.729053/55.537789%，原 As30 为97.408809/68.782119/55.632751%。这是原 NoAs30 的正确集合，不能冒充 own20 父模型集合；本项未另算 own20 的固定集合。

## 源判定与退化分层

以下 source_matrix_decision 在读取正式成绩前已经写定，formal_scores_read=false、target_feedback=false。

| 固定晋级条件 | 实际值 | 结论 |
|---|---:|---|
| full V：As clean−匹配 NoAs ≥−0.1pp | −0.111111pp | FAIL |
| 两 draw mid/low 均值−原 As30 ≥+0.5pp | −0.097222pp | FAIL |
| 两 draw low−原 As30 ≥0pp | −0.222222pp | FAIL |

full V27000 clean：原 NoAs98.359259%、原 As98.255556%、PAOFF NoAs98.351852%、PAOFF As98.240741%；各最差源 RX 为95.611111%、95.333333%、95.592593%、95.185185%。匹配源 clean 相差30/27000，超出0.1pp允许的27个样本3个；这不是显著损伤的证明，但不得事后放宽固定判据。

V1800×两 draw 的源关联分层（两 draw 不是两个模型 seed）：

| 场景／传播与锁定状态 | 次数 | 原 As30→PAOFF As | Δpp | 修复／新增 |
|---|---:|---:|---:|---:|
| mid BLOCKED/degraded | 656 | 42.530488%→42.378049% | −0.152439 | 8／9 |
| mid BLOCKED/unlocked | 678 | 31.563422%→31.710914% | +0.147493 | 7／6 |
| low BLOCKED/degraded | 414 | 42.028986%→41.304348% | −0.724638 | 2／5 |
| low BLOCKED/unlocked | 1565 | 30.351438%→30.287540% | −0.063898 | 21／22 |

没有一致的遮挡/低锁定退化缓解证据。完整 source_strata.csv 保留 LOS/PARTIAL/BLOCKED、locked/degraded/unlocked、质量与 signal_to_added_noise_db 分箱、RX/day/TX 和空格；不删除 n=0 或小格，不把新增噪声 SNR 写作总 SNR。正式数据没有逐样本接收状态，不能给 formal 捏造状态分层。

既有 P24 已区分固定接收状态的受控效应与移除损伤后重算状态的总效应；旧模型上新增噪声恢复证据不能直接套给 P60。实际功率代理可切换状态，但旧证据不支持它解释主要缺口；FIR62taps/数值延迟24/前61边界及内部195区域亦已有观察。本项未新增 EQ-on 噪声着色试验，不能声称验证。详见[既有损伤控制范围](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-PAwaveform-tail-e30-s2026092701-m2-r01/source_counterfactual_context.md>)。本项不能把失败确定归因于多径、CFO、门控或 PA 标签破坏。

## 实际覆盖、曲线与验证

两新行各2220个 ordinary调用/284160包，实际 shadow PA2202次，选中10192包，其中10178包本会发生波形变化；实际注入均为0。两行与各自原 tail 的4440条物理序列记录完全一致；ordinary MSE 的2198/2220步发生预期变化，平均差−.002560813、最大绝对差.278115490，不能错误要求波形 MSE 相同。影子数值计算不增加网络前向。

As 实际142080包、15 RX/day session，覆盖全部三场景/三传播/三锁定状态及全部源 TX×RX，dropped=0、confidence_filter=false；NoAs 的142080 identity_control 统计为 NONE/NOT_APPLICABLE，不是真实信道包或15个硬件会话。

完整80个 epoch 记录已分析：原两行各E1–30，新两行仅E21–30，无缺轮；步日志各2220条。损失、V、梯度、学习率、时长无非有限值，梯度非零；末段损失仍波动，预算用尽不代表收敛。曲线使用真实epoch、不平滑、不伪造新行E1–20连接点。

已执行：11项聚焦协议/实现检查；一次独立P0/P1审查11项PASS、P0/P1=0/0；本地合成前向5376（成功4608＋一次错误toy断言768）；19项零NN分析检查PASS；一次N607预检；一次release归档本远端SHA比较；远端15py编译PASS；真实两父checkpoint复合无query smoke4NN/2CP/0IQ/0更新PASS后直接训练。启动后PID/CWD/argv/CUDA/GPU/log增长已独立核实。

独立评分后 CPU 汇总验证四逻辑行全部七视图，504个混淆矩阵重计、84项分区对账完成。分析自身NN0、IQ0、checkpoint0、updates0，耗时28.188888秒，missing=[]。无需再做一次未改变代码的复审或测试。

## 成本与预算

NN 表示网络前向样本量；并行进程 wall 不相加冒充整个实验历时。

| 成本 | NoAs PAOFF | As PAOFF |
|---|---:|---:|
| 父20 updates／native NN | 4440／4797722 | 4440／4797722 |
| 父20训练wall秒 | 1329.394069 | 1810.249311 |
| 新10 updates／L NN | 2220／1704960 | 2220／1704960 |
| 新10 student train／eval NN | 2271960／324002 | 2271960／324002 |
| 新10 teacher train／eval NN | 0／567000 | 0／567000 |
| 新10 native NN | 3162962 | 3162962 |
| 新10训练wall秒 | 829.358156 | 1114.902985 |
| 源读出 NN | 12600 | 12600 |
| 正式预测 NN | 1176002 | 1176002 |
| 正式预测wall秒 | 48.400314 | 47.689010 |
| 每条累计 E／updates | 30／6660 | 30／6660 |

新增合计20模型epoch／4440更新／L NN3409920；新增native NN6325924、源25200、正式2352004、真实smoke4，远端新NN8703132。本地合成5376另计。两个旧 E30 参考已知NN18298572，包含共享父20，故整个四态矩阵唯一训练80模型epoch／17760更新、已知远端NN27001704，不重复算成120epoch。旧六行共享smoke无法分摊，保留未分摊标记；源读出wall未单独落盘，不补估。详见costs.json。

## 文件、命令与未完成项

实际代码：[候选实现](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_pa_waveform60/>)、[冻结分析](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_pa_waveform60_analysis_20261018.py>)、[自动等待分析](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_pa_waveform60_wait_20261018.py>)；精确config/数据/父来源/逐行矩阵见[experiment.json](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-PAwaveform-tail-e30-s2026092701-m2-r01/experiment.json>)。

已执行训练命令：在 PROJECT/releases/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01 内，`CVS-RFFI/python -u experiments/cvs_pa_waveform60/runner.py`。ROOT为 PROJECT/runs/20261018-phase1-FcrossRX-PAwaveform-tail-e30-s2026092701-m2-r01；PROJECT 为原 N607 项目目录，仅普通账号。不要在原ROOT重新执行或覆盖。analysis wrapper沿原dispatcher完成后执行冻结分析，不追加训练/预测/评分。

本地完整交付：[E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01>)。具体文件：[complete.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/complete.json>)、[formal_metrics.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv>)、[formal_paired_effects.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv>)、[formal_interaction.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_interaction.csv>)、[formal_confusions.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_confusions.json>)、[source_strata.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv>)、[source_paired_effects.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/source_paired_effects.csv>)、[source_interaction.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/source_interaction.csv>)、[training_curves.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.csv>)、[training_coverage.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/training_coverage.csv>)、[PA_shadow_steps.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/PA_shadow_steps.csv>)、[ordinary_MSE_comparison.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/ordinary_MSE_comparison.csv>)、[costs.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/costs.json>)。数值输出和图不入Git；Git仅提交代码/config/公共报告与登记。

已逐张查看五张实际PNG，PNG/PDF均保留：[formal_accuracy](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_accuracy.png>)、[formal_pair_changes](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_pair_changes.png>)、[training_curves](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.png>)、[practical_mid_urban_source_states](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/practical_mid_urban_source_states.png>)、[practical_low_urban_source_states](<E:/type10-7/local_artifacts/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01/retrieved_r01/analysis_r01/practical_low_urban_source_states.png>)。

本轮预登记交付无缺项。研究未完成：该新机制无星地增量；缺少跨modelseed双提升证据、own20固定clean正确集合、正式状态标签与本候选的受控物理因果分解。后两项不作为追加当前试验门槛。不是所有未完成研究都需要补实验；本固定候选按源FAIL关闭。

表现好且合法源证据支持的方案可按用户授权扩到总计E50；本项不符合。既有原As配方E50三seed扩展已完成，lowurban相对各自As30平均+1.156746pp、mid+.954167pp；对匹配NoAs50 clean平均−.150198pp，未满足≤.1pp，亦无可重复双提升。不得据正式分数再挑种子/重跑。详见[已完成E50三seed](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01/E50_three_seed_readout.md>)。

另一聊天的stepALT为相同As总剂量、按step交替128/0而非每step64的独立假设；当前PA旁路不重复其路线、不修改其健康任务。任何后续新候选先核对该聊天最新状态与索引，仍以合法源证据设计，不使用本轮正式结果调参或晋级。

最终回答：本候选未证明缓解某一类urban退化；默认头相对原As30的mid/low/urban分别−.104762/−.093452/−.088889pp，clean+.194643pp。相对匹配无As对照的阶段clean要求单seed通过，但未获得星地增强同时改善clean的可重复证据。
