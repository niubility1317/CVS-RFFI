# P45 完整结果：L-only 硬标签 CE 未通过源域条件

**状态：ANALYZED，固定 L0 消融关闭；联合 Ag＋As 目标仍 ACTIVE。** 本轮沿用户确认的 clean 监督保留方案，完成两条新 E30、自动 clean＋六 practical 预测与独立 truth-last 评分。没有实现新的 IQ 泛化增强。

相对匹配原 As 对照 B01，H01 默认头 clean 为 **85.497%（+0.156 个百分点）**，mid urban 为 **62.871%（+0.064）**，low urban 为 **51.318%（+0.074）**，urban 三场景均值为 **63.973%（+0.034）**。这不是实质 urban 突破，未达到 low urban >60% 或 suburban 各距 clean ≤2 个百分点；只运行一个模型 seed，未证明跨 seed 重现。源域六项固定条件 2 PASS／4 FAIL，已在新正式数值读取前提交关闭结论，正式小幅变化不用于翻案或扩种子。

## 1. 实际代码、数据与训练边界

- 编辑起点 commit：`db766cf0e7d27df8826815ff5812fd3dcc3ed4c9`；实际训练代码/config：`62b9bb6247b29fc98bf1379b15f3c262f9a495dc`；独立分析代码：`1928b5ec55e584f4a93c77124fe1dfd7541c7e6f`；正式新数值读取前的源否定结论：`043256b73db74300b03830c7d70aaa047f286b23`。均已推送并独立比较远端 OID 一致。交付收尾 commit 见本文件 Git 历史。
- 新行 H00/H01 从零训练自己的 E20（13 L＋7 原 U），再严格载入各自 student/EMA 追加 E10（10 原 U），累计 E30／6,660 更新。L 贯穿全部 30 轮。追加段 reset AdamW、lr=0.0002、WD=0.0001、cosine10；自身 E20 已计入 E30，无额外父训练。没有加载 E200 或其他历史模型初始化。
- F／SIG_TIME_X3＋统一 Sinc CENTER_25K/CENTER_UNITS＋NoExplicitCFO＋原内部门控／参考响应＋joint160／默认 cosine30，随骨干联合训练。原 CROSS_RX 同 TX 对比学习、RDrop、旧 U 门控及 own EMA 保持；没有新分类头、独立拟合头或部署参数。
- 原 all5 源 RX=[1,3,4,6,8]、day1/2/3、6 TX；L=6,300、U=56,700（TX 真值隐藏）、V=27,000，split=392005，IQ=256、fs=25 MHz。无训练 RX 留出或 L_dev。native 每包 RMS、center=false；增强顺序保持，As 从普通增强前的原 IQ 生成。
- 实际导入不是根 `code/leo_practical`：四个新训练段 `__file__` 均读回为固定旧 release `cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py` 及同目录 `residual.py`。原 `DataAugmentation.py`、`SSDG/train_ssdg.py` 也来自该固定目录。完整绝对路径保存在私有 `terminal_verification.json`，未按当前 CWD 猜测导入。
- 实际 As 为 residual／post_sync，fc=2.462 GHz、reference_snr_db=30、EQ=false，故 G=1。质量适配、相位跟踪、IQ 补偿、AGC 开启；reflect 边界。配置中的 MMSE／129 taps／delay64 不表示实际执行 EQ。low urban 的 locked/unlocked 代理阈值 15/5 dB、quality reference20 dB、max residual scale10；fractional_delay_half_length=24。全部六场景完整配置在 H01 与 B01 的两个训练段逐字段相同。
- 训练场景仍 low_suburban／mid_urban／low_urban，RX 硬件按 RX/day/EQ session 稳定，不按 TX 定义，传播／遮挡／噪声／状态／补偿依赖及命名随机流保持。正式 received IQ、测试 SNR、困难样本和全部七视图不变。新增噪声比不是总 SNR；本轮不把质量代理解释成总链路 SNR。

本次查看实际 resolved config 时发现继承的**描述性**字段 `budget_authorization`、`cost_claim`、`source_selection`、部分 `training_mechanism.expected_L_forward_packets` 仍带旧 E20/旧前向文本；`stage`／`method`／禁用的 clock/noise/factorial 说明也含历史名字。这是配置说明缺陷，原冻结产物保留，不回写伪装成当时正确。实际执行由当前 runner 的 segment/cumulative_epochs/cumulative_updates、L_NN_per_step=768、实测逐步计数和初始化记录确认：**两行各 E30／6,660 更新／5,114,880 L NN，new_Ag=false，L0、native/U=.01**。分析字段 `extra_Ag_control_identity_NN` 实为共同 raw 控制的 1,704,960 NN，不是已执行 Ag。不得按这些旧描述推断额外训练、噪声或 SRO 已开启。

## 2. 最小实验与实际实现

| 行 | 来源 | L CE 平滑 | As | 实际预算 |
|---|---|---:|---|---|
| B00 | P43 已冻结控制，未重训 | .01 | 关闭 | 原 E30 |
| H00 | 本轮新增，从零＋自身续训 | 0 | 关闭 | E30 |
| B01 | P43 已冻结控制，未重训 | .01 | 开启 | 原 E30 |
| H01 | 本轮新增，从零＋自身续训 | 0 | 开启 | E30 |

P43 对照代码为 `19aedc3604fed9c193237dd92366164178af0983`，来源为 `20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01`。这是 **L 损失定义×As** 四格，不是 Ag×As；不能把 B01 的既有 As 收益算成本轮新增机制收益。

四行每步均为 6×128 L NN：0.5 RD(ordinary128)＋0.5 RD(ordinary64＋As64 或 raw64)＋0.25 RD(raw128)，L 总尺度 1.25，另保留原 ordinary 上两层／两 pass 的 CROSS_RX SupCon。每个 RD 对复用同一个已生成 IQ，没有第二次增强重抽。所有有标签 As 均参加 CE。

本轮仅在 L hook 三次 terms 中把 CE.01 换成 CE0，复用原 KL；以原 hook 的 total−identity_loss 抵消 native 初始有标签损失，finally 恢复函数后才返回 U。native 参数及 U 平滑始终 .01，无新 U 信道、门控、损失、教师或前向。CE0/CE01 差在同一 H 更新后权重重新计算，是损失定义差，不能当成沿 B 轨迹的反事实训练结果。

实际修改文件：`experiments/cvs_hard_lce45/{design,losses,model,source,observer,dispatch,evaluate,qualification,quality,channel,protocol,preflight,publish,criteria,report,runner,checks,__init__}.py`；分析为 `analysis/cvs_hard_LCE_E30_results_20261018.py` 与 `analysis/cvs_hard_LCE_analysis_checks_20261018.py`，复用已有四格读数工具。具体代码以训练／分析 commit 区分。

## 3. 源域预登记判断先行

新正式数值首次读取在源否定结论 commit `043256b73db74300b03830c7d70aaa047f286b23` 之后。完整源判断见 [source_verdict.json](source_verdict.json)。全 V clean：

| B00 | H00 | B01 | H01 |
|---:|---:|---:|---:|
| 98.359259% | 98.325926% | 98.255556% | 98.214815% |

H00−B00=−0.033333 个百分点，未达到 +0.2；H01−B00=−0.144444、H01−B01=−0.040741。两 draw×mid/low 的 H01−B01 分别为 [[+0.222222,−0.111111],[+0.111111,+0.333333]] 个百分点，0/4 达到预定 +0.5。source low_suburban 与逐源 RX 保护通过，不能抵消其余四项失败。

源 panel 是同一 V1800×2 信道 draw（1,800 个唯一 IQ，不是 3,600 个独立样本）；其旧代理 fc=2.45 GHz 与训练／正式 fc=2.462 GHz 的差异保留披露。下表汇总两 draw 的 low urban 状态，仅用于描述源域，不用于推断正式目标的损伤因果：

| 源状态／接收锁定 | 两 draw 观察数 | B01 准确率% | H01 准确率% |
|---|---:|---:|---:|
| LOS/locked | 1060 | 94.717 | 94.906 |
| PARTIAL/locked | 307 | 88.925 | 88.274 |
| PARTIAL/degraded | 253 | 79.447 | 79.051 |
| PARTIAL/unlocked | 1 | 0.000 | 0.000 |
| BLOCKED/degraded | 414 | 42.029 | 42.029 |
| BLOCKED/unlocked | 1565 | 30.351 | 30.671 |

BLOCKED/unlocked 仍明显薄弱；PARTIAL/unlocked 只有 1 次观察，不能外推。全部状态和 added-noise-ratio 分箱均保留，未删除困难样本。H01 的局部变化没有证明某一类 urban 损伤得到稳定缓解。

覆盖／随机流核对：每行实际 L 852,480 次物理输入、全部 6,300 唯一样本；原 U 963,900 次输入、全部 56,700 唯一样本。H01 与 B01 实际 As 都为 426,240 包。两个匹配对的 stage20 5,994 条、追加段 4,440 条 L/U 物理序列记录均 0 mismatch；ordinary 实际增强 MSE 的各 6,660 步均一致；H01/B01 的 As 包数／场景／状态逐步各 6,660 条均一致，且两段实际信道配置全字段一致。先前 P43 直接统计的 TX×RX×day×scene×state 810 格均非空、最小 75，本轮依据这些匹配证据继承其覆盖；**没有重新读取 IQ 或逐包重放 810 格**，不伪称新增直接测量。

## 4. 正式默认头：全部七个固定视图

模型 seed=2026092701，只有 1 个；每行每视图 168,000 个固定样本。所有七视图预测固定后才独立 CPU truth-last 评分。四行的 504 个 RX/TX/day/overall 混淆矩阵由保存 prediction 再算，与原 scorer 全部一致；1168 个源矩阵也复算一致。两新行共 2,352,000 个实际正式判断；两控制的既有 2,352,000 个判断只读复用，没有重测。

准确率%，最后一列为 H01−B01 的百分点：

| 视图 | B00 | H00 | B01 | H01 | 增量 |
|---|---:|---:|---:|---:|---:|
| `clean` | 85.248 | 85.160 | 85.341 | 85.497 | 0.156 |
| `practical_high` | 42.685 | 47.996 | 81.405 | 81.352 | -0.054 |
| `practical_mid` | 40.995 | 45.851 | 78.661 | 78.618 | -0.043 |
| `practical_low_suburban` | 39.473 | 43.850 | 75.862 | 75.924 | 0.062 |
| `practical_high_urban` | 41.440 | 46.371 | 77.767 | 77.729 | -0.038 |
| `practical_mid_urban` | 35.133 | 38.289 | 62.807 | 62.871 | 0.064 |
| `practical_low_urban` | 30.140 | 32.295 | 51.244 | 51.318 | 0.074 |
| urban 三场景均值 | 35.571 | 38.985 | 63.939 | 63.973 | +0.034 |

宏平均 F1（六类统一口径，×100）：

| 视图 | B00 | H00 | B01 | H01 |
|---|---:|---:|---:|---:|
| `clean` | 84.980 | 84.872 | 85.044 | 85.233 |
| `practical_high` | 39.311 | 43.940 | 81.084 | 81.060 |
| `practical_mid` | 37.955 | 42.174 | 78.360 | 78.345 |
| `practical_low_suburban` | 36.759 | 40.596 | 75.573 | 75.669 |
| `practical_high_urban` | 38.481 | 42.806 | 77.423 | 77.431 |
| `practical_mid_urban` | 33.447 | 36.318 | 62.450 | 62.563 |
| `practical_low_urban` | 29.185 | 31.311 | 50.887 | 50.999 |

最差目标 RX 准确率%：

| 视图 | B00 | H00 | B01 | H01 |
|---|---:|---:|---:|---:|
| `clean` | 78.450 | 77.087 | 78.362 | 79.204 |
| `practical_high` | 34.304 | 35.721 | 71.908 | 71.167 |
| `practical_mid` | 33.521 | 34.779 | 70.213 | 69.558 |
| `practical_low_suburban` | 32.054 | 33.350 | 67.742 | 67.354 |
| `practical_high_urban` | 34.000 | 35.167 | 69.108 | 68.313 |
| `practical_mid_urban` | 29.496 | 30.354 | 56.458 | 56.092 |
| `practical_low_urban` | 25.750 | 26.375 | 46.379 | 46.383 |

H01 的 clean 与三个 suburban 差距分别为 **4.145、6.879、9.573 个百分点**，全部超过探索目标 2。mid urban 的整体准确率虽微增，最差 RX 下降 0.367 个百分点；low urban 最差 RX 仅增 0.004 个百分点，未显示广泛 RX 改善。

## 5. 收益归因与反证

- 原 As 的独立收益 B01−B00：clean +0.093、mid urban +27.674、low urban +21.104、urban 均值 +28.368 个百分点。主要星地收益来自既有完整 As 训练。
- 仅硬 L 的独立变化 H00−B00：clean −0.088、mid urban +3.157、low urban +2.155、urban 均值 +3.414。这发生在无 As、星地识别仍很低的模型上，不能叠加预测组合成绩。
- 硬 L 加到 As 上 H01−B01：clean +0.156、mid +0.064、low +0.074、urban 均值 +0.034。urban 交互项约 −3.381 个百分点，单独增量未被联合保留；详见 [完整交互 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/final_analysis_r01/factorial_effects.csv)。

H01 相对 B01 的同 ID 错误转移：

| 视图 | 修复原错误 | 新增错误 | 净正确数 |
|---|---:|---:|---:|
| `clean` | 1328 | 1066 | 262 |
| `practical_high` | 1315 | 1405 | -90 |
| `practical_mid` | 1535 | 1608 | -73 |
| `practical_low_suburban` | 1806 | 1701 | 105 |
| `practical_high_urban` | 1463 | 1527 | -64 |
| `practical_mid_urban` | 2230 | 2122 | 108 |
| `practical_low_urban` | 2626 | 2501 | 125 |

固定 B00 FINAL30 clean 正确集合共 143,216 个样本（它是匹配固定对照，不是共同训练父模型）：B01→H01 的 mid 保持数 98,507→98,534；low 保持数 **79,675→79,673（少 2）**，保持率约 55.633%→55.631%。两模型共同 clean 正确集合 142,307 个样本中，mid 修复／新增错误 1,776／1,743，low 为 2,150／2,118。整体 low +125 个净正确样本大部分被新错误抵消，不是“clean 身份稳定迁移”已成立。

cosine 尺度真实固定 30，无温度或额外头拟合。H01 平均真实类间隔÷30 在 clean 为 0.205966（B01 0.146124）、mid 0.076361（0.053810）、low 0.041798（0.028076）。间隔增加与正确率微增并不等价，也未测 ECE，不宣称校准改善。没有保存本轮逐包训练分支／内部门控轨迹，不补造该机制证据。

## 6. 完整日志、必要验证与成本

本地候选 7 项聚焦检查、18 文件 AST PASS；仅一次独立限定 P0/P1 审查 PASS。launcher 第一步真实合法 checkpoint 的无 query 合成 smoke=2 NN＋两行实际完整 F/L-hook 合成=1,536 NN，PASS 后立即训练。一次 N607 普通账号 preflight、一次训练归档 SHA 比较、一次 remote compile PASS。分析本地与 N607 各 7 项纯合成检查、单分析归档比较／compile PASS，无 IQ／checkpoint／训练或 head-fit 更新。

完整四行共 120 个 epoch 记录、各 6,660 步与 4,869 行 stdout 均读取，无错误标记、无有效数值非有限。stdout 每行模型的 300 次 NaN 文本来自禁用的原 sat_cos／DM-ACCEPT／target-protected 诊断位，结构化记录对应 null，不是活跃训练 loss NaN；未隐藏这些文本。两个新行 13,320 个 hard-L 记录重算 CE0／同 logits CE01／原 KL，最大误差≤4.7684e−7；全部 U 函数均恢复。native supervised 总目标重算四行最大误差≤5.3645e−7。原 U 接受数会随权重变化，规则保持不等于接受样本数相同。

| 行 | 本轮属性 | 模型 epoch | 更新 | 原生两训练段墙钟秒 | 七视图预测墙钟秒 |
|---|---|---:|---:|---:|---:|
| B00 | 只读复用 | 30 | 6660 | 2146.190 | 48.089 |
| H00 | 新增 | 30 | 6660 | 2107.374 | 47.715 |
| B01 | 只读复用 | 30 | 6660 | 2860.909 | 49.650 |
| H01 | 新增 | 30 | 6660 | 2891.011 | 48.021 |

原生段墙钟包括该段原生训练／源健康等执行，不是纯 GPU kernel 时间；外部冻结源 panel 时间未单独记录，不能填造端到端速度。源独立分析 26.975677 秒，正式独立分析 12.557850 秒，均 0 NN。

每模型 L NN=5,114,880；native training＋validation identity NN=7,960,684（student train6,078,780、student eval918,004、teacher eval963,900、teacher train0）；冻结源 readout=12,600，source 合计7,973,284；正式预测含2次合成 smoke=1,176,002；合计9,149,286。新两模型共 **60 model-epoch／13,320 更新／10,229,760 L NN**，加首个共享 smoke1538后新增 identity NN **18,300,110**。复用控制历史 identity NN **18,298,572** 单列，不计成新成本；P43 原共享 smoke 不再新增分摊。相对匹配控制，0额外模型前向／teacher／head-fit，只有 CE 算术变化。固定预算耗尽不称收敛。

最终独立后态 VERIFIED：所属 dispatcher／两 worker／源分析／正式分析均自然退出；queue completed 两行、active/pending/failures 为空；两行 TRAINED_AND_SCORED、prediction truth_read=false、all_parameters_frozen=true、score truth_last=true。无技术重跑、热改配置、停止或修改其他 owner 的健康任务。见 [final_verification.json](final_verification.json)。

## 7. 命令、实际图表与未完成项

以下是已执行入口的相对路径等价命令；应在相应不可变 release CWD、原 CVS-RFFI Python 环境运行。完整绝对 argv、CWD、GPU、输出根已在 [experiment.json](experiment.json) 与执行记录固定。**当前 run 已完成，不得将这些命令对同一输出根重复执行。**

`python -u experiments/cvs_hard_lce45/runner.py`（自动 smoke→两 row 训练→冻结→逐 row 原七视图→truth-last score）

`python -u analysis/cvs_hard_LCE_E30_results_20261018.py --mode source --output ../../runs/20261018-phase1-FcrossRX-hardLCE-e30-s2026092701-m2-r01/source_analysis_r01 --wait-pids 961519 962003 962080`（原只读源分析，原样等待本 run PID）

`python -u analysis/cvs_hard_LCE_E30_results_20261018.py --mode formal --output ../../runs/20261018-phase1-FcrossRX-hardLCE-e30-s2026092701-m2-r01/final_analysis_r01 --source-dir ../../runs/20261018-phase1-FcrossRX-hardLCE-e30-s2026092701-m2-r01/source_analysis_r01 --truth ../../runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json`

正式分析使用已冻结 source-dir、固定 truth、不可覆盖的 final_analysis_r01。没有为本报告新增 NN 或预测。

实际本地产物根：[E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018)。完整逐 row／seed／RX／TX／day：[per_row_seed.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/final_analysis_r01/per_row_seed.csv)（504 行；seed_count=1）；源全部分层：[source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/source_analysis_r01/source_strata.csv)；全训练曲线数据：[training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/source_analysis_r01/training_curves.csv)。原 IQ、权重、逐步日志和 generated raw outputs 留在私有产物，不提交 Git。

八组实际 PNG/PDF 已生成并逐一视觉检查可读：

| 图 | 实际 PNG |
|---|---|
| 完整训练曲线 | [full_training_curves](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/source_analysis_r01/full_training_curves.png) |
| L CE0/CE01 同 logits 重算 | [hard_CE_value_recount](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/source_analysis_r01/hard_CE_value_recount.png) |
| 源状态／锁定／质量分箱 | [source_state_quality](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/source_analysis_r01/source_state_quality.png) |
| 源间隔与固定集合保持 | [source_margins_retention](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/source_analysis_r01/source_margins_retention.png) |
| 全七视图准确率 | [formal_accuracy](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/final_analysis_r01/formal_accuracy.png) |
| 损失×As 主效应／交互 | [formal_effects](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/final_analysis_r01/formal_effects.png) |
| 正式逐 RX mid/low | [formal_RX](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/final_analysis_r01/formal_RX.png) |
| 固定 cosine30 的正式间隔 | [formal_margins](E:/type10-7/local_artifacts/cvs_FcrossRX_hardLCE_E30_20261018/final_analysis_r01/formal_margins.png) |

本轮未完成的联合研究内容：有效 IQ Ag、组合 IQ 视图的独立价值、正式数据逐包物理状态因果分解、完整分支／门控训练轨迹、多模型 seed 重现、low urban>60% 与 suburban 距 clean≤2。P45 没有提供推进这些结论的充分证据，不将它标记为 goal complete。

本轮被证实的是 clean 监督保留组织下，完整 As 仍提供主要星地适应；被否定的是“仅去掉 L 标签平滑足以同时突破 clean 和 urban”的固定候选。后续保留 clean 监督份额，关闭本 L0 路线，不扫描平滑值／权重／父／预算／seed救跑。新的 IQ 候选必须由新的合法源域证据提出最小可证伪检验，不能从本轮正式微增倒推。已关闭的 FIR、噪声、课程与共享响应模型不重包装为新候选；共享响应的归一化先后归因由引用对话负责，本分支不重复启动。
