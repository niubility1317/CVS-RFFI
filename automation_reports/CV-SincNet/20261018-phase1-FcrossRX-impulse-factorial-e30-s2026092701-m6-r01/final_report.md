# P43：等功率噪声形态 × practical，E30 完整结果

固定 BG 候选失败并关闭；本轮 6 个模型和全部正式评价完成，整体“clean 正收益＋low urban>60%＋suburban 保持＋跨 seed 重现”目标未完成。

相对普通实际 As 的 B01，B11_BG 的 clean 下降 0.292857 个百分点、low urban 只增加 0.010714 个百分点（净多对 18/168000 个样本）、urban 三场景均值下降 0.018254 个百分点。不能把这一结果称为联合增强有效。源域否定结论在首次读取本轮正式成绩前已提交并推送：`dfa1b04c86408d4d5e114ec4c71869b584480eca`。

## 完整正式结果

所有行均为模型 seed `2026092701`，每个视图保留全部 168000 个固定样本；准确率单位为 %。每行全部 7 个视图预测固定后，独立 truth-last 评分。未改正式 IQ、信道、SNR、样本或默认分类头。

B00 保留共有旧普通增强、RDrop、跨 RX 同 TX SupCon 和原 U 机制，不能称为完全无增强或纯 CE；B10 加 Ag，B01 加 As，B11 为 Ag(raw) 与 As(raw) 的独立监督视图，没有执行 As(Ag(x))。WHITE 是同新增噪声功率、同前向控制，不作为失败后的新晋级候选。

| 行 | clean | high sub | mid sub | low sub | high urban | mid urban | low urban | urban 均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B00 | 85.247619 | 42.685119 | 40.995238 | 39.472619 | 41.439881 | 35.132738 | 30.139881 | 35.570833 |
| B10_WHITE | 85.185714 | 42.450595 | 40.855952 | 39.322619 | 41.298810 | 34.975595 | 30.080357 | 35.451587 |
| B10_BG | 85.216667 | 42.816667 | 41.168452 | 39.645238 | 41.594048 | 35.167262 | 30.153571 | 35.638294 |
| B01 | 85.341071 | 81.405357 | 78.661310 | 75.861905 | 77.767262 | 62.806548 | 51.244048 | 63.939286 |
| B11_WHITE | 85.432738 | 81.441667 | 78.689881 | 75.897024 | 77.786905 | 62.815476 | 51.308929 | 63.970437 |
| B11_BG | 85.048214 | 81.335714 | 78.591667 | 75.835119 | 77.699405 | 62.808929 | 51.254762 | 63.921032 |

宏平均 F1（%）：

| 行 | clean | practical_high | practical_mid | practical_low_suburban | practical_high_urban | practical_mid_urban | practical_low_urban |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B00 | 84.979636 | 39.311416 | 37.954813 | 36.758731 | 38.481030 | 33.446603 | 29.184589 |
| B10_WHITE | 84.905752 | 38.931129 | 37.691607 | 36.463415 | 38.232811 | 33.191244 | 29.052323 |
| B10_BG | 84.949305 | 39.387560 | 38.075542 | 36.853268 | 38.582418 | 33.419620 | 29.163212 |
| B01 | 85.044474 | 81.084230 | 78.359706 | 75.572741 | 77.423208 | 62.450463 | 50.887035 |
| B11_WHITE | 85.145974 | 81.125855 | 78.390515 | 75.603531 | 77.446364 | 62.453291 | 50.952890 |
| B11_BG | 84.739763 | 81.012393 | 78.283154 | 75.536312 | 77.351961 | 62.440678 | 50.891279 |

最差 RX 准确率（%）：

| 行 | clean | practical_high | practical_mid | practical_low_suburban | practical_high_urban | practical_mid_urban | practical_low_urban |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B00 | 78.450000 | 34.304167 | 33.520833 | 32.054167 | 34.000000 | 29.495833 | 25.750000 |
| B10_WHITE | 77.962500 | 33.366667 | 32.666667 | 31.120833 | 33.037500 | 28.766667 | 25.495833 |
| B10_BG | 77.966667 | 35.091667 | 34.295833 | 32.912500 | 34.808333 | 29.837500 | 26.025000 |
| B01 | 78.362500 | 71.908333 | 70.212500 | 67.741667 | 69.108333 | 56.458333 | 46.379167 |
| B11_WHITE | 78.437500 | 71.583333 | 69.920833 | 67.570833 | 68.816667 | 56.333333 | 46.329167 |
| B11_BG | 78.383333 | 71.241667 | 69.612500 | 67.304167 | 68.341667 | 56.095833 | 46.391667 |

[42 个 row/seed/view 完整总体指标 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/overall_by_row_seed.csv)；[756 个正式分层结果与混淆矩阵对应指标](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/per_row_seed.csv)。实际 7 个 RX 和 4 个采集日期均保留；756 个混淆矩阵已由固定预测和独立真值重新计算，与原 scorer 一致。[完整正式混淆矩阵与总体间隔 JSON](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/analysis_summary.json)；[1752 个源域分层矩阵 JSON](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/source_summary.json)。

## 收益归因与代价

以下均为当前六行内匹配的训练预算和前向成本差异，单位为百分点；不与历史四前向行冒充同总成本。

| 视图 | Ag 独立 BG−B00 | As 独立 B01−B00 | 联合 BG−B00 | Ag 给定 As | 交互项 B11−B10−B01+B00 |
| --- | --- | --- | --- | --- | --- |
| clean | -0.030952 | 0.093452 | -0.199405 | -0.292857 | -0.261905 |
| practical_high | 0.131548 | 38.720238 | 38.650595 | -0.069643 | -0.201190 |
| practical_mid | 0.173214 | 37.666071 | 37.596429 | -0.069643 | -0.242857 |
| practical_low_suburban | 0.172619 | 36.389286 | 36.362500 | -0.026786 | -0.199405 |
| practical_high_urban | 0.154167 | 36.327381 | 36.259524 | -0.067857 | -0.222024 |
| practical_mid_urban | 0.034524 | 27.673810 | 27.676190 | 0.002381 | -0.032143 |
| practical_low_urban | 0.013690 | 21.104167 | 21.114881 | 0.010714 | -0.002976 |

普通 As 相对 B00 的 low urban 增加 21.104167 个百分点、mid urban 增加 27.673810 个百分点、urban 均值增加 28.368452 个百分点，clean 增加 0.093452 个百分点。这是本 seed 的实际适应收益，不是新 Ag 的贡献或跨 seed 证明。

BG 独立增强的 clean 相对 B00 为 −0.030952 个百分点；联合 BG 相对 B00 为 −0.199405 个百分点。WHITE 联合相对 B01 的 low urban 仅 +0.064881、clean +0.091667 个百分点，源域判据和既有白噪声负历史不因此被正式成绩推翻，不按这些微小目标差异选择或扩种子。

B11_BG 对比 B01 的同物理样本修复与新增错误：

| 视图 | 修复 | 新增错误 | 净变化 | 准确率变化 pp |
| --- | --- | --- | --- | --- |
| clean | 428 | 920 | -492 | -0.292857 |
| practical_high | 767 | 884 | -117 | -0.069643 |
| practical_mid | 910 | 1027 | -117 | -0.069643 |
| practical_low_suburban | 1031 | 1076 | -45 | -0.026786 |
| practical_high_urban | 815 | 929 | -114 | -0.067857 |
| practical_mid_urban | 1254 | 1250 | 4 | 0.002381 |
| practical_low_urban | 1559 | 1541 | 18 | 0.010714 |

[全部 10 对比较 × 7 视图错误修复 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/paired_repairs.csv)；[全部因子效应 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/factorial_effects.csv)。low urban 的 1559 次修复几乎被 1541 次新增错误抵消，不能只报道修复数量。

B01 的三个 suburban 距自身 clean 为 3.935714 / 6.679762 / 9.479167 个百分点；BG 联合为 3.712500 / 6.456548 / 9.213095 个百分点，均未达预设 ≤2 的探索目标。BG 的 suburban 绝对准确率也下降，差距缩小主要来自 clean 降低，不是 suburban 突破。

## 源域判据、状态与头间隔

预登记 10 项源域条件独立复算与 launcher 一致：4 项通过、6 项失败。BG 独立 full V clean 相对 B00 为 −0.007407pp，联合相对 B01 为 −0.033333pp；2 draw×mid/low urban 中，BG−B01 四格为 [−0.111111,+0.055556,+0.222222,−0.111111]pp，0/4 达到 +1pp；相对 WHITE 仅 1/4 达到 +0.5pp。RX3/RX8 的联合 clean 相对 B00 下降 0.407407/0.370370pp。

通过的是 As 独立和含 Ag 时的 urban 适应条件、BG 相对 WHITE 的 clean 保持和源 low suburban 保持。固定 BG 的形态、概率、功率、权重、模型 seed 和预算不再调节救跑；不自动扩种子。完整源域结论：[source_decision.json](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01/source_decision.json)。

源 V 27,000 个样本用于 full clean 健康读数；独立源诊断面板为 V1800，每 TX×RX×day 20 个样本、2 个信道抽样，仍是 1800 个不同物理样本，不是 3600 个不同样本或 2 个模型 seed。它们来自已见源 RX，不是目标跨 RX 泛化证明。

low urban 的全部非空 state/lock 分层（2 draw 合并，每场景 3600 个观测，%）：

| 状态 | n | B00 | B01 | B11_BG |
| --- | --- | --- | --- | --- |
| LOS/locked | 1060 | 55.471698 | 94.716981 | 94.716981 |
| PARTIAL/locked | 307 | 46.579805 | 88.925081 | 89.250814 |
| PARTIAL/degraded | 253 | 41.501976 | 79.446640 | 78.260870 |
| PARTIAL/unlocked | 1 | 0.000000 | 0.000000 | 0.000000 |
| BLOCKED/degraded | 414 | 30.193237 | 42.028986 | 42.512077 |
| BLOCKED/unlocked | 1565 | 20.511182 | 30.351438 | 30.287540 |

源域分层显示：普通 As 主要改善 LOS/PARTIAL、locked 或仍有可用观测的退化；BLOCKED/unlocked 的 low urban 仍仅约 30.35%，新增 BG 未改善这一瓶颈。新增噪声比低于 3 dB 的区间仍很弱，高于 15 dB 的区间明显较好。该量为 signal_to_added_noise_db，不是原录音的总 SNR。正式 received IQ 没有逐包 state/lock 标签，不能把源域分层直接当作正式目标状态归因；也不能从这些相关性判定只有多径、频偏或门控负责。

本次没有共同的已训练父模型；保持率固定参照匹配 B00 FINAL30 的 clean 正确集合（143216 个样本），不是共同 E20 父模型。

| 行 | mid urban 保持率 % | low urban 保持率 % |
| --- | --- | --- |
| B00 | 38.619288 | 32.807089 |
| B01 | 68.782119 | 55.632751 |
| B11_BG | 68.807954 | 55.641129 |

[完整固定 B00 clean 正确集合保持率](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/fixed_B00_retention.csv)。所有头间隔统一除以固定 cosine scale30；完整总体均值和分位数见 analysis_summary.json，源状态间隔见 source_strata.csv。均值间隔、非零梯度、U 接受数均不替代分类收益。

本轮保存并分析了原骨干/总梯度、默认头 logits/间隔和旧 U 接受数量；未保存每包内部门控与各分支梯度轨迹，不能据此声称门控或某分支发生了特定因果变化。未训练身份或接收机诊断头。

## 实际增强、数据与信道

固定 F/SIG_TIME_X3、统一 Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、既有内部门控和参考响应、joint160 与原默认 cosine30 头；骨干和头按原流程联合更新。全部 5 个源 RX [1,3,4,6,8]、day1/2/3，L6300/U56700 隐藏/V27000，split392005，无 RX 留出、无 L_dev。U TX 真值未解封。原数据选择 equalized1、每包 RMS、center=false，在 dataset 处理后构造 raw/新视图；Ag 不再做波形归一化或剪裁，不反演原录音未知 TX 硬件。

Ag 为新增噪声 SNR30dB、每包噪声 RMS 精确匹配，WHITE 与 BG 使用同私有高斯创新，BG p=.02、激活方差为背景50倍，I/Q 共用每个复采样点 mask。私有 seed202610185301 加累计 step，不由 TX 或 RX 标签定义，跨 epoch 刷新，同一 RDrop 对复用同 IQ。真实6660步平均 relative noise RMS=0.0316227749；峰值点噪声能量占比 WHITE=2.392547%、BG=20.748355%。这证明算子实际不同且等能量，不证明 WiSig 存在该 EMI 分布或 Ag 有效。

As 从 raw IQ 生成，三个名义训练场景为 low_suburban/mid_urban/low_urban。实际 fs25MHz、fc2.462GHz、IQ256，residual/post_sync、equalization_enabled=false，因此 G=1，reference_snr_db=30，reflect 边界；接收质量适配、相位跟踪、IQ 补偿和 AGC 仍启用。质量阈值 locked_snr_db=15、unlocked_snr_db=5、quality_reference_snr_db=20、max_residual_scale=10，AGC 限制30dB；phase_tracking_residual_std_deg=1、相关时间1ms，receiver_cfo_std_hz=200、residual_cfo_std_hz=100、coarse_doppler_error_std_hz=500，IQ估计幅度/相位标准差为0.05dB/0.2°（每包实现仍受接收状态适配，不把名义参数当固定观测值）。分数延迟 FIR half-width24 存在；MMSE129/delay64 的实现未因 EQoff 被启用。源诊断旧缓存 fc2.45GHz 是披露的源代理，正式测试配置和 IQ 保持原样。

实际 leo_practical 从已钉住的 `cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/` 导入，而非根 code/；全部实际 __file__ 和接收处理配置已取回。RX 硬件按原 RX/day/eq session 稳定，不由 TX 定义，所有合法有标签 As 均参与 CE，不要求 urban 学生先高置信。

每个 As 行累计426240个真实信道观测，全部L6300都获得过As；6TX×5RX×3day×3scene×3state共810格全部非空，最小75次。三行命名信道流、逐批scene/state与原生汇总一致；3个As-off控制实际信道调用0。覆盖复算只读有标签L元数据，没有解码U标签、读取IQ或额外模型前向。

[全部 810 格逐行计数](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/state_coverage_r01/TX_RX_day_scene_state.csv)；[逐 epoch 状态支持](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/state_coverage_r01/epoch_TX_RX_state.csv)；[实际原生参数示例](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/runtime_metadata_r01/B11_BG-s2026092701/source/actual_native_args.json)；[实际接收处理配置](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/runtime_metadata_r01/B11_BG-s2026092701/source/actual_channel.json)；[真实模块路径](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/runtime_metadata_r01/B11_BG-s2026092701/source/actual_modules.json)。12个分段共84份运行配置/来源/冻结元数据均在runtime_metadata_r01中。

日志/配置有继承的描述字段：`concat/L_forwards=4`、旧`training_mechanism.L_forwards_per_sample=4`和方法名称来自原四前向底座；实际本轮L_NN_per_step=768、reference_steps+shape_steps和encoder/result均为6前向，As-off真实计数为0。不能以这些旧描述或第一原CE代替完整L总目标。真实目标为.5commonRD+.5mixedRD+原SupCon+.25shapeRD，L总系数1.25；完整六行数值已独立复算通过。

## 预算、环境和必要验证

每行从零 own20（13L+7U）再自身 own10（10U），L 在全部30轮都训练；own20包含在30内，没有加载历史模型作训练父模型。普通增强实际E13冻结，自身student/EMA严格恢复后重置AdamW/cos10；lr2e-4、weight_decay1e-4、label_smoothing=.01、RDrop alpha1，原SupCon仅ordinary128、两深度×两已有forward，temp.2/max.1/warm5/ramp10；没有新增U前向或伪标签门控。原默认头与骨干共同训练。

每行6660更新、L网络包5114880，其中新Ag或raw控制槽1704960NN；L物理观察852480，原U观察963900。原先4L前向历史行的L计算量更少，本次每步多50%的L前向成本，仅当前6行严格匹配总预算。

每行原生student_trainNN6078780、student_evalNN918004、teacher_evalNN963900、teacher_trainNN0，冻结源读出12600NN，source总7973284NN；正式推理1176002NN（1176000正式样本＋2既有检查前向），每模型总9149286NN。6模型共180 model-epoch、39960更新，加共享真实CP/合成smoke4610NN，全部identity NN=54900326。外部P41模型只作2个合成无query smoke前向，绝未用作新训练初始化；其历史训练不冒充当前父训练。

| 行 | 原生训练＋验证墙钟 秒 | 正式预测墙钟 秒 |
| --- | --- | --- |
| B00 | 2146.190 | 48.089 |
| B10_WHITE | 2153.091 | 47.099 |
| B10_BG | 2203.393 | 48.988 |
| B01 | 2860.909 | 49.650 |
| B11_WHITE | 2887.771 | 48.928 |
| B11_BG | 2884.087 | 47.180 |

各模型并行执行，上表不能相加冒充矩阵总墙钟；外部冻结源读出墙钟没有独立记录。完整源分析42.565秒、正式独立复算19.133秒、元数据状态覆盖270.536秒、新覆盖图4.579秒，均0NN。预算耗尽不等于已收敛，未追加轮次。

N607普通账号原环境Python3.10.19、PyTorch2.1.0、NumPy2.2.5、Matplotlib3.10.8，训练GPU0-5、分析CUDA为空/CPU线程2。没有修改其他owner的健康任务、包或系统配置。原dispatcher和6worker以及分析进程均自然结束，队列6完成、0失败。

必要验证：原8项聚焦算子/协议负例PASS；唯一独立P0/P1 PASS；launcher首步真实已核实同源CP无query smoke2NN及6个真实F合成梯度/计数4608NN PASS后立即训练；单训练归档SHA匹配、remotecompile、12分段config/args核对和PID/CWD/argv/GPU/loggrowth均已保留。分析7项合成检查本地/N607均PASS；六行180epoch/39960steps完整源日志和序列复算，实际L目标误差小于阈值；756个正式混淆矩阵独立一致，1752个源域分层矩阵保存。

已修正并保留的技术负记录：本地真值隔离测试桩使用正斜线后缀，改为Path.name；分析r01包漏criteria.py参考文件，编译通过但导入检查失败，r02补齐后7项PASS。没有训练行失败、重跑或因观察超时重启。正式分析PID929309在首次post-state前已完成，用完整completion/execution产物独立核实，没有为捕捉PID重复运行。

## 文件、命令与图表

训练执行代码提交：`19aedc3604fed9c193237dd92366164178af0983`；完整分析：`c02154d6cf9fb29fafecc4145bb6c08cb26d2490`；源域否定提交：`dfa1b04c86408d4d5e114ec4c71869b584480eca`；完整状态/覆盖绘图：`d315b28b5229a335bbfdd7e9be7a4900e69f4e84`，均已推送并独立比较远端OID一致。最新报告提交见本轮交付。

主要修改：[IQ算子](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_shape_factorial43/noise.py)、[配对训练hook](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_shape_factorial43/model.py)、[固定六行配置](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_shape_factorial43/design.py)、[启动/自动评价](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_shape_factorial43/dispatch.py)、[逐行正式评价](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_shape_factorial43/evaluate.py)、[真实计数](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_shape_factorial43/observer.py)、[源域判据](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_shape_factorial43/criteria.py)、[8项检查](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_shape_factorial43/checks.py)；[完整源/正式分析](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_impulse_factorial_E30_results_20261018.py)、[7项分析检查](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_impulse_factorial_analysis_checks_20261018.py)、[元数据覆盖](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_impulse_factorial_coverage_20261018.py)、[完整覆盖图](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_impulse_factorial_plot_coverage_20261018.py)。训练包共17个新文件；没有修改骨干结构或分类头。

历史实际训练命令（输出已存在，不能覆盖重跑）：

```text
CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_impulse_factorial_E30_20261018_r01
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_impulse_factorial_E30_20261018_r01/experiments/cvs_noise_shape_factorial43/runner.py
```

[完整预登记与精确执行记录](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01/experiment.json)；[分析/覆盖执行命令](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/analysis_launch_r02.json)；[正式独立复算命令](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_launch.json)。本地执行检查为 `E:/type10-7/local_envs/ssr-gpu/python.exe E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_impulse_factorial_analysis_checks_20261018.py`；N607使用同分析release内该文件。训练入口为runner.py，先smoke后继续，不使用smoke许可令牌。

下列PNG/PDF均实际生成并已查看，未生成或承诺不存在的图：

- 全部正式准确率与Ag效应：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/formal_factorial_accuracy.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/formal_factorial_accuracy.pdf)

- 全部7RX/7视图：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/formal_all_RX_accuracy.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/formal_all_RX_accuracy.pdf)

- 交互项与suburban差距：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/factorial_interaction_and_suburban_gap.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/final_analysis_r01/factorial_interaction_and_suburban_gap.pdf)

- 全部180训练epoch：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/training_curves.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/training_curves.pdf)

- 实际噪声/骨干梯度/学习率：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/noise_shape_and_gradient_curves.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/noise_shape_and_gradient_curves.pdf)

- 源新增噪声比与准确率：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/source_added_noise_accuracy.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/source_added_noise_accuracy.pdf)

- 全部非空源状态/锁定组合，含n=1/2：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_coverage_figures_r01/source_all_nonempty_state_lock.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_coverage_figures_r01/source_all_nonempty_state_lock.pdf)

- 全部TX×RX×传播状态曝光：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_coverage_figures_r01/all_TX_RX_state_exposure.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_coverage_figures_r01/all_TX_RX_state_exposure.pdf)

- 逐轮状态支持：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_coverage_figures_r01/all_state_support_progress.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_coverage_figures_r01/all_state_support_progress.pdf)

- 原生RX×场景曝光：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/training_As_support.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/training_As_support.pdf)

旧5个常见状态组合图保留于source_analysis_r01/source_state_lock_accuracy.*；新图完整显示6个非空组合，所有源观测均计入总体和分层CSV，未删除稀有或困难样本。全部训练曲线CSV：[training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/training_curves.csv)；源域全部分层：[source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/source_analysis_r01/source_strata.csv)。

## 负结果与未完成项

已证实：当前普通As能显著改善同seed的固定practical识别；BG/WHITE确实等新增能量而形态不同，且所有训练状态有覆盖。已否定：本固定BG候选能够达到预登记clean与联合urban收益、优于等能量WHITE。缺少证据：任何新Ag跨RX clean有效、low urban>60%、每个suburban距clean≤2pp、跨模型seed重现、以及具体分支或内部门控的训练轨迹归因。

仅1个模型seed；2个源信道draw不是seed复现。按源判据不扩种子，因此跨seed结论为未验证，不虚构标准差或置信区间。历史benchmark已暴露，本次不是全新独立盲测；本轮源判定在读取本轮正式成绩前固定，不允许目标成绩回流调参、选模、晋级或选择性重跑。

下一步先复用已有合法源域11条件损伤控制（full；no_noise/flat_H/no_carrier/no_RX_IQ各fixed/total；实际功率质量与边界），不再重复同一冻结probe。本轮已核实其真实配置：全5源RX、L6300/U56700/V27000、V1800、无heldRX/L_dev、0更新/0头拟合/0target；旧E20 B01/RAW_PRAC在去新增噪声的fixed控制下4个urban格均超过+5pp，其余所列去损伤控制未同时达该标准。该旧冻结干预不等于重训因果效应，也不证明继续随机噪声会有效。其元数据已取回用于下一步分层误差复盘，当前没有登记新的训练或变相调参。

本轮可执行实验与交付已完成；整体goal保持ACTIVE，不能因一次科学负结果或预算用尽标记目标达成。

后续结构与 RX session 的合成审计见 [postmortem_audit.md](postmortem_audit.md)；没有新训练或新性能成绩，不改变本报告的负结果。
