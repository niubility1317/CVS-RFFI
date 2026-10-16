# 孤立 AWGN(raw)：额外 raw／ordinary 的三匹配 E20

**ANALYZED／VERIFIED，FAIL_FIXED_NATIVE_AWGN_20_36_FULL_COVERAGE_TRAINING_CLOSED。** [原始预登记](preregistration.md)与[实际矩阵／配置](experiment.json)保留。三组均为F＋跨RX同TX、3SIG_TIME／统一Sinc／joint160／NoExplicitCFO／原cosine30默认头，正常联合训练、scratchE20／13L+7U／4440更新。没有父模型初始化、另训头或部署分支。

结果只属于 sourceRX1-held／seed2026092701。训练fit RX3/4/6/8、L5040/U45360/V21600；held原L1260，day1/2/3、6TX。不混同fullL6300或L_train5580/L_dev720，不构造target或正式truth评分。原U真值隐藏、旧gate/自身EMA/原增强保持，没有新增U/teacher视图。三组实际L/U物理训练序列相同，nativeargs仅candidate_id/output_dir不同。

## 真实身份识别

| 场景 | B00 accuracy % | C_ordinary accuracy % | B10 accuracy % | B10−B00 pp | B10−C_ordinary pp |
|---|---:|---:|---:|---:|---:|
| clean | 79.682540 | 81.111111 | 80.634921 | +0.952381 | -0.476190 |
| practical_high | 38.571429 | 35.396825 | 41.587302 | +3.015873 | +6.190476 |
| practical_mid | 39.206349 | 35.476190 | 42.857143 | +3.650794 | +7.380952 |
| practical_low_suburban | 36.349206 | 33.015873 | 37.857143 | +1.507937 | +4.841270 |
| practical_high_urban | 38.174603 | 34.920635 | 40.793651 | +2.619048 | +5.873016 |
| practical_mid_urban | 32.222222 | 30.555556 | 34.047619 | +1.825397 | +3.492063 |
| practical_low_urban | 27.301587 | 27.301587 | 28.253968 | +0.952381 | +0.952381 |

每场景同一1260源物理包，完整accuracy／macro-F1／混淆矩阵、配对修复新错和全部TX/day分层见[source_analysis.json](source_analysis.json)。这不是正式target成绩，七场景重复相同物理包不冒充8820独立新样本。

| 对照 | clean macro-F1变化 pp | clean修复／新错 | lowurban修复／新错 | seenV变化 pp |
|---|---:|---|---|---:|
| B10−B00 | +1.218630 | 26/14 | 40/28 | +0.018519 |
| B10−C_ordinary | -0.356864 | 21/27 | 60/48 | -0.041667 |

| 原固定条件 | 结果 |
|---|---|
| B00_clean | FAIL |
| B00_clean_macroF1 | PASS |
| B00_each_practical | PASS |
| B00_seen_V | PASS |
| C_ordinary_clean | FAIL |
| C_ordinary_clean_macroF1 | FAIL |
| C_ordinary_each_practical | PASS |
| C_ordinary_seen_V | PASS |
| healthy | PASS |

候选须相对两个控制同时clean≥+1pp、clean macro-F1不降、每practical及seenV≥−.5pp；健康要求不能代替识别条件。上述判据没有根据最终分数调整。

| 行 | high suburban−ownclean绝对差 pp | mid suburban−ownclean绝对差 pp | low suburban−ownclean绝对差 pp |
|---|---:|---:|---:|
| B00 | 41.111111 | 40.476190 | 43.333333 |
| C_ordinary | 45.714286 | 45.634921 | 48.095238 |
| B10 | 39.047619 | 37.777778 | 42.777778 |

suburban探索目标仍为每场景距自身clean≤2pp，不能通过降低clean制造接近；同时给出各场景绝对accuracy和相对控制的变化。

## 实际训练组织与算子

共同完整base：raw.75＋旧ordinary.25＋原ordinary-only浅／中层CROSS_RX SupCon，原RDrop1／LS.01及U/EMA规则。各组额外.25 R(view)，R=同一IQ两次随机网络的平滑CE均值＋within-IQ RDrop KL。B00用raw、C_ordinary复用同一已生成ordinary、B10用独立native AWGN(raw)；均8L前向，总pair权重1.25，raw绝对监督份额稳定、相对占比变化如实报告。不存在跨视图一致性或新损失，也未把RDrop称为IQ增强。

B10调用原native propercomplex AWGN，各包uniform[20,36]dB、按raw包功率，private seed202610172301+step，fork CPU及当前CUDA状态并恢复。CPU/GPU同native分布，不声称抽样相同。没有MP/PA/CFO/phase/shift/DAC/量化/clip/RMS恢复/过滤/重抽。算子不读标签/RX/day/置信度；source训练全部L均参与额外监督。名义SNR有界，高斯幅度不严格有界，完整实际能量和尾部保留；新增噪声SNR不是录音总SNR，数字IQ变换不是替换真实RX。

旧ordinary仍为共同原配方，实际args/base/调用路径保存，并未声称各组完全无增强。C_ordinary额外前向不重抽普通增强；每步其extra_raw_MSE与common ordinary_raw_MSE逐项核对一致。当前训练As为0，读出保留原完整六practical；测试传播、SNR、补偿、received IQ及困难样本均未改。本轮不是Ag/As四格，不计算虚假的最终交互项。

## 全量复算与成本

独立reader复算全部26460硬判决的accuracy、6类macro-F1、混淆矩阵、TX/day分层和相对两个控制的修复／新错、suburban差距。原readout只存硬判决，CE保留运行时默认头报告，未冒称独立重算logit CE。全部13320native/view/Sincunit记录、60epoch日志／CSV、L/U物理序列、raw/ordinary/extra CE与KL、SupCon系数、总损失及BN计数已核对；native final_L_CE是第一ordinary平滑CE，不是完整目标。

568320份B10 noise的名义SNR、raw/noise/output RMS、相对IQ变化、actual added-noise SNR、功率变化与交叉项全部有限并检查功率恒等式，不挑尾部。运行时72个TX×RX×day格全部覆盖，RX/day计数由实际L序列独立复算；逐step TX未单独存储，因此不声称从逐包TX列表再次复算全部cell分配。完整分布见聚合JSON。

| 行 | native ordinary CE | common base目标 | extra pair目标 | 实际 NN 包前向 | 训练秒 |
|---|---:|---:|---:|---:|---:|
| B00 | 0.26407478 | 0.31094601 | 0.09732051 | 5823318 | 1393.017 |
| C_ordinary | 0.24278840 | 0.31115677 | 0.27673898 | 5823318 | 1464.055 |
| B10 | 0.26530390 | 0.30940691 | 0.10977747 | 5823318 | 1506.563 |

三组60model-E／13320更新，总实际NN 17469956，含dispatcher真实CP合成smoke2；每行LNN4546560，其中extra1136640，旧ordinary568320。仅B10 noise568320，控制noise0，As0。原U／EMA／V及8820held每row单列；teacher仪表扣除继承初始counter2/row，未把重复计数当额外前向。

B00 student/teacher仪表：{"student_train_forward_packets": 4942928, "student_eval_forward_packets": 475202, "teacher_train_forward_packets": 0, "teacher_eval_forward_packets": 396370}，扣除teacher offset2，held8820。

C_ordinary student/teacher仪表：{"student_train_forward_packets": 4942928, "student_eval_forward_packets": 475202, "teacher_train_forward_packets": 0, "teacher_eval_forward_packets": 396370}，扣除teacher offset2，held8820。

B10 student/teacher仪表：{"student_train_forward_packets": 4942928, "student_eval_forward_packets": 475202, "teacher_train_forward_packets": 0, "teacher_eval_forward_packets": 396370}，扣除teacher offset2，held8820。

本地工程4114合成NN／3合成联合更新／640native noise／128ordinary、0realIQ/CP；G0源检验50402NN／0更新；两参照G0历史训练40model-E／8880更新／9373356NN，均单列，不混入当前scratch训练。无另训头、续训或新增教师分支。

一个普通账户预检／archive传输SHA比较／remotecompile与真实CP两合成包no-querysmoke PASS，首项PASS立即继续；每row真实PID/CWD/argv/GPU及存活loggrowth已独立观察。三worker与dispatcher自然退出，无技术failure、重启或健康旧进程干预。一次独立P0/P1审查PASS，本地28负例PASS，没有追加通用审批或重复审查。

执行commit 06ebf579c87c834d0ebc14856212e3fa97619d30。分析／绘图0源IQ/NN/更新；完整曲线、七场景／gap、TX/day分层图保存在 local_artifacts/F_isolated_AWGN_E20_20261017/final/，PNG实际查看后交付。原IQ／checkpoint／物理ID／逐包预测／全日志不提交Git。

## 判定与下一步

固定20至36dB／全覆盖的额外孤立AWGN训练方案未通过原条件，关闭；不抬SNR、改范围／系数、clip／删尾／重抽、按RX/TX/day条件或换seed/RX/预算救跑。G0身份保持没有自动转化为未见RX识别收益，前向和健康通过不能替代性能条件。

完整联合目标ACTIVE：仍需有效Ag的跨RX身份提取、稳定As、同完整契约B00/B10/B01/B11及交互项、跨seed与正式clean+六practical独立truth-last，以及clean净收益／lowurban>60%／suburban接近自身clean同时成立。E20来自历史测试基准已暴露的压缩探索，不声称新独立盲测。


## 收益集中与错误转移

noise相对raw的clean净增加12个正确判决，其中TX2净增加19个，其他5个TX合计净减少7个；相对ordinary的clean净减少6个。六practical的accuracy变化为正，但不能据此称所有TX都受益，lowurban相对ordinary的macro-F1仍下降0.295357pp。完整按TX净正确数见[TX贡献](TX_net_correct_contributions.json)，全部同物理包四组clean/practical错误转移见[错误转移分析](error_transitions.md)及其JSON；不以条件集合筛选、删样本或改变原判据。

| 场景 | B10−raw，TX0..5各自净正确数 | 总净正确数 |
|---|---|---:|
| clean | -1／-7／19／-1／0／2 | 12 |
| practical_high | 1／0／49／-3／-9／0 | 38 |
| practical_mid | 0／0／45／-1／2／0 | 46 |
| practical_low_suburban | -3／-1／38／-13／0／-2 | 19 |
| practical_high_urban | -2／0／39／-1／-3／0 | 33 |
| practical_mid_urban | -4／-2／28／-3／2／2 | 23 |
| practical_low_urban | -2／-2／9／0／8／-1 | 12 |

lowurban在双方clean均正确的固定990包集合内，noise相对raw修复28／新错18，净+10；其余组净+2，整体净+12。相对ordinary的双方clean正确995包集合，修复38／新错36，仅净+2；其余组净+10。条件分析仅描述原固定源结果，不证明传播原因或新的星地机制，也不抵消clean条件FAIL。

与历史同配方raw控制（sameTXIQmix矩阵中的B00）相比，当前七视图的物理ID和标签完全相同，但预测差异分别为clean13／highsub36／midsub42／lowsub30／highurban34／midurban34／lowurban41；actual nativeargs只有output_dir/run_id差异。现有证据未单独识别跨运行数值差异的来源，不能声称精确复现。本次所有增量均使用当前同矩阵控制，没有以历史分数替代、择优或重跑当前控制。

这里的失败包含两点：相对raw的clean +0.952381pp未到原+1pp；相对ordinary的clean−0.476190pp及F1−0.356864pp，失败并非只由前一个边界差距导致。固定方案关闭，不把ordinary控制或TX2子群事后晋级成替代候选。

## 已查看的完整图表

- [完整20轮训练曲线 PNG](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_isolated_AWGN_E20_20261017/final/complete_training_curves.png)；[PDF](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_isolated_AWGN_E20_20261017/final/complete_training_curves.pdf)。
- [七场景绝对准确率与ownclean差距 PNG](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_isolated_AWGN_E20_20261017/final/held_source_results.png)；[PDF](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_isolated_AWGN_E20_20261017/final/held_source_results.pdf)。
- [全部TX/day相对两个控制的分层变化 PNG](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_isolated_AWGN_E20_20261017/final/source_strata.png)；[PDF](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_isolated_AWGN_E20_20261017/final/source_strata.pdf)。

三个PNG均已实际查看；曲线标题排版第二版仅重绘聚合数据，保留原图版和新图版，不重复源IQ、权重加载、模型前向或训练。联合目标仍ACTIVE。


下一唯一待检验假设与最小source-only依据分析见[相位曲率源依据](../../../analysis/cvs_phase_curvature_source_basis_20261017.md)。目前仅说明、尚未实现登记启动；不重开旧commonphase／phase_curvature特征模块或AWGN方案，不声称已找到可用Ag。
