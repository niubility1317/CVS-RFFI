# 有界非恒定相位(raw)：额外 raw／ordinary 的三匹配 E20

**ANALYZED／VERIFIED，FAIL_FIXED_NATIVE_BOUNDED_NONAFFINE_PHASE_006_01_FULL_COVERAGE_TRAINING_CLOSED。** [原始预登记](preregistration.md)与[实际矩阵／配置](experiment.json)保留。三组均为F＋跨RX同TX、3SIG_TIME／统一Sinc／joint160／NoExplicitCFO／原cosine30默认头，正常联合训练、scratchE20／13L+7U／4440更新。没有父模型初始化、另训头或部署分支。

结果只属于 sourceRX1-held／seed2026092701。训练fit RX3/4/6/8、L5040/U45360/V21600；held原L1260，day1/2/3、6TX。不混同fullL6300或L_train5580/L_dev720，不构造target或正式truth评分。原U真值隐藏、旧gate/自身EMA/原增强保持，没有新增U/teacher视图。三组实际L/U物理训练序列相同，nativeargs仅candidate_id/output_dir不同。

## 真实身份识别

| 场景 | B00 accuracy % | C_ordinary accuracy % | B10 accuracy % | B10−B00 pp | B10−C_ordinary pp |
|---|---:|---:|---:|---:|---:|
| clean | 80.634921 | 81.428571 | 80.079365 | -0.555556 | -1.349206 |
| practical_high | 39.126984 | 35.634921 | 39.841270 | +0.714286 | +4.206349 |
| practical_mid | 39.126984 | 35.793651 | 40.079365 | +0.952381 | +4.285714 |
| practical_low_suburban | 35.714286 | 33.492063 | 36.904762 | +1.190476 | +3.412698 |
| practical_high_urban | 38.174603 | 35.634921 | 39.126984 | +0.952381 | +3.492063 |
| practical_mid_urban | 32.301587 | 30.476190 | 32.222222 | -0.079365 | +1.746032 |
| practical_low_urban | 27.857143 | 27.142857 | 27.698413 | -0.158730 | +0.555556 |


| 场景 | B00 macro-F1 % | C_ordinary macro-F1 % | B10 macro-F1 % | B10−B00 pp | B10−C_ordinary pp |
|---|---:|---:|---:|---:|---:|
| clean | 80.327323 | 81.037343 | 79.734219 | -0.593104 | -1.303124 |
| practical_high | 28.736903 | 27.576773 | 28.671768 | -0.065135 | +1.094995 |
| practical_mid | 29.406604 | 28.676542 | 30.233207 | +0.826603 | +1.556666 |
| practical_low_suburban | 27.945410 | 27.923360 | 29.028984 | +1.083575 | +1.105625 |
| practical_high_urban | 29.147531 | 28.525983 | 29.391255 | +0.243724 | +0.865271 |
| practical_mid_urban | 26.706456 | 26.439453 | 26.666706 | -0.039750 | +0.227254 |
| practical_low_urban | 24.550875 | 24.574575 | 24.292384 | -0.258491 | -0.282191 |

每场景同一1260源物理包，完整accuracy／macro-F1／混淆矩阵、配对修复新错和全部TX/day分层见[source_analysis.json](source_analysis.json)。这不是正式target成绩，七场景重复相同物理包不冒充8820独立新样本。

| 对照 | clean macro-F1变化 pp | clean修复／新错 | lowurban修复／新错 | seenV变化 pp |
|---|---:|---|---|---:|
| B10−B00 | -0.593104 | 7/14 | 13/15 | +0.004630 |
| B10−C_ordinary | -1.303124 | 7/24 | 38/31 | -0.060185 |

| 原固定条件 | 结果 |
|---|---|
| B00_clean | FAIL |
| B00_clean_macroF1 | FAIL |
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
| B00 | 41.507937 | 41.507937 | 44.920635 |
| C_ordinary | 45.793651 | 45.634921 | 47.936508 |
| B10 | 40.238095 | 40.000000 | 43.174603 |

suburban探索目标仍为每场景距自身clean≤2pp，不能通过降低clean制造接近；同时给出各场景绝对accuracy和相对控制的变化。

## 实际训练组织与算子

共同完整base：raw.75＋旧ordinary.25＋原ordinary-only浅／中层CROSS_RX SupCon，原RDrop1／LS.01及U/EMA规则。各组额外.25 R(view)，R=同一IQ两次随机网络的平滑CE均值＋within-IQ RDrop KL。B00用raw、C_ordinary复用同一已生成ordinary、B10用独立bounded nonaffine phase(raw)；均8L前向，总pair权重1.25，raw绝对监督份额稳定、相对占比变化如实报告。不存在跨视图一致性或新损失，也未把RDrop称为IQ增强。

B10的全256点算子与G0相同：CPU Float64 iidGaussian增量sigma.006，cumsum后正交去常数/线性项，再整路径收缩到peak≤.1rad，全包乘exp(j theta)，回到原Float32与输入设备。私有seed202610173401+step，只fork CPU生成器并恢复全局CPU/Python/CUDA随机流；CPU构造、往返IQ传输和统计walltime另记。所有路径保留，无逐点clip、重抽、删尾或标签/RX/day/置信度条件。没有新增AWGN/MP/PA/CFO/常相位/时移/DAC/量化/后RMS。旧ordinary本来含Wiener，这只是独立全覆盖有界去仿射视图，不是新增强原理；也不把其分布称为未修改Wiener或实际RX替换。

旧ordinary仍为共同原配方，实际args/base/调用路径保存，并未声称各组完全无增强。C_ordinary额外前向不重抽普通增强；每步其extra_raw_MSE与common ordinary_raw_MSE逐项核对一致。当前训练As为0，读出保留原完整六practical；测试传播、SNR、补偿、received IQ及困难样本均未改。本轮不是Ag/As四格，不计算虚假的最终交互项。

## 全量复算与成本

独立reader复算全部26460硬判决的accuracy、6类macro-F1、混淆矩阵、TX/day分层和相对两个控制的修复／新错、suburban差距。原readout只存硬判决，CE保留运行时默认头报告，未冒称独立重算logit CE。全部13320native/view/Sincunit记录、60epoch日志／CSV、L/U物理序列、raw/ordinary/extra CE与KL、SupCon系数、总损失及BN计数已核对；native final_L_CE是第一ordinary平滑CE，不是完整目标。

568320份B10相位观察的RMS/峰值/均值/线性项/整路径缩放、输入输出RMS与相对功率、全IQ和参考窗RMS变化、模长误差、参考零功率标志全部保存并复算有限性、界限和功率关系，未挑路径或尾部。运行时72个TX×RX×day格全部覆盖，RX/day计数由实际L序列独立复算；逐step TX未单独存储，因此不声称再次复算全部逐包TX分配。本轮训练日志保留每包汇总而非所有256点路径，完整路径核对属于此前G0，不冒称重新从原IQ或全相位轨迹做本轮复算。完整分布与生成/传输walltime见聚合JSON。

| 行 | native ordinary CE | common base目标 | extra pair目标 | 实际 NN 包前向 | 训练秒 |
|---|---:|---:|---:|---:|---:|
| B00 | 0.26382983 | 0.31063154 | 0.09715993 | 5823318 | 1388.089 |
| C_ordinary | 0.24313702 | 0.31099806 | 0.27694581 | 5823318 | 1434.133 |
| B10 | 0.26384267 | 0.30989098 | 0.09865627 | 5823318 | 1409.290 |

三组60model-E／13320更新，总实际NN 17469956，含dispatcher真实CP合成smoke2；每行LNN4546560，其中extra1136640，旧ordinary568320。仅B10 phase568320，控制phase0，As0。原U／EMA／V及8820held每row单列；teacher仪表扣除继承初始counter2/row，未把重复计数当额外前向。

B00 相位构造/传输/统计walltime 0.0秒；student/teacher仪表：{"student_train_forward_packets": 4942928, "student_eval_forward_packets": 475202, "teacher_train_forward_packets": 0, "teacher_eval_forward_packets": 396370}，扣除teacher offset2，held8820。

C_ordinary 相位构造/传输/统计walltime 0.0秒；student/teacher仪表：{"student_train_forward_packets": 4942928, "student_eval_forward_packets": 475202, "teacher_train_forward_packets": 0, "teacher_eval_forward_packets": 396370}，扣除teacher offset2，held8820。

B10 相位构造/传输/统计walltime 24.94264426175505秒；student/teacher仪表：{"student_train_forward_packets": 4942928, "student_eval_forward_packets": 475202, "teacher_train_forward_packets": 0, "teacher_eval_forward_packets": 396370}，扣除teacher offset2，held8820。

本地工程4114合成NN／3合成联合更新／640合成phase／128ordinary、0realIQ/CP；G0源检验50402NN／0更新；两参照G0历史训练40model-E／8880更新／9373356NN，均单列，不混入当前scratch训练。无另训头、续训或新增教师分支。

一个普通账户预检／archive传输SHA比较／remotecompile与真实CP两合成包no-querysmoke PASS，首项PASS立即继续；每row真实PID/CWD/argv/GPU及存活loggrowth已独立观察。三worker与dispatcher自然退出，无技术failure、重启或健康旧进程干预。一次独立P0/P1审查PASS，本地28负例PASS，没有追加通用审批或重复审查。

执行commit 51fc25d6862b7773e8d67fcc95a852c2e52ecc86。分析／绘图0源IQ/NN/更新；完整曲线、七场景／gap、TX/day分层图保存在 local_artifacts/F_bounded_nonaffine_phase_E20_20261017/final/，三个PNG已实际查看。原IQ／checkpoint／物理ID／逐包预测／全日志不提交Git。

## 判定与下一步

固定sigma.006／peak.1rad／全覆盖的额外孤立phase训练方案未通过原条件，关闭；不改sigma／峰值／范围／系数、clip／删尾／重抽、按RX/TX/day条件或换seed/RX/预算救跑。G0身份保持没有自动转化为未见RX识别收益，前向和健康通过不能替代性能条件。

完整联合目标ACTIVE：仍需有效Ag的跨RX身份提取、稳定As、同完整契约B00/B10/B01/B11及交互项、跨seed与正式clean+六practical独立truth-last，以及clean净收益／lowurban>60%／suburban接近自身clean同时成立。E20来自历史测试基准已暴露的压缩探索，不声称新独立盲测。

## 收益分布与结论边界

这不是接近阈值的纯边界失败：B10的clean accuracy和macro-F1相对两个控制均实际下降。mid/low urban相对raw分别−0.079365/−0.158730pp，low urban macro-F1相对raw/ordinary分别−0.258491/−0.282191pp。相对ordinary的六场景accuracy均上升，仍不足以证明联合目标或完整身份收益。

| 场景 | TX0净正确数 | TX1 | TX2 | TX3 | TX4 | TX5 |
|---|---:|---:|---:|---:|---:|---:|
| clean | -5 | -1 | -5 | +1 | +0 | +3 |
| practical_high | +0 | +0 | +11 | -4 | +3 | -1 |
| practical_mid | +3 | +0 | +14 | -6 | -1 | +2 |
| practical_low_suburban | +2 | +2 | +12 | -7 | +5 | +1 |
| practical_high_urban | -2 | +0 | +11 | -1 | +4 | +0 |
| practical_mid_urban | -1 | +0 | +0 | -6 | +4 | +2 |
| practical_low_urban | -1 | +0 | +3 | -4 | +2 | -2 |

相对raw，前三个suburban的TX2净改善分别11/14/12个正确样本，同时TX3减少4/6/7；不能把总场景小幅提高解释为所有身份共同变稳健。clean修复7、新错14，净−7。low urban总净−2，其中双方clean均正确的1002包中修复10/新错10、净0；该描述不作为挑包或新门槛。完整四组和预测变化流见[错误转移](error_transitions.md)。

B10 suburban与自身clean差距仍为40.238095/40.000000/43.174603pp，离2pp探索目标很远。相对raw的差距缩小分别1.269841/1.507937/1.746032pp，其中各有0.555556pp来自clean下降；不能把这一部分算作保护clean后的接近。

训练的568320份相位视图相对IQ RMS变化median0.02238647、q950.03645559、max0.06468859，参考窗max0.07682813；相位峰值≤0.1rad，均值/线性项为浮点误差级。实际输入变化和源G0保护成立，均未转化为本轮clean训练收益。生成/传输/统计计时24.942644秒包含同步等待，不是纯kernel时间；不同GPU和共享负载下的总训练秒数不作受控速度因果比较。

完整图与PDF已生成、三个PNG已实际查看：

- [全部20轮训练曲线](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_bounded_nonaffine_phase_E20_20261017/final/complete_training_curves.png)
- [七个源视图与suburban差距](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_bounded_nonaffine_phase_E20_20261017/final/held_source_results.png)
- [全部TX/day分层](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_bounded_nonaffine_phase_E20_20261017/final/source_strata.png)

下一唯一假设转向同一合法物理包的接收预处理变化；先做代码与来源配对核实，不读取新IQ或启动模型。见[下一研究说明](../../../analysis/cvs_paired_preprocessing_basis_20261017.md)。不能仅因raw/eq数组同sig_i就认定同物理包；若真实配对或原L权限无法证明，就放弃此候选入口。
