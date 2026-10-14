# 真实 L 跨 RX 双边身份匹配：最小源原型

状态 **ANALYZED / VERIFIED，双边机制 INITIAL 0/4、FINAL20 0/4**。下文保留原预登记。上一RX历史过滤源资格0/4，CE四组改善而raw跨RX关系四组下降；删除RX差异方向没有身份依据。旧教师全关系、可信集合/补偿、CROSS_DIFF、episodic/ridge及原始周期pretext的限制保留。本轮只开一条不同的假设：用**真实L身份对应**约束实际四项joint160的跨RX匹配；不要求把接收机公共偏移从z删除，不新增头/prototype或teacher判定链。

每侧同RX/day包含六个不同TX，各一包，两侧物理ID不交。Dij=||a_i-b_j||²/160，B=D-rowmean-colmean+grandmean，C=B/max(RMS(B),1e-8)，RMS保持可微。令J=I-11ᵀ/6，则B=JDJ=-2(Ja)(Jb)ᵀ/160；任意两侧公共feature平移被消去，正全局幅度在floor未活跃时不改变C。可微分母阻止纯全局norm膨胀降低损失；只在损失成本表去除这些项，部署z仍是原actualjoint。它不保证真实RX失真只是加性，不消除任意非线性/频率相关失真，不宣称全部因素可辨识。

BI从exp(-C/.1)在log域做固定80次行列均匀边际缩放（目标1/6），报告实际残差和approximate标记，不默称精确OT最优。由于有限迭代未必同时完全平衡，正确TX质量分别由logP行/列条件归一化得到；只有精确均匀边际极限才等于6P正确元素。辅助loss=.1*(mean relu(log.9-logqrow)+mean relu(log.9-logqcol))/2，达到.9即停压。ROW对照使用同C的独立行列softmax，无共享列容量。W不进辅助图，主CE仍正常联合训练原默认cos头；三个gain与所有实际F71/195041coord保持原可微路径，没有分支删除或辅助embedding替代。

四条件ZERO/SAME_BI/CROSS_ROW/CROSS_BI。原L180为5RX×3day×2side×6trueTX；CROSS在side0的RXr与side1的RXt（r!=t）同day，共20有序RX对×3day=60个6×6矩阵，2160成本格/360row+360col真实身份关系；SAME同RX/day15unique重复4以匹配60成本表及每packet4端点曝光。四组都计算全部目标并保留sameforward预算，ZERO只加graphconnected0。已知类均衡只用于合法L监督与源特征分析；U不能按TX配对或用真类数，部署/target不运行全局匹配或quota。

固定own BASE_CE第一seed2026092701的原INITIAL及final20，父执行6693e4480ef990bc23c5021ff1440b547fe5fcaf，不按目标最高分选父。source_proof先核对完整roles/IDs/scratch来源/fixedfinal20后才load两个产物，strictmodel与actualargs检查。INITIAL用原groupoptions的空moments，FINAL保留实际m/v/step；两者临时新stageLR2e-4，无整体清矩或state过滤。固定四current L180 ordinal0/1、17/18、35/36、52/53，共720包；held180 ordinal68/69。900native5ID与globalbaseindex不交，held已被父训练见过，不能冒充盲测或未见RX。

源判据每state至少2/4 cohort的CROSS_BI同时优于ZERO、SAME_BI和CROSS_ROW：actualF差norm>1e-6、heldjointRMS差>1e-5、至少1浅中层tensor变化、held真实匹配条件质量均值增益>1e-4、held720双向匹配错数不增；ACTUAL与PRE_W heldCE相对三对照恶化不超过INITIAL1e-4/FINAL1e-6。当前CROSS_BI有活跃身份关系及实际aux梯度。这里直接评价新机制所要保留的身份对应，rawcos关系另报但不是资格代理，因为允许公共RXoffset而不强行几何全对齐；这不是修改旧RX过滤判据救活旧候选。INITIAL通过支持早期scratch20研究，若只有FINAL通过支持独立E20+20；没有在source assay中启动真实训练。所有阈值/τ/π/权重/iterations先于真实结果固定，失败不扫描救活或替换对照晋级。

实际after全188参数进入临时eval模型，after三个gain参与真实joint160；ACTUAL afterW与PRE_W仅beforeW分析同一z，用同FP64默认cos scale30/eps1e-4，无fit。报告各RX CE、classmargin、rawidentitygeometry、matching质量/错数、有限80边际残差、RMSfloor、aux/CE梯度方向、Sinc核与浅中层实际差。匹配损失下降或源proxy通过都不等于未见RX性能；before/after样本及配对关系曝光完整披露。

预算900uniqueL/6300metadata/7564packet网络前向（两state各2零IQ烟测）/64临时CPUAdamW、0实际训练/头拟合/EMA/U/V/target。两个own CP无query smoke PASS立即继续固定32单元。技术不一致/非有限异常保存并仅停本run，不影响其他owner。

run `20261015-phase1-SIGTIME-identity-transport-sourceL-s2026092701-r01`，owner codex/root/identity-transport-source-20261015，actualGitHEAD写release_commit.txt。普通N607/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_transport_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_transport_sourceL_20261015_r01/experiments/cvs_identity_transport20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_transport_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-identity-transport-sourceL-s2026092701-r01/source_diagnostic`、日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_transport_sourceL_20261015_r01/source_probe.stdout.log`。一个归档SHA比较/remotecompile/所属PID-CWD-cmd-GPU-loggrowth核验和独立终态读回；没有新增seal/receipt/report-onlygate。详细matrix/stop/artifacts见[experiment.json](experiment.json)。

双边匹配、熵正则OT和assignment表征学习都有先例：[Cuturi](https://arxiv.org/abs/1306.0895)、[SwAV](https://arxiv.org/abs/2006.09882)。候选待检验区别是CVS完整部署特征的**真实跨RX身份对应、损失中公共cost项与幅度中性化、有限双边容量和达到身份质量即停压**，不是Sinkhorn本身新发明。目前没有完整文献首创性或泛化收益证明。SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU历史正路线按各配方保留，84%默认头clean三seed目标active、未完成。正式训练仍只能scratch20或同父E20+20，与正常defaulthead/原hiddenU配方匹配，并在冻结后逐row自动clean预测和独立truth-last。


## 完整真实源结果

实际执行 `6b4f3fde6ed607b95f0502e81dd71c8e74ff5bbb`，commit/push/remoteOID、归档传输SHA、remotecompile、两个ownCP noquery smoke、所属PID/CWD/cmd/CPU/GPU/loggrowth以及独立自然终态均VERIFIED，无failure。完整32单元/8cohort/24成对比较/160RX双读出；source wall 67.84 s。

| state | ordinal | 条件 | aux梯度norm | held BI质量 | held BI错数/720 | PRE_W CE减少 | ACTUAL CE减少 |
|---|---|---|---:|---:|---:|---:|---:|
| INITIAL | 0 | ZERO | 0 | 0.273828 | 528 | +0.7466 | +0.7832 |
| INITIAL | 0 | SAME_BI | 1.376 | 0.272580 | 529 | +0.7448 | +0.7817 |
| INITIAL | 0 | CROSS_ROW | 3.369 | 0.283455 | 524 | +0.7358 | +0.7727 |
| INITIAL | 0 | CROSS_BI | 3.247 | 0.282202 | 525 | +0.7362 | +0.773 |
| INITIAL | 17 | ZERO | 0 | 0.273650 | 527 | +0.7493 | +0.7857 |
| INITIAL | 17 | SAME_BI | 1.863 | 0.274596 | 527 | +0.7461 | +0.7826 |
| INITIAL | 17 | CROSS_ROW | 2.859 | 0.280990 | 528 | +0.7406 | +0.7773 |
| INITIAL | 17 | CROSS_BI | 2.876 | 0.280859 | 530 | +0.7401 | +0.7767 |
| INITIAL | 35 | ZERO | 0 | 0.273138 | 529 | +0.7478 | +0.7843 |
| INITIAL | 35 | SAME_BI | 2.034 | 0.273870 | 529 | +0.7449 | +0.7816 |
| INITIAL | 35 | CROSS_ROW | 3.083 | 0.282634 | 527 | +0.7366 | +0.7736 |
| INITIAL | 35 | CROSS_BI | 3.095 | 0.282864 | 526 | +0.7367 | +0.7737 |
| INITIAL | 52 | ZERO | 0 | 0.272573 | 529 | +0.7517 | +0.788 |
| INITIAL | 52 | SAME_BI | 2.201 | 0.273750 | 528 | +0.7465 | +0.7829 |
| INITIAL | 52 | CROSS_ROW | 3.074 | 0.280566 | 524 | +0.7419 | +0.7785 |
| INITIAL | 52 | CROSS_BI | 3.429 | 0.281117 | 526 | +0.7406 | +0.7772 |
| FINAL20 | 0 | ZERO | 0 | 0.955080 | 17 | +5.687e-05 | +5.118e-05 |
| FINAL20 | 0 | SAME_BI | 0.1167 | 0.955201 | 17 | +6.613e-05 | +6.052e-05 |
| FINAL20 | 0 | CROSS_ROW | 0.2775 | 0.955915 | 17 | +9.723e-05 | +9.2e-05 |
| FINAL20 | 0 | CROSS_BI | 0.09779 | 0.955200 | 17 | +6.423e-05 | +5.857e-05 |
| FINAL20 | 17 | ZERO | 0 | 0.955285 | 17 | +0.0001028 | +0.0001001 |
| FINAL20 | 17 | SAME_BI | 0.06729 | 0.955323 | 17 | +0.000105 | +0.0001023 |
| FINAL20 | 17 | CROSS_ROW | 0.2906 | 0.956119 | 17 | +0.000145 | +0.0001428 |
| FINAL20 | 17 | CROSS_BI | 0.05138 | 0.955348 | 17 | +0.0001064 | +0.0001038 |
| FINAL20 | 35 | ZERO | 0 | 0.955079 | 17 | +4.018e-05 | +3.265e-05 |
| FINAL20 | 35 | SAME_BI | 0.06789 | 0.955189 | 17 | +3.909e-05 | +3.143e-05 |
| FINAL20 | 35 | CROSS_ROW | 0.317 | 0.955828 | 17 | +7.099e-05 | +6.373e-05 |
| FINAL20 | 35 | CROSS_BI | 0.0585 | 0.955133 | 17 | +4.036e-05 | +3.274e-05 |
| FINAL20 | 52 | ZERO | 0 | 0.955004 | 17 | +3.883e-06 | -7.327e-06 |
| FINAL20 | 52 | SAME_BI | 0.03739 | 0.955074 | 17 | +8.501e-06 | -2.62e-06 |
| FINAL20 | 52 | CROSS_ROW | 0.2761 | 0.955651 | 17 | +4.022e-05 | +2.959e-05 |
| FINAL20 | 52 | CROSS_BI | 0.01511 | 0.955043 | 17 | +5.435e-06 | -5.764e-06 |

固定source规则：{"passed": false, "stage_signal": null, "qualified_ordinals_by_state": {"INITIAL": [], "FINAL20": []}, "cell_conditions": [{"state": "INITIAL", "ordinal": 0, "qualified": false, "failed_conditions": ["ZERO:ACTUAL_CE_harm", "ZERO:PRE_W_CE_harm", "SAME_BI:ACTUAL_CE_harm", "SAME_BI:PRE_W_CE_harm", "CROSS_ROW:held_correct_matching_mass_gain", "CROSS_ROW:matching_error_harm"]}, {"state": "INITIAL", "ordinal": 17, "qualified": false, "failed_conditions": ["ZERO:matching_error_harm", "ZERO:ACTUAL_CE_harm", "ZERO:PRE_W_CE_harm", "SAME_BI:matching_error_harm", "SAME_BI:ACTUAL_CE_harm", "SAME_BI:PRE_W_CE_harm", "CROSS_ROW:held_correct_matching_mass_gain", "CROSS_ROW:matching_error_harm", "CROSS_ROW:ACTUAL_CE_harm", "CROSS_ROW:PRE_W_CE_harm"]}, {"state": "INITIAL", "ordinal": 35, "qualified": false, "failed_conditions": ["ZERO:ACTUAL_CE_harm", "ZERO:PRE_W_CE_harm", "SAME_BI:ACTUAL_CE_harm", "SAME_BI:PRE_W_CE_harm"]}, {"state": "INITIAL", "ordinal": 52, "qualified": false, "failed_conditions": ["ZERO:ACTUAL_CE_harm", "ZERO:PRE_W_CE_harm", "SAME_BI:ACTUAL_CE_harm", "SAME_BI:PRE_W_CE_harm", "CROSS_ROW:matching_error_harm", "CROSS_ROW:ACTUAL_CE_harm", "CROSS_ROW:PRE_W_CE_harm"]}, {"state": "FINAL20", "ordinal": 0, "qualified": false, "failed_conditions": ["SAME_BI:held_correct_matching_mass_gain", "SAME_BI:ACTUAL_CE_harm", "SAME_BI:PRE_W_CE_harm", "CROSS_ROW:held_correct_matching_mass_gain", "CROSS_ROW:ACTUAL_CE_harm", "CROSS_ROW:PRE_W_CE_harm"]}, {"state": "FINAL20", "ordinal": 17, "qualified": false, "failed_conditions": ["ZERO:held_correct_matching_mass_gain", "SAME_BI:held_correct_matching_mass_gain", "CROSS_ROW:held_correct_matching_mass_gain", "CROSS_ROW:ACTUAL_CE_harm", "CROSS_ROW:PRE_W_CE_harm"]}, {"state": "FINAL20", "ordinal": 35, "qualified": false, "failed_conditions": ["ZERO:held_correct_matching_mass_gain", "SAME_BI:held_correct_matching_mass_gain", "CROSS_ROW:held_correct_matching_mass_gain", "CROSS_ROW:ACTUAL_CE_harm", "CROSS_ROW:PRE_W_CE_harm"]}, {"state": "FINAL20", "ordinal": 52, "qualified": false, "failed_conditions": ["ZERO:held_correct_matching_mass_gain", "SAME_BI:held_correct_matching_mass_gain", "SAME_BI:ACTUAL_CE_harm", "SAME_BI:PRE_W_CE_harm", "CROSS_ROW:held_correct_matching_mass_gain", "CROSS_ROW:ACTUAL_CE_harm", "CROSS_ROW:PRE_W_CE_harm"]}], "controls_not_fallback": true, "interpretation": "True-label source identity correspondence signal at specified stage; not unseenRX/DG/novelty proof"}

双边CROSS_BI固定资格0/8，关闭本次双边候选，不扫描τ/π/权重/iterations或将CROSS_ROW控制直接替换为晋级。

INITIAL存在明确身份学习机会：CROSS_BI当前542/566/566/577条双向关系未满足，aux梯度norm2.876至3.429，F内与主CE梯度cos为−.0694至−.0228，不是共线放大。对ZERO held正确BI质量四组增加.007209至.009726，raw身份间隔也均小幅正增；但PRE_W heldCE比ZERO高.00924至.01112、ACTUAL高.00904至.01087，且未稳定优于CROSS_ROW。before heldCE3.372846、default准确率21.1111%；BI更新本身的CE仍比before降低约.74，负结论仅为相对控制与固定资格，没有声称训练一步把分类整体破坏。匹配错数对ZERO三组少3、一组多3，与平均质量不同。

FINAL20默认头before已100%准确、CE.004644，但独立无双边容量的ROW身份对应仍有64/720错误、143/720未满足，平均true质量.889723；BI容量缩放使诊断质量达到.955437、错误18/720、未满足67/720。两种诊断不能当成相同难度或源性能指标；容量约束能在未更新θ前改变匹配结果，不能把这部分“会匹配了”归给特征学习。BI当前只有36/36/61/44 active，梯度norm.0151至.0978。其heldBI质量自身四组小降，相对ZERO只增加3.92e-5至1.20e-4，且全部不优于CROSS_ROW的质量与CE，源资格未通过。60/60 heldBI矩阵都有限迭代近似，max边际残差约.0028；这是算法近似限制，不能包装成精确OT。本次正确质量是条件归一化，所以不使用可能>1的6P当概率；没有为结果增加迭代。

固定双边方法失败，不等于本次发现的true-L跨RX身份信号全部为零。以下是**控制中的探索性结果**，不冒充预登记CROSS_BI晋级，也不报告目标域增益：
| state | ordinal | ROW自身质量增益 | ROW错数差/720 | PRE_W CE差 | ACTUAL CE差 |
|---|---|---:|---:|---:|---:|
| INITIAL | 0 | +0.00719 | -2 | +0.010811 | +0.010518 |
| INITIAL | 17 | +0.00584 | -3 | +0.0087617 | +0.0084788 |
| INITIAL | 35 | +0.00695 | -4 | +0.011204 | +0.010766 |
| INITIAL | 52 | +0.00591 | -7 | +0.0098191 | +0.0095918 |
| FINAL20 | 0 | +0.00190 | -2 | -4.0358e-05 | -4.0824e-05 |
| FINAL20 | 17 | +0.00206 | -1 | -4.2175e-05 | -4.2676e-05 |
| FINAL20 | 35 | +0.00222 | +0 | -3.0807e-05 | -3.1082e-05 |
| FINAL20 | 52 | +0.00205 | -1 | -3.6335e-05 | -3.6917e-05 |

FINAL20 CROSS_ROW相对ZERO的独立无容量true匹配质量四组均提高.001904至.002218，错误分别减少2/1/0/1；ACTUAL heldCE四组减少3.11e-5至4.27e-5，PRE_W同向。它的F内aux/main cos为−.0784至+.1818，实际新增特征信号并非旧梯度简单加倍；但raw cosine身份间隔反而下降，两个关系量与默认头confidence不能互换。当前只是firstseed两状态的一步source证据，不是完整训练/DG。INITIAL ROW也四组有正匹配质量，但相对ZERO的CE改善较少；不能因此直接确定应早期开启。

下一优先动作是独立确认**不使用双边容量的真实跨RX身份对应监督**：保留所有本次结果及primary失败，另开预登记SOURCE-only确认，在原L角色的新固定physical ordinals2/3、19/20、37/38、54/55和held66/67上比较ZERO/SAME_ROW/CROSS_ROW，τ/.9/.1及centered可微RMS保持，不做强度扫描；不重用本次900包来宣称新确认。统一用无容量ROW对应的质量/错数、ACTUAL/PRE_W/defaultaccuracy及实际F/浅中层增量检验，先与SAME_ROW隔离“只是另一种身份对比损失”。该确认尚未实施/发布，CROSS_ROW控制不能直接替代已失败的primary。只有独立source条件通过，再固定E20+20同父同预算CE/原SSL控制和aux时序，继续保持隐藏U/正常defaulthead/逐rowclean独立truth-last；不按目标分数挑父或扩seed。

备选U中层统计donor属于MixStyle类扩展，旧MixStyle负结果和身份污染风险保留，目前不实现，不同时堆新机制。历史正收益SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU独立保留、配方不混合，84%默认头clean三seed目标仍active、未达成。

真实900uniqueL/6300metadata/7564packet网络前向/64临时CPUAdamW，0实际训练/头拟合/EMA/U/V/target。actualafter3gain进入joint160，辅助WNone，主W正常联合梯度；所有live状态/optimizer/CPpayload/grad/RNG保持，heldL曾被父训练见过，不是未见RX或独立盲测。全部RMSfloor未触发，数值不变性验证不等于物理RX模型已成立。完整[32单元](source_cells.csv)、[160RX记录](source_RX_metrics.csv)、[24对照](paired_regimes.csv)、[ROW探索正迹象](ROW_exploratory_vs_ZERO.csv)、[标量证据](result_evidence.json)、[运行核验](launch_evidence.json)已保存，私有原输出/CP/日志保留。
