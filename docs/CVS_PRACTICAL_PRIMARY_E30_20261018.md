# 固定 CVS：practical 优先研究


## P67：物理样本跨轮确认，匹配 NoAs／As E30

20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01 新增两条 own20＋10 尾段，固定 F＋CROSS_RX／默认头／全5源RX和 L6300／hidden U56700／V27000。唯一变化是 native batch_neighbor→epoch_bank，min_streak2／min_conf.8 与原 confidence/domain/strong 规则保持，缓存 E21 重置。教师原本直接读取 raw U，不存在旁路弱教师增强的变化。无新 IQ 增强、损失、教师或前向，也不把该时间规则称为 A_g。旧 epoch_bank 的其他架构 E200／无As历史保持。

源晋级要求 AS_BANK fullV clean 分别对 matched NOAS_BANK、原 NOAS30 均≥−.1pp；midlow 对原 AS30≥＋.5pp且low≥0。任一FAIL关闭固定方式，不能调窗口/阈值/seed或按formal扩预算；源合格后才可总计E50。科学失败仍逐row完成clean＋六固定practical独立truth-last。四条件已完成全部七视图评分与分析，完整日志、配对误差及分层结果已保留。源四判据均 FAIL，不扩 E50。正式单 seed AS_BANK 对匹配 NOAS_BANK：clean ＋0.041071pp、mid urban ＋27.739286pp、low urban ＋21.054762pp、urban 均值 ＋28.405159pp，满足正式 clean 下降≤0.1pp 的阶段要求。新增时间机制对原 AS30：clean −0.086310pp、mid ＋0.023214pp、low −0.017262pp、urban 均值 ＋0.012103pp；不归因既有 As 收益，不宣称可重复双提升，整体目标 ACTIVE。

预登记：[P67 report](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01/report.md)。


用户允许表现好的方案延长到总计 E50；本轮按预登记合法源域条件判断，不以正式测试分数晋级。P67 未通过，未启动 E50，也未扩种子或扫时间窗口。后续候选先与另一对话去重。

最终分析：[P67 final](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01/final_analysis.md)。


## P68：固定6L内两个独立完整As观察

20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01只新增INDEP_CE6一条own SAT18 AS_CE20＋10，总E30/6660；既有SAT18 AS_CE30作同父SAME_CE6控制直接复用。P43原As30前20轮KL历史不同，仅作整配方参照，不把全部差异归给独立信道。旧10L/heldRX MC2的真实负结果保留；本次关键改变是现有6L预算内的观测分配，不增加前向或跨IQ KL。原raw/ordinary KL、dropout、SupCon、U/EMA、1.25L尺度保持，硬件session相同，两个As各从same raw独立完整生成。

源晋级两项：fullVclean对原NoAs30≥−.1pp；pooledmidlow对max(原As30,SAME_CE6)≥＋.5pp；任一FAIL关闭固定方式，仍评分全部七视图。合格才允许总计E50，正式不晋级；当前 ANALYZED：772NN 真实父与合成6L smoke、GPU0所属进程/CWD/argv/日志增长均已核实，E30与七固定视图／独立truth-last／完整CPU分析完成。新增尾段2220更新／4352336remoteNN（含smoke）；实际父source-only诊断另12602NN，合计4364938NN。旧控制不重训／推理／评分。

[P68预登记](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01/report.md)。


P68机制读出补充：实际SAT18 AS_CE20没有已核实源预测缓存，P43 As20不同父不可替代。独立source-only诊断 20261018-diagnostic-FcrossRX-ASCE20cohort-sourceV-s2026092701-m1-r01只补默认头在原V1800＋两draw三practical上的12602NN（含2NNCPsmoke），0训练/teacher/信道新生成/target；不写健康P68代码/root，不参与source晋级。新增成本单列，主P68expected4352336NN＋该诊断12602＝4364938NN，非新训练预算。唯一新诊断P0/P1及7puretoy/4compile通过，实际12602NN/0更新source读出与独立NPZ重数完成：父clean正确1767/1800，pooled3534观察中lowSub3171、mid2584、low2073；非正式测试准确率。短任务首次实时/proc观察前已结束，实时PID/CWD/GPU快照不可用，已披露，不重跑。


P68 最终：源 fullVclean 对NoAs −0.029630pp PASS，midlow 对同父SAME ＋0.069444pp 未达＋0.5pp，SOURCE_FAIL，不延长 E50。主比较 INDEP−SAME urban＋0.114286pp、mid＋0.087500pp、low＋0.075595pp、clean＋0.019643pp，仅单seed微增。对同预算NoAs urban＋28.302778pp、clean−0.013690pp，满足0.1pp阶段目标；对原As urban−0.065675pp且父KL历史不同。正式low51.245833%，未达60%；无可重复clean＋星地双提升证据。504个混淆矩阵／84分区独立重数，missing=[]。实际父clean正确1767源集合low保持60.016978%不是正式low准确率。

[P68 最终分析](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01/final_analysis.md)。整体目标 ACTIVE；允许给源合格方案总计 E50，不以本轮正式微增晋级。
