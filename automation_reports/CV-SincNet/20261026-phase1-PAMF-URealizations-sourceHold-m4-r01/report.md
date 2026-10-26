# 四种固定损失的多实现恢复视图匹配对照

ANALYZED，独立核算VERIFIED。四条只有一个学生seed2026092701，全部own原E30+20→E50/11100，实际代码`abc13af0b1e145e886eab61f2054611774cd0002`。四个真实final默认头、全部17760条新增记录及3196行完整stdout、80个epoch和四实现等量曝光已核实；544个源分层单元由保存逐包预测独立重算。原L768/AsKL0/ORD.1/rawPL保持，零非有限跳步、U真值、目标训练、G/teacher/另训头更新。

## 完整源结果

相对同seed原PLControl、同sourceV27000/八视图，单位pp。RX1只在附加分支留出，原L/rawPL/E30/G见过它，不是真正未见基座RX。

| 方法 | heldG三景差 | overall V raw差 | 原+.3/−.1门槛 |
| --- | ---: | ---: | --- |
| GTaskMulti | +0.148148 | -0.018519 | FAIL |
| GGCLMulti | +0.209877 | -0.011111 | FAIL |
| HGCLMulti | +0.203704 | +0.003704 | FAIL |
| GJointMulti | +0.172840 | -0.029630 | FAIL |

四条均未通过，不扩种子或目标；不是三个学生种子性能。每row U1134000、held226800、附加接受838780、额外FW2268000；每份r0/r1/r2/r3各U283500/受理209695/calls1110。GTask无CL，另三条有效anchor均1677560；相同接受/held/实现日程已逐步比较，未强迫GTask拥有CL计数。

## 仅改变数据支持的直接效应

新方法与自己的旧fixed控制保持原损失、系数、原父权重和两次附加前向，只改四份信道实现各5个完整Uepoch。表中差值是multi减fixed；共同PLControl抵消，源held三景共16200次判断，直接由共同基线净正确数相减。

| multi | fixed控制 | multi−fixed（pp） | 净正确数变化 |
| --- | --- | ---: | ---: |
| GTaskMulti | GFeatureHard | +0.006173 | 1 |
| GGCLMulti | BridgeGG | +0.006173 | 1 |
| HGCLMulti | BridgeHG | +0.024691 | 4 |
| GJointMulti | GJoint | -0.006173 | -1 |

多实现有三条略升、一条略降，幅度只有−1至+4次净正确，未达到原+.3pp。不同G输出确有波形和预测差异，但这次完整匹配训练说明视图多样性单独不足以提供所需迁移收益；不能把波形变化比例当新身份信息，也不按分数挑信道seed。

当前四条保留旧SIG_TIME结构；用户随后指定所有后续新增网络去SIG_TIME。即使将来回看源结果，也不据此启动旧SIG扩展。新方向见[完整复盘](../20261026-analysis-PAMF-SSL-retrospective-r01/report.md)与[NoSIG新设计](../../../docs/goals/PAMF_NOSIG_RGPL_DESIGN_20261026.md)。原rawclean/raw六景/同G六景三学生seed联合目标仍未达成。

证据：[全部源单元](source_all_strata.csv)、[固定判断](source_candidate_decision.csv)、[fixed/multi对照](source_fixed_vs_multi.csv)、[完整20轮曲线](source_training_curves.csv)、[完整训练执行](finished_source_execution.csv)、[结果核算](source_result_evidence.json)、[原预登记](preregistration.md)。
