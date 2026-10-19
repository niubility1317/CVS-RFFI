# 冻结恢复器的小改变量保留

状态：PLANNED。本候选尚无源V或目标结果。

目的：在冻结R03恢复器上，检查只由无标签源波形确定的小改变量保留规则，能否减少正常IQ被不必要改写，并保持已有星地恢复收益。原对话继续物理算子估计，本候选独立负责输入保留；不改其代码或进程。

方法：q=mean((R(y)-y)^2)/max(mean(y^2),1e-4)。使用固定8192条L_s/U_s clean IQ、随机seed2026101933，无重复采样。tau=min(0.02,校准q最大值)。q<=tau时原样输出y，否则输出R(y)。阈值在V或F打开前冻结；不用TX标签、F特征/输出、场景真值或目标数据。它是小改变量启发式，不是真实clean检测器；受损样本也可能旁路，须如实报告。

固定原IQ、原R03、保留R03三组，接原50轮F＋跨RX首seed和checkpoint内默认头，全部冻结，不另训头。继承R03/R02原63000源角色和无TX训练来源，复用上轮已核实的原F50完整source role_ids与native图等价结论，不新增数据重验。新增优化器/模型/头更新为0；R03历史恢复E40与原F50成本单列。

校准完成、真实父checkpoint无query smoke PASS后立即继续1800源V七视图诊断。全部预测完成后另启独立CPU scorer，才读取V标签和配对clean参考；报告accuracy、Macro-F1、逐TX/RX/day修复/损坏/净修复、NMSE、门旁路数。clean、郊区三档和城区三档分别报告，不把相关视图当独立seed。

源证伪条件在运行前固定：1800条clean预测逐样本等于raw identity且clean IQ NMSE为0；每个practical的accuracy/Macro-F1均不低于原R03，配对NMSE不高于原R03。任一失败即记录本候选负结果，不据分数改阈值或选择性重跑。本run明确只做源诊断；本对话不新增practical目标测试。

8项聚焦检查已通过，覆盖逐样本/顺序不变、有限阈值、无V校准、冻结父模型和全视图完成先于标签读取。具体矩阵、参数、命令、环境、路径、CPU和停止规则见[experiment.json](experiment.json)。
