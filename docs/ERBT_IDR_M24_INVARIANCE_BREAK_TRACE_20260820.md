# ERBT-IDR M2.4非等价机制实现追踪

设计输入：用户提供的《对提交`8d712e7cb35e4c908f9975357132c10656f26a76`的全面诊断》。基线提交：`8d712e7cb35e4c908f9975357132c10656f26a76`。主比较基线固定为去RF32的D92 E0。

|ID|指导要求|实现位置|状态|验证证据|
|---|---|---|---|---|
|M24-IB01|R2不再作为性能路线，保留为等价回归证据|既有D1/R2报告；本轮G0–G4矩阵|implemented|G0固定为去RF32的D92 E0；R2未进入候选矩阵|
|M24-IB02|冻结度量，不拟合target全协方差|`stage2_m24_invariance_breaking.py`G1|verified_local|平衡IF能量与support-only API测试通过|
|M24-IB03|非可逆低秩receiver nuisance suppression|同模块G2，复用D74正交rank-1投影|verified_local|rank loss、投影幂等、类中心安全测试通过|
|M24-IB04|类别不确定性惩罚且类别对称|同模块G3|verified_local|类别置换和高离散类惩罚测试通过|
|M24-IB05|K≥5局部多原型，按类归一化聚合|同模块G4|verified_local|K阈值、双原型、log-mean-exp测试通过|
|M24-IB06|K1/K2专用头，不强制退回历史F1|G1–G4的K策略|verified_local|K1/K2审计字段与集成测试通过|
|M24-IB07|query逐样本独立，不进入拟合|新模块API和row executor|verified_local|API负测、batch composition及truth-unopened artifact测试通过|
|M24-IB08|完整125实验与D92 E0同row比较|新runner、scorer、analyzer|implemented|625行矩阵和scorer静态闭合通过；待N607实测|
|M24-IB09|状态、margin、中心角距、help/harm和遗忘诊断|row diagnostics和结果汇总|implemented|分析器已接入；待truth-last结果|

`REJECTED_EXTRA_GATE`：指导中的“非等价Gate0–Gate3”按项目最高优先级工作流实现为科学诊断和预注册停止规则，不增加设计SHA、逐row seal、额外审核或发布许可。

本地验证：48项聚焦回归通过，Python编译通过，`git diff --check`通过，625行矩阵与scorer静态闭合通过。一次独立P0/P1审查结论为`NO_P0_P1`，未改文件。
