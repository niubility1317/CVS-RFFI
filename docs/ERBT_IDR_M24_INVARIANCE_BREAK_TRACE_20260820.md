# ERBT-IDR M2.4非等价机制实现追踪

设计输入：用户提供的《对提交`8d712e7cb35e4c908f9975357132c10656f26a76`的全面诊断》。基线提交：`8d712e7cb35e4c908f9975357132c10656f26a76`。主比较基线固定为去RF32的D92 E0。

|ID|指导要求|实现位置|状态|验证证据|
|---|---|---|---|---|
|M24-IB01|R2不再作为性能路线，保留为等价回归证据|既有D1/R2报告；本轮G0–G4矩阵|implemented|G0固定为去RF32的D92 E0；R2未进入候选矩阵|
|M24-IB02|冻结度量，不拟合target全协方差|`stage2_m24_invariance_breaking.py`G1|verified_n607|125行完成；H=0.297636，低于G0的0.537558|
|M24-IB03|非可逆低秩receiver nuisance suppression|同模块G2，复用D74正交rank-1投影|verified_n607|125行完成；相对G1的行等权H仅+0.000107，作用不稳定|
|M24-IB04|类别不确定性惩罚且类别对称|同模块G3|verified_n607|125行完成；相对G2的行等权H−0.012096，仅1行提升|
|M24-IB05|K≥5局部多原型，按类归一化聚合|同模块G4|verified_n607|125行完成；H=0.278228，为四个候选最低|
|M24-IB06|K1/K2专用头，不强制退回历史F1|G1–G4的K策略|verified_n607|K1/new20 H=0.227895；K2候选H最高0.252259，均低于G0|
|M24-IB07|query逐样本独立，不进入拟合|新模块API和row executor|verified_n607|625个truth-unopened prediction闭合后才启动独立scorer|
|M24-IB08|完整125实验与D92 E0同row比较|新runner、scorer、analyzer|verified_n607|625行、1875场景单元、500组paired-vs-G0全部闭合；G1–G4 H逐row均低于G0|
|M24-IB09|状态、margin、中心角距、help/harm和遗忘诊断|row diagnostics和结果汇总|verified_n607|`results_summary.json`状态`ANALYZED`，全部预登记维度已生成|

`REJECTED_EXTRA_GATE`：指导中的“非等价Gate0–Gate3”按项目最高优先级工作流实现为科学诊断和预注册停止规则，不增加设计SHA、逐row seal、额外审核或发布许可。

本地验证：48项发布前聚焦回归通过；汇总兼容修复另有19项聚焦测试通过。Python编译、`git diff --check`、625行矩阵与scorer静态闭合通过。一次独立P0/P1审查结论为`NO_P0_P1`。

## 完整125结果追踪

- run ID：`erbt_idr_m24_invariance_break_full125_20260820_v1`；
- prediction：625/625行，G0–G4各125，`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`；
- scoring：625个same-row、625个four-state、500个paired-vs-G0和500个标准化遗忘结果，状态`PASS`；
- 汇总修复提交：`d5b004a396b4ac306601129645072a9e6e317718`；
- 性能裁决：G1/G2/G3/G4的query加权H分别为0.297636/0.297678/0.285538/0.278228，G0为0.537558；四个候选全部`DO_NOT_PROMOTE`；
- 详细报告：`automation_reports/CV-SincNet/erbt_idr_m24_invariance_break_full125_20260820_v1/report.md`；
- 机器结果：同目录`results_summary.json`。
