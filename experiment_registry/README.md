当前D92新增：[GroupBarrierJoint完整源码与support预登记](../automation_reports/CV-SincNet/20261001-phase2-d92-group-barrier-joint-support-m2-r01/report.md)，4行／160parents，RUNNING，首次launch/live runtime VERIFIED（9518d46d2，supervisor1004125）。58项相关合成验证不代表性能改善。原Margin query矩阵已观察到两行技术失败、一行完成、一行运行；健康行继续，不评分成功子集。

<!-- GROUP_BARRIER_JOINT_ENTRY_20261001 -->

# 实验总索引

更新：2026-09-28T06:32:12+00:00

先按方法/问题检索，再用run ID读取精确配置与证据。历史组数量不等于独立实验数量。

- [管理规范与常用命令](../docs/EXPERIMENT_MANAGEMENT.md)
- [完整目录CSV](catalog.csv) · [结构化目录](catalog.jsonl) · [覆盖与缺项](coverage.json)
- 通过show的`--section artifacts/facts`读取该条目的路径或配置出处；细目按记录压缩保存于details/，不扫描全库。

## 登记规模

|记录类型|数量|
|---|---:|
|managed_run|15|

历史状态统一为HISTORICAL_UNVERIFIED；RUNNING等原文声明只供查证，不能证明此刻仍在运行。

## 按方法与用途查找

|路径标签（定位提示）|证据组数|
|---|---:|
|[daot](by_method/daot.md)|11|
|[rc4](by_method/rc4.md)|10|
|[practical](by_method/practical.md)|7|
|[residual](by_method/residual.md)|7|
|[phase1](by_method/phase1.md)|6|
|[fasttrust](by_method/fasttrust.md)|6|
|[concat](by_method/concat.md)|6|
|[full](by_method/full.md)|5|
|[zf](by_method/zf.md)|5|
|[mmse](by_method/mmse.md)|5|
|[cvs](by_method/cvs.md)|3|
|[residual_noeq](by_method/residual_noeq.md)|3|
|[d92](by_method/d92.md)|3|
|[response](by_method/response.md)|3|
|[core90](by_method/core90.md)|3|
|[sixscene](by_method/sixscene.md)|2|
|[evaluation](by_method/evaluation.md)|2|
|[true256](by_method/true256.md)|2|
|[final_eval](by_method/final_eval.md)|2|
|[truth_last](by_method/truth_last.md)|2|
|[three_seed](by_method/three_seed.md)|2|
|[adv3b02](by_method/adv3b02.md)|1|
|[original_leo](by_method/original_leo.md)|1|
|[ratio](by_method/ratio.md)|1|
|[fasttrust_rc4](by_method/fasttrust_rc4.md)|1|
|[source_only](by_method/source_only.md)|1|
|[d92_parent](by_method/d92_parent.md)|1|
|[d42](by_method/d42.md)|1|
|[diagnostic](by_method/diagnostic.md)|1|
|[support_only](by_method/support_only.md)|1|

## 最近记录入口

|名称|类型|报告/原目录|
|---|---|---|
|ADV3B02＋DAOT＋FastTrust-RC4原LEO拼接增强|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-original-leo-manysig-s392005-r01/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r01/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r02/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验r03|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r03/report.md)|
|practical四组最新保存权重测试|managed_run|[打开](../automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r01/report.md)|
|practical四组最新保存权重测试r02|managed_run|[打开](../automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r02/report.md)|
|DAOT＋FastTrust-RC4 residual环境比例六组|managed_run|[打开](../automation_reports/CV-SincNet/20260920-phase1-daot-rc4-residual-ratios-manysig-s392005-r01/report.md)|
|旧Practical四组checkpoint统一六环境测试|managed_run|[打开](../automation_reports/CV-SincNet/20260920-phase1-practical4-sixscene-eval-s392005-r01/report.md)|
|CVS：DAOT＋FastTrust-RC4原始residual_noeq五种子实验|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01/report.md)|
|CVS最终clean/星地测试与D92 E0真实256维注册|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase2-cvs-d92-practical-manytx-m5-r01/report.md)|
|D42 support-only技术诊断|managed_run|[打开](../automation_reports/CV-SincNet/20260928-diagnostic-d42-support-manytx-m2-r01/report.md)|
|D92 E0 true256 numerical recovery: all five fixed seeds|managed_run|[打开](../automation_reports/CV-SincNet/20260928-phase2-cvs-d92-practical-manytx-m5-r02/report.md)|
|原生DAOT＋RC4与仅动力博弈三seed对照|managed_run|[打开](../automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r1/report.md)|
|原生DAOT＋RC4与仅动力博弈三seed对照|managed_run|[打开](../automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r2/report.md)|
|响应博弈矩阵测试评估：VERIFIED|managed_run|[打开](../automation_reports/CV-SincNet/response_matrix_eval_20260917/report.md)|

## 2026-10-01 D92 Margin联合support预登记

- [20261001-phase2-d92-margin-joint-support-m2-r01](../automation_reports/CV-SincNet/20261001-phase2-d92-margin-joint-support-m2-r01/report.md)：TRAINING_ON_SUPPORT（启动VERIFIED），4row/160parent，旧6×K4×新增5，runtime e50de0ad4/PID756554；未完成，无真实Margin成绩。

## 2026-10-01 Margin独立Ground A配对预登记

- [20261001-phase2-d92-margin-ground-a-support-m2-r01](../automation_reports/CV-SincNet/20261001-phase2-d92-margin-ground-a-support-m2-r01/report.md)：待唯一Margin数学摘要完成，单来源四行/160parent；尚未启动，不训练。

### MarginJoint 完整三阶段重复基准（2026-10-01）

- 20261001-phase2-d92-margin-joint-repeat-m2-r01：PLANNED；4行/2400parents，旧6、完整K×新增矩阵；[预登记报告](../automation_reports/CV-SincNet/20261001-phase2-d92-margin-joint-repeat-m2-r01/report.md)。新query未启动，非新独立验证。

### MarginJoint 当前独立证据（2026-10-01）

- 20261001-phase2-d92-margin-joint-support-m2-r01：ANALYZED；support数学分析、真实A/B/C、全训练诊断与紧凑日志均完成；[结果报告](../docs/D92_MARGIN_JOINT_SUPPORT_RESULT_OVERVIEW_20261001.md)。
- 20261001-phase2-d92-margin-ground-a-support-m2-r01：ANALYZED，4行/160parent，原数据保留。
- 20261001-phase2-d92-margin-joint-repeat-m2-r01：RUNNING，supervisor869328，4行/2400parent；[当前报告](../automation_reports/CV-SincNet/20261001-phase2-d92-margin-joint-repeat-m2-r01/report.md)，未评分、非新独立验证。

## D92合成软件开销补充记录（2026-10-01）

- [20261001-d92-margin-single-query-cost-synthetic-r01](../automation_reports/CV-SincNet/20261001-d92-margin-single-query-cost-synthetic-r01/report.md)：ANALYZED/VERIFIED，8行和24组实际计时完成；固定8行，纯合成单样本推理成本，不训练或读取真实query，不替代现有完整准确率基准。

- 20261001-phase2-d92-margin-joint-repeat-m2-r01：RUNNING；一行 `UNSUPPORTED_ACTIVE_JACOBIAN` 技术失败，完整矩阵未固定，其他健康任务继续；[当前报告](../automation_reports/CV-SincNet/20261001-phase2-d92-margin-joint-repeat-m2-r01/report.md)。<!-- row_failure_observed_1790861110983221800 -->
