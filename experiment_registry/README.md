# 实验总索引

更新：2026-09-19T12:14:55+00:00

先按方法/问题检索，再用run ID读取精确配置与证据。历史组数量不等于独立实验数量。

- [管理规范与常用命令](../docs/EXPERIMENT_MANAGEMENT.md)
- [完整目录CSV](catalog.csv) · [结构化目录](catalog.jsonl) · [覆盖与缺项](coverage.json)
- 通过show的`--section artifacts/facts`读取该条目的路径或配置出处；细目按记录压缩保存于details/，不扫描全库。

## 登记规模

|记录类型|数量|
|---|---:|
|managed_run|9|

历史状态统一为HISTORICAL_UNVERIFIED；RUNNING等原文声明只供查证，不能证明此刻仍在运行。

## 按方法与用途查找

|路径标签（定位提示）|证据组数|
|---|---:|
|[daot](by_method/daot.md)|8|
|[rc4](by_method/rc4.md)|8|
|[phase1](by_method/phase1.md)|6|
|[fasttrust](by_method/fasttrust.md)|6|
|[concat](by_method/concat.md)|6|
|[practical](by_method/practical.md)|5|
|[full](by_method/full.md)|5|
|[residual](by_method/residual.md)|5|
|[zf](by_method/zf.md)|5|
|[mmse](by_method/mmse.md)|5|
|[response](by_method/response.md)|3|
|[core90](by_method/core90.md)|3|
|[three_seed](by_method/three_seed.md)|2|
|[adv3b02](by_method/adv3b02.md)|1|
|[original_leo](by_method/original_leo.md)|1|
|[evaluation](by_method/evaluation.md)|1|

## 最近记录入口

|名称|类型|报告/原目录|
|---|---|---|
|ADV3B02＋DAOT＋FastTrust-RC4原LEO拼接增强|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-original-leo-manysig-s392005-r01/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r01/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r02/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验r03|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r03/report.md)|
|practical四组最新保存权重测试|managed_run|[打开](../automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r01/report.md)|
|practical四组最新保存权重测试r02|managed_run|[打开](../automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r02/report.md)|
|原生DAOT＋RC4与仅动力博弈三seed对照|managed_run|[打开](../automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r1/report.md)|
|原生DAOT＋RC4与仅动力博弈三seed对照|managed_run|[打开](../automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r2/report.md)|
|响应博弈矩阵测试评估：VERIFIED|managed_run|[打开](../automation_reports/CV-SincNet/response_matrix_eval_20260917/report.md)|
