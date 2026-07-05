# Phase2 Target-Old-Only Upper-Bound Diagnostic

|字段|值|
|---|---|
|实验ID|`phase2_target_old_only_upper_bound_20260706_0315`|
|记录时间|2026-07-06 03:15 CST|
|目标|在feature-space upper-bound仍为负证据后，隔离检查目标接收机域`R_t`内旧类`Y_old`样本本身能否支撑高old target query准确率|
|协议边界|只使用`target_old`角色内部support/query确定性拆分；不使用`target_new`、`target_unknown`、`proxy_unknown`训练、校准、阈值拟合、早停或模型选择|
|判定用途|该诊断是`TARGET_OLD_ONLY_UPPER_BOUND_DIAGNOSTIC`，不是Stage2-C成功、不是部署成功；若旧类上限不足，后续应先修旧类目标域适应；若旧类上限高，则瓶颈集中在open-set边界与seen-new/unknown隔离|

## 本地变更与验证

|文件|目的|SHA256|
|---|---|---|
|`code/scripts/eval_target_old_only_upper_bound.py`|新增target-old-only support/query上限诊断；输出K-shot old accuracy和min old class floor|`0008DC897CFDE140764FA10E00458494C5DFEAB3639C2E9F92DE6CE38FD5ABCD`|
|`code/tests/test_target_old_only_upper_bound.py`|验证诊断只使用target_old行、忽略target_unknown、support/query无重叠、缺少target_old时fail closed|`21F5FB3A0BCDAE316AC3D086A138FE99F1C2DB463D9DC8FAA3C75D8E196A3775`|

本地snapshot：`E:\type10-7\code\snapshots\phase2_target_old_only_upper_bound_20260706_0315`。

|命令|结果|
|---|---|
|`PYTHONIOENCODING=utf-8; PYTHONUTF8=1; conda run -n ssr-gpu python -m pytest code\tests\test_target_old_only_upper_bound.py code\tests\test_phase1_ood_geometry_baseline.py -q`|PASS：`5 passed`，仅`.pytest_cache`权限warning|

## 计划N607动作

同步脚本、测试、报告和`code/SYNC_MANIFEST.txt`，远端运行`py_compile`、focused test，再复用两个已有Stage2-C特征包执行K-shot target-old-only诊断。

|候选|输入特征包|输出目录|
|---|---|---|
|`ADV3B02_CORE90_FROZEN`|`runs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz`|`runs/phase2_target_old_only_upper_bound_20260706_0315/ADV3B02_CORE90_FROZEN`|
|`EPOC_B_OSPR_FEATURES`|`runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz`|`runs/phase2_target_old_only_upper_bound_20260706_0315/EPOC_B_OSPR_FEATURES`|

运行参数：`target_old_tx_ids=14-10,14-7,20-15,20-19,6-15,8-20`，`k_values=1,2,5,10,20,50`。

## N607执行记录

执行时间：2026-07-06 03:15-03:16 CST。  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。远端`pytest`不可用，因此执行`py_compile`后用直接调用测试函数方式复验，结果`direct_target_old_only_tests=PASS`。本轮运行不启动训练、不使用GPU、不改变R4/R5/R6进程。

远端输出：

|候选|文件|SHA256|
|---|---|---|
|`ADV3B02_CORE90_FROZEN`|`target_old_metrics.json`|`60184dacb3053db38c17b95418f0e9753ae5c8e5ab4f5fe42aa20338ee96f0fe`|
|`ADV3B02_CORE90_FROZEN`|`target_old_summary.csv`|`4a64aa9cd4bd21d32c0bf03f6ae24dbe1da941f9f31c16324c48b2ea7c2d9a00`|
|`ADV3B02_CORE90_FROZEN`|`target_old_detail.csv`|`514d124909858e3bf03238f1709ce4c2a9a35ee75c585387fee70924dd5f7744`|
|`EPOC_B_OSPR_FEATURES`|`target_old_metrics.json`|`83eb200160614b01d9c073d0e910773bbffad9a97de09e028cd07b9b2a0c7dbe`|
|`EPOC_B_OSPR_FEATURES`|`target_old_summary.csv`|`2b2b078191931a390dffec16a9eb3e24247d8fc237b9e325c5999e57c12b153f`|
|`EPOC_B_OSPR_FEATURES`|`target_old_detail.csv`|`f0dd7469665e754741529a23209cf236688c2f0fef6d223c6cff553a6db34d9d`|

本地已拉回两个`target_old_metrics.json`到`artifacts/`；summary/detail CSV保留在N607 run目录。SSH清理复查：本地`ssh.exe`为`none`，N607/bridge `ESTABLISHED`连接为`none`。

## 结果

|候选|K|support_count|query_count|old_acc|min_old_class_acc|overlap|
|---|---:|---:|---:|---:|---:|---:|
|`ADV3B02_CORE90_FROZEN`|1|6|9594|64.98%|36.27%|0|
|`ADV3B02_CORE90_FROZEN`|2|12|9588|71.59%|44.49%|0|
|`ADV3B02_CORE90_FROZEN`|5|30|9570|71.77%|48.28%|0|
|`ADV3B02_CORE90_FROZEN`|10|60|9540|71.66%|47.48%|0|
|`ADV3B02_CORE90_FROZEN`|20|120|9480|71.62%|46.39%|0|
|`ADV3B02_CORE90_FROZEN`|50|300|9300|67.82%|37.74%|0|
|`EPOC_B_OSPR_FEATURES`|1|6|9594|61.66%|4.25%|0|
|`EPOC_B_OSPR_FEATURES`|2|12|9588|73.43%|31.54%|0|
|`EPOC_B_OSPR_FEATURES`|5|30|9570|74.76%|38.93%|0|
|`EPOC_B_OSPR_FEATURES`|10|60|9540|74.15%|36.04%|0|
|`EPOC_B_OSPR_FEATURES`|20|120|9480|74.48%|41.20%|0|
|`EPOC_B_OSPR_FEATURES`|50|300|9300|68.35%|14.52%|0|

解释：target-old-only上限诊断仍为负证据。即使完全隔离未知拒识和seen-new，只在`R_t`旧类内部做support/query prototype分类，old_acc最高也只有`74.76%`，min old class最高只有`48.28%`，远低于目标`old_acc=99%`和`min_old_class_acc=95%`。因此当前主要瓶颈不只是open-set阈值或协同融合，而是目标接收机域旧类特征几何本身没有达到可适应上限。

下一步决策：在没有更强旧类目标域上限前，不应继续堆叠unknown拒识阈值或更多协同投票。后续新ADV3B02教师蒸馏应优先做`support-protected target-old geometry`：保持source-only训练不接触真实未知类，同时用目标旧类支持邻域的可迁移结构约束作为后续少样本adapter/原型注册的设计目标；完成后必须重新跑target-old-only上限和真实Stage2-C qknn8 M=1..全接收机复评。
