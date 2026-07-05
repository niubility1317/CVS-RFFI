# Phase2 Feature-Space Upper-Bound Diagnostic

|字段|值|
|---|---|
|实验ID|`phase2_feature_space_upper_bound_diag_20260706_0300`|
|记录时间|2026-07-06 03:00 CST|
|目标|在R6 E35窗口仍呈负趋势后，转入只读feature-space upper-bound诊断，检查ADV3B02 frozen与EPOC_B特征包在`z_id`空间对old、seen-new、unknown的几何可分性|
|协议边界|诊断只使用已有Stage2-C特征包；`source`角色只用于source-calibrated阈值；`target_unknown`只作为评估query，不参与训练、阈值拟合、早停或模型选择|
|比较对象|`ADV3B02_CORE90_SOFT_E200` frozen特征包；`EPOC_B`蒸馏特征包；R6 E35负趋势作为触发条件|
|判定用途|若source-calibrated几何诊断仍无法同时保持known覆盖与unknown拒识，则说明当前特征空间没有可用开放边界，应优先做target-old-only上限诊断和support-protected feature geometry设计|

## 本地验证

|命令|结果|
|---|---|
|`conda activate ssr-gpu; python -m pytest code\tests\test_phase1_ood_geometry_baseline.py -q`|失败：当前PowerShell未正确切换到`ssr-gpu`，落到base且缺少`pytest`|
|`conda run -n ssr-gpu python -m pytest code\tests\test_phase1_ood_geometry_baseline.py -q`|首次触发Conda GBK编码崩溃，不是测试失败|
|`PYTHONIOENCODING=utf-8; PYTHONUTF8=1; conda run -n ssr-gpu python -m pytest code\tests\test_phase1_ood_geometry_baseline.py -q`|PASS：`3 passed`，仅`.pytest_cache`权限warning|

## 计划N607动作

N607动作只读复用已有特征包，不导出新特征、不启动训练、不修改R4/R5/R6进程。

|候选|输入特征包|输出目录|
|---|---|---|
|`ADV3B02_CORE90_FROZEN`|`runs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz`|`runs/phase2_feature_space_upper_bound_diag_20260706_0300/ADV3B02_CORE90_FROZEN`|
|`EPOC_B_OSPR_FEATURES`|`runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz`|`runs/phase2_feature_space_upper_bound_diag_20260706_0300/EPOC_B_OSPR_FEATURES`|

运行参数：`source_tx_ids=14-10,14-7,20-15,20-19,6-15,8-20`，`calibration_roles=source`，`known_query_roles=target_old,target_new`，`unknown_query_roles=target_unknown`，`distance_quantile=0.95`，`energy_quantile=0.95`，`knn_k=8`，`use_energy_gate=true`。

成功/失败解释边界：该诊断不产生Stage2-C成功声明；它只回答“冻结特征空间中，source-calibrated几何阈值是否存在保旧/保新/拒未知的可用上限”。最终目标仍需后续真实Stage2-C qknn8 M=1..全接收机同row复评。

## N607执行记录

执行时间：2026-07-06 03:00-03:01 CST。  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，`numpy/torch`导入正常；`python3`缺少`torch`，未用于诊断。  
远端命令：在`/home/szu2070436088/2510044040/CV-SincNet`下运行`code/scripts/eval_phase1_ood_geometry_baseline.py`两次，输出到`runs/phase2_feature_space_upper_bound_diag_20260706_0300/<candidate>/`。  
输出hash：

|候选|文件|SHA256|
|---|---|---|
|`ADV3B02_CORE90_FROZEN`|`geometry_metrics.json`|`c3360af278a8431440c472fdf800a6ea3705094324a58375f2040e6faf661bde`|
|`ADV3B02_CORE90_FROZEN`|`geometry_scores.csv`|`3e227c36341f8dc2c02e1e412deea6d71a5ceafed3f4d1e88da178f2bd0c6984`|
|`EPOC_B_OSPR_FEATURES`|`geometry_metrics.json`|`84d6ee2fb2aaec03cdb0ba2656e4575fea3f0f892d9e81ba9c2b9763cb39d4f4`|
|`EPOC_B_OSPR_FEATURES`|`geometry_scores.csv`|`7018661e86929b73ad5a13b750734cd52064b3ed497fbf976f8bd80a8b3f06da`|

本地已拉回两个`geometry_metrics.json`到`artifacts/`；score CSV保留在N607 run目录。SSH清理复查：本地`ssh.exe`为`none`，N607/bridge `ESTABLISHED`连接为`none`。

## 结果

|候选|known_query_count|known_closed_acc_no_reject|known_full_acc_after_reject|known_accepted_acc|known_coverage|unknown_query_count|unknown_FAR|unknown_reject_rate|FPR95|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`ADV3B02_CORE90_FROZEN`|9600|72.24%|22.91%|96.79%|23.67%|8000|10.60%|89.40%|84.25%|负证据：source-calibrated几何阈值能提升accepted-only准确率，但覆盖过低，unknown拒识也未到99%|
|`EPOC_B_OSPR_FEATURES`|9600|75.09%|24.32%|98.07%|24.80%|8000|10.19%|89.81%|81.52%|负证据：比ADV3B02 closed known略高，但同样以拒掉约75% known query为代价，unknown拒识仍不足|

解释：feature-space upper-bound诊断没有发现可直接部署的source-calibrated开放边界。两个特征包在不使用`target_unknown`调阈值的条件下，unknown拒识只能到约89%-90%，距离99%目标明显不足；若压低FAR，known覆盖会急剧下降，导致known full accuracy只有约23%-24%。这与OSPR-CI++“强拒识但old/seen-new崩溃”的负证据一致，说明主要瓶颈仍是特征空间边界，而不是协同投票或资源预算。

下一步：执行`target-old-only`上限诊断，隔离判断目标接收机域旧类样本本身是否能把old target query推到高准确率。如果target-old-only仍低，优先修旧类目标域适应上限；如果target-old-only高，则后续新ADV3B02教师蒸馏应采用support-protected feature geometry，而不是继续单纯增强proxy unknown。
