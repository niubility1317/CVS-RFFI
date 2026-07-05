# Phase2 Target-Old Linear-Probe Upper-Bound Diagnostic

|字段|值|
|---|---|
|实验ID|`phase2_target_old_linear_probe_upper_bound_20260706_0330`|
|记录时间|2026-07-06 03:18 CST|
|目标|在prototype式target-old-only上限最高仅`74.76%`后，检查闭式ridge线性头是否能从同一`R_t/Y_old`support中恢复更高target-old query准确率|
|协议边界|`TARGET_OLD_ONLY_UPPER_BOUND_DIAGNOSTIC`；只使用`target_old`行拆分support/query；不使用`target_new`、`target_unknown`、`proxy_unknown`训练、归一化统计、阈值拟合、早停或模型选择|
|声明边界|不声明Stage2-C成功，不报告seen-new，不报告unknown FAR/拒识，不声明部署成功；lambda网格逐行报告，不自动挑最好lambda作为部署结论|
|比较对象|`ADV3B02_CORE90_FROZEN`与`EPOC_B_OSPR_FEATURES`两个已有Stage2-C特征包|

## 本地变更与验证

|文件|目的|SHA256|
|---|---|---|
|`code/scripts/eval_target_old_linear_probe_upper_bound.py`|新增target-old-only闭式ridge线性头上限诊断，输出old/macro/min class、per-class、confusion和support/query索引hash|`CEEFC45F40D31D8931794320DC24F55BC63A3C7B8F6D530700D44ADA6F766864`|
|`code/tests/test_target_old_linear_probe_upper_bound.py`|验证只使用target_old行、忽略target_unknown、support/query无重叠、协议字段完整、无target_unknown阈值或模型选择|`D5E40F613012690AAC579FB71E70184E817FC201D19608566DEA9EB4DA78A7FA`|

本地snapshot：`E:\type10-7\code\snapshots\phase2_target_old_linear_probe_upper_bound_20260706_0330`。

|命令|结果|
|---|---|
|`PYTHONIOENCODING=utf-8; PYTHONUTF8=1; conda run -n ssr-gpu python -m pytest code\tests\test_target_old_linear_probe_upper_bound.py code\tests\test_target_old_only_upper_bound.py -q`|PASS：`4 passed`，仅`.pytest_cache`权限warning|
|`conda run -n ssr-gpu python -m py_compile code\scripts\eval_target_old_linear_probe_upper_bound.py code\tests\test_target_old_linear_probe_upper_bound.py`|PASS|

## 计划N607动作

1. 运行N607 read-only preflight。
2. 同步脚本、测试、报告和`code/SYNC_MANIFEST.txt`。
3. 远端hash、`py_compile`和直接测试函数验证；远端`CVS-RFFI`环境无pytest时使用直接函数调用。
4. 复用已有Stage2-C特征包，CPU前台运行只读诊断，不启动训练，不占用GPU。

|候选|输入特征包|输出目录|
|---|---|---|
|`ADV3B02_CORE90_FROZEN`|`runs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz`|`runs/phase2_target_old_linear_probe_upper_bound_20260706_0330/ADV3B02_CORE90_FROZEN`|
|`EPOC_B_OSPR_FEATURES`|`runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz`|`runs/phase2_target_old_linear_probe_upper_bound_20260706_0330/EPOC_B_OSPR_FEATURES`|

运行参数：`target_old_tx_ids=14-10,14-7,20-15,20-19,6-15,8-20`，`k_values=1,2,5,10,20,50`，`ridge_lambdas=0.001,0.01,0.1,1.0,10.0`，support-only标准化开启，L2 normalize开启。

## N607执行记录

执行时间：2026-07-06 03:18-03:19 CST。  
N607 preflight：PASS；GPU0/GPU1/GPU4-GPU7有低显存训练进程，GPU2/GPU3空闲。本诊断为CPU前台只读运行，不启动训练、不占用GPU、不改变R4/R5/R6进程。  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。远端hash、`py_compile`和直接测试函数验证PASS：`direct_target_old_linear_probe_tests=PASS`。  
SSH清理复查：本地`ssh.exe`为`none`，N607/bridge `ESTABLISHED`连接为`none`。

远端输出：

|候选|文件|SHA256|
|---|---|---|
|`ADV3B02_CORE90_FROZEN`|`linear_probe_metrics.json`|`52aa52516b353931fc50ed4123c7e880a7d3e2eaa79229b1426d6f2efcfd096e`|
|`ADV3B02_CORE90_FROZEN`|`linear_probe_summary.csv`|`412502c4c2e602efde60f70fcb207610c2522e63a2d9f1e8ada4c03d3db27a41`|
|`EPOC_B_OSPR_FEATURES`|`linear_probe_metrics.json`|`4034153bd380c35173f36325d2e11443a52e5b84b54edcb530b987729471e8ec`|
|`EPOC_B_OSPR_FEATURES`|`linear_probe_summary.csv`|`ef05fd0e952beef3c63c40159789b7324de51098f961efc754369cbcab1a7bcb`|

本地已拉回两个`linear_probe_metrics.json`到`artifacts/`；summary CSV保留在N607 run目录。

## 结果

|候选|选择口径|K|ridge_lambda|support_count|query_count|old_acc|macro_old_acc|min_old_class_acc|overlap|判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`ADV3B02_CORE90_FROZEN`|best old_acc row|10|10.0|60|9540|71.42%|71.42%|45.53%|0|未达OLD80，旧类floor远低于95%|
|`ADV3B02_CORE90_FROZEN`|best min_old row|50|10.0|300|9300|69.92%|69.92%|49.94%|0|K=50仍只是higher-shot诊断，floor最高也不足50%|
|`EPOC_B_OSPR_FEATURES`|best old_acc row|2|10.0|12|9588|74.81%|74.81%|34.86%|0|old_acc略高于prototype诊断，但仍未达OLD80|
|`EPOC_B_OSPR_FEATURES`|best min_old row|20|0.001|120|9480|59.82%|59.82%|43.73%|0|改善floor会明显损失overall old_acc，旧类几何不稳定|

解释：闭式ridge线性头没有突破prototype上限。即使允许多个预设`ridge_lambda`逐行诊断，最高old_acc仍只有`74.81%`，最高min old class只有`49.94%`，低于`OLD80_FIRST`阶段门槛，更远低于最终`old_acc=99%`、`min_old=95%`。这进一步证明当前底座/特征包在目标接收机旧类空间中还没有可由轻量线性头恢复的稳定几何结构。

结论：下一步不应继续增加unknown拒识阈值、协同投票或简单线性头。应进入更底层的support-protected target-old geometry修复：在合规边界内优先提高目标旧类少样本适应上限，再回到Stage2-C qknn8 M=1..全接收机复评。真实`Y_unknown`仍只能用于最终评估，不能进入地面训练或阈值/模型选择。
