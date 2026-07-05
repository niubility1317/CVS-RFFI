# Phase1 EPOC-R4 Teacher-Locked Tail Quarantine Report

## 基本信息

|项目|值|
|---|---|
|run_id|`phase1_epoc_r4_teacher_tailq_20260706`|
|时间|2026-07-06|
|operator|Codex|
|目标|在`ADV3B02_CORE90_SOFT_E200`教师约束下，启动source-only地面蒸馏/再训练，修复R3暴露的虚拟未知仍贴近旧类流形问题|
|定位|Phase1/source-only底层修复；不是Stage2-C成功，不是部署成功，不是真实未知类训练|
|触发证据|完整Stage2-C冻结诊断`phase2_adv3b02_frozen_manytx_unknown_diag_20260706`显示`goal_satisfied_counts=[]`，M=1..5均失败；R3 warmup后段`proxy_auc<0.55`且`virtual_accept>0.5`|

## 协议边界

|边界|执行方式|
|---|---|
|训练数据|只加载`ManySig.pkl`；不加载`ManyTx.pkl`；不使用`--new_wisig_pkl`|
|真实未知类|`Y_unknown`和`target_unknown`不进入训练、阈值、prototype、adapter、profile、receiver权重或early stopping|
|目标接收机|地面训练不接触`R_t`样本、统计、BN、阈值、prototype、adapter或验证结果|
|底座/教师|`ADV3B02_CORE90_SOFT_E200`同时作为`baseline_ckpt`和`teacher_ckpt`|
|LEO视图|只使用源域派生的`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|后续成功判定|R4即使Phase1改善，也必须重新进入真实Stage2-C qknn8协同诊断，同一行满足旧类、新类、未知类目标后才可升级声明|

## R4设计假设

R3使用较强energy/VOS和几何约束后，旧类未塌陷，但`proxy_auc`低于R2，说明当前虚拟未知仍可能贴近旧类流形或被旧类接受区覆盖。R4不继续简单加大拒识权重，而是调整为：

1. 加强ADV3B02教师锁定，避免旧类/新类可迁移特征被边界损失破坏。
2. 收紧自动接受半径和prototype fuse半径，把tail样本从自动接受区剥离。
3. 强化tail quarantine、source safe和energy margin quantile，使source-only虚拟负样本远离旧类core。
4. 保持真实未知类和目标接收机完全不可见，后续只用Stage2-C评估验证。

## 候选矩阵

|候选|GPU|定位|关键机制|
|---|---:|---|---|
|`EPOC_R4_TEACHER_LOCK_TAILQ`|4|教师锁定+温和tail隔离|更强teacher clean/sat/zid蒸馏，较低proxy/soft权重，更小`ow_feat_radius_deg=14`和`phase2_fuse_radius_cap_deg=16`|
|`EPOC_R4_SOURCE_OUTWARD_SHELL`|5|外推shell分离|更强proxy/soft外推、更多虚拟样本、更高energy margin，更严格`radius_inter_ratio_target=0.12`|

## 本地文件变更

|文件|用途|SHA256|
|---|---|---|
|`E:\type10-7\code\scripts\launch_phase1_epoc_r4_teacher_tailq_20260706.sh`|新增R4 source-only teacher-locked tail quarantine启动器|`CCC5134C20049824BEFD3E337795CC7592F275959E3E15982A1FBEA03ADBA80B`|
|`E:\type10-7\code\tests\test_phase1_epoc_r4_teacher_tailq_launcher.py`|新增R4启动器协议与dry-run测试|`964D8F4AB70449B9EC70CB32659D1CF808F759BC430E3FCB8330BB8835323A89`|

本地`E:\type10-7`和`E:\type10-7\code`不是Git仓库；已创建快照：

`E:\type10-7\code\snapshots\phase1_epoc_r4_teacher_tailq_20260706`

## 本地验证

|命令|结果|
|---|---|
|`conda run -n ssr-gpu python -m py_compile code\tests\test_phase1_epoc_r4_teacher_tailq_launcher.py`|PASS|
|`conda run -n ssr-gpu python -m pytest -q code\tests\test_phase1_epoc_r4_teacher_tailq_launcher.py`|PASS，`2 passed`；仅`.pytest_cache`权限warning|
|`bash -n code/scripts/launch_phase1_epoc_r4_teacher_tailq_20260706.sh`|PASS|
|`bash code/scripts/launch_phase1_epoc_r4_teacher_tailq_20260706.sh --dry-run`|PASS；显示`ManySig.pkl`、GPU4/5、ADV3B02教师、tail quarantine和LEO源域视图；未出现`ManyTx.pkl`、`--new_wisig_pkl`或`target_unknown`|

## N607同步与远端验证

|项目|证据|
|---|---|
|preflight|2026-07-06 01:33 CST，`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1` PASS；直连`N607`、项目根目录、GPU可见|
|启动前GPU|GPU4/5/6/7均约`10 MiB`显存；GPU4/5无训练进程；GPU0-3已有R2/R3相关训练，未中断|
|远端根目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|远端碰撞检查|`runs/phase1_epoc_r4_teacher_tailq_20260706`和`logs/phase1_epoc_r4_teacher_tailq_20260706`启动前均不存在|
|同步文件|R4启动器、R4测试、本报告、`code/SYNC_MANIFEST.txt`|
|远端启动前hash|启动器`ccc5134c20049824befd3e337795cc7592f275959e3e15982a1fbea03adba80b`；测试`964d8f4ab70449b9ec70cb32659d1cf808f759bc430e3fcb8330bb8835323a89`；报告`4b4bfb820d1f5c713e5cd99af96980f90ea59bdc7cf7df6d59d403949bb0094c`；manifest`58b32a77f0096552fe42e303fc1f9600a97ec932d33f0b7993d39ec3a0a5a756`|
|远端最终同步校验|2026-07-06 01:42 CST之后再次同步报告和manifest；最终hash以`code/SYNC_MANIFEST.txt`条目和远端`sha256sum`输出为准，避免在报告正文内记录自身hash导致自引用漂移|
|远端验证|`bash -n` PASS；远端dry-run PASS；远端直接测试函数`direct_r4_launcher_tests=PASS`；禁用字段扫描未出现`ManyTx.pkl`、`--new_wisig_pkl`、`target_unknown`|
|SSH/SCP清理|preflight、远端验证、同步、启动前检查、启动命令后均检查本地`ssh.exe`和到`172.31.111.215:22`/`172.31.105.18:22`的ESTABLISHED连接，无残留|

## N607启动记录

启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase1_epoc_r4_teacher_tailq_20260706.sh --only=EPOC_R4_TEACHER_LOCK_TAILQ,EPOC_R4_SOURCE_OUTWARD_SHELL
```

|候选|PID|GPU|日志|输出目录|
|---|---:|---:|---|---|
|`EPOC_R4_TEACHER_LOCK_TAILQ`|`3104085`|4|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r4_teacher_tailq_20260706/EPOC_R4_TEACHER_LOCK_TAILQ.out`|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_r4_teacher_tailq_20260706/EPOC_R4_TEACHER_LOCK_TAILQ`|
|`EPOC_R4_SOURCE_OUTWARD_SHELL`|`3104917`|5|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r4_teacher_tailq_20260706/EPOC_R4_SOURCE_OUTWARD_SHELL.out`|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_r4_teacher_tailq_20260706/EPOC_R4_SOURCE_OUTWARD_SHELL`|

## 启动健康检查

2026-07-06 01:35 CST启动后约1分钟检查：

|候选|进程状态|GPU显存|日志进度|配置证据|错误扫描|早期指标|
|---|---|---:|---|---|---|---|
|`EPOC_R4_TEACHER_LOCK_TAILQ`|`Rl`，elapsed约`01:02`|GPU4约`2179 MiB`|已到`E004/200`，保存`latest_safe_ssdg.pth`和`latest_ssdg.pth`|出现`[CONFIG-LOSS]`、`[CONFIG-TEACHER]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`；教师为`ADV3B02_CORE90_SOFT_E200`；训练数据为`ManySig.pkl`|未见Traceback、RuntimeError、CUDA OOM、unrecognized arguments；未见`ManyTx`、`target_unknown`、`new_wisig`|`VAL tx=98.64%`；早期`TEST overall_tx=nan (0/0)`和未激活proxy/source episode的`nan`记为观察项，不作为失败|
|`EPOC_R4_SOURCE_OUTWARD_SHELL`|`Rl`，elapsed约`00:42`|GPU5约`2237 MiB`|已到`E002/200`，保存`latest_safe_ssdg.pth`和`latest_ssdg.pth`|出现`[CONFIG-LOSS]`、`[CONFIG-TEACHER]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`；教师为`ADV3B02_CORE90_SOFT_E200`；训练数据为`ManySig.pkl`|未见Traceback、RuntimeError、CUDA OOM、unrecognized arguments；未见`ManyTx`、`target_unknown`、`new_wisig`|`VAL tx=98.51%`；早期`TEST overall_tx=nan (0/0)`和未激活proxy/source episode的`nan`记为观察项，不作为失败|

2026-07-06 01:36 CST二次检查：

|候选|进程状态|GPU显存|日志进度|健康结论|
|---|---|---:|---|---|
|`EPOC_R4_TEACHER_LOCK_TAILQ`|`Rl`，elapsed约`02:33`|GPU4约`1781 MiB`|已到`E009/200`；`VAL tx=98.48%`；`JOINT-GUARD safe=1`；`latest_safe_ssdg.pth`持续保存|继续运行；未见Traceback、RuntimeError、CUDA OOM、unrecognized arguments、ManyTx、target_unknown、new_wisig；proxy仍`active=0`符合`proxy_unknown_start_epoch=20`前状态|
|`EPOC_R4_SOURCE_OUTWARD_SHELL`|`Rl`，elapsed约`02:12`|GPU5约`2241 MiB`|已到`E008/200`；`VAL tx=98.60%`；`JOINT-GUARD safe=1`；`latest_safe_ssdg.pth`持续保存|继续运行；未见Traceback、RuntimeError、CUDA OOM、unrecognized arguments、ManyTx、target_unknown、new_wisig；proxy仍`active=0`符合`proxy_unknown_start_epoch=22`前状态|

2026-07-06 01:39 CST启动后约5分钟检查：

|候选|进程状态|日志进度|健康结论|
|---|---|---|---|
|`EPOC_R4_TEACHER_LOCK_TAILQ`|`Rl`，elapsed约`05:45`|已到`E015/200`；`VAL tx=98.42%`；`JOINT-GUARD safe=1`|继续运行；短日志窗口未见Traceback、RuntimeError、CUDA OOM或unrecognized arguments|
|`EPOC_R4_SOURCE_OUTWARD_SHELL`|`Rl`，elapsed约`05:24`|已到`E014/200`；`VAL tx=98.45%`；`JOINT-GUARD safe=1`|继续运行；短日志窗口未见Traceback、RuntimeError、CUDA OOM或unrecognized arguments|

## 观察指标

|阶段|重点指标|判据|
|---|---|---|
|启动|配置标记、epoch开始、无硬错误|启动健康，不等于路线成功|
|E20-E30|`proxy_auc`、`virtual_accept`、`soft_virtual_accept`、`val_tx_acc`|若`proxy_auc`仍低于0.55且`virtual_accept>0.5`，R4应降级为负证据|
|完成后|Phase1 best checkpoint、LEO strict floor、proxy指标、prototype导出|只能作为后续Stage2-C评估输入|
|Stage2-C复评|`old_acc/min_old/seen_new/min_seen/unknown_reject/unknown_FAR`同row|必须同一协同数量行满足目标才可声明推进成功|

## 当前状态

R4已同步N607并启动两个候选。当前证据只支持“source-only底层修复实验已健康启动”，不能声明开集未知拒识目标完成。后续需在E20-E30观察`proxy_auc`、`virtual_accept`、`soft_virtual_accept`和LEO/source指标；完成后必须用导出的Phase2原型回到真实`ManyTx` Stage2-C/qknn8/协同M=1..5评估。

## 复核备注

|问题|处理|
|---|---|
|`--lambda_energy_in`/`--lambda_energy_out`|当前训练图中属于解析参数，不作为R4有效机制证据；R4有效的energy/tail约束来自`proxy_unknown_energy_margin_quantile_weight`、`proxy_unknown_energy_margin_target`、`proxy_unknown_tail_quarantine_weight`等proxy loss入口|
|tail quarantine动态生效|启动健康只证明配置加载和训练运行；要等`proxy_unknown_start_epoch=20/22`之后检查`proxy_unknown_tail_quarantine_loss`、`proxy_auc`、`virtual_accept`和`soft_virtual_accept`|
