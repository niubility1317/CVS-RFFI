# Phase1 EPOC-R5 Proxy Accept Crush Report

## 基本信息

|项目|值|
|---|---|
|run_id|`phase1_epoc_r5_proxy_accept_crush_20260706`|
|时间|2026-07-06|
|operator|Codex|
|目标|在`ADV3B02_CORE90_SOFT_E200`教师约束下，针对R4早期proxy未知仍被旧类接受的问题，进一步压低source-only虚拟未知接受率|
|定位|Phase1/source-only底层修复；不是Stage2-C成功，不是部署成功，不是真实未知类训练|
|触发证据|R4 E24-E29：`proxy_auc<0.55`且`virtual_accept>0.82`，`soft_virtual_accept≈1.0`；旧类source验证仍约98.6%|

## 协议边界

|边界|执行方式|
|---|---|
|训练数据|只加载`ManySig.pkl`；不加载`ManyTx.pkl`；不使用`--new_wisig_pkl`|
|真实未知类|`Y_unknown`和`target_unknown`不进入训练、阈值、prototype、adapter、profile、receiver权重或early stopping|
|目标接收机|地面训练不接触`R_t`样本、统计、BN、阈值、prototype、adapter或验证结果|
|底座/教师|`ADV3B02_CORE90_SOFT_E200`同时作为`baseline_ckpt`和`teacher_ckpt`|
|LEO视图|只使用源域派生的`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|无效机制排除|不使用`--lambda_energy_in`或`--lambda_energy_out`作为R5机制证据；R5只依赖训练图实际消费的proxy loss入口|

## R5设计假设

R4未破坏旧类source验证，但proxy虚拟未知仍高度被旧类接受。R5不继续增加parsed-only能量参数，而是直接加大实际进入`proxy_unknown_energy_loss`的accept/tail约束：

1. 增大`proxy_unknown_tail_quarantine_weight`、`bridge_accept_weight`、`shell_outward_accept_weight`、`energy_margin_quantile_weight`和`radius_inter_ratio_weight`。
2. 把`bridge_accept_target`压到`0.01/0.00`，把`tail_accept_target`压到`0.05/0.03`。
3. 降低`phase2_fuse_radius_cap_deg`到`14/12`，避免tail样本继续进入自动接受半径。
4. 保持较强ADV3B02教师蒸馏和`joint_safe`守护，先看旧类source验证是否还能维持。

## 候选矩阵

|候选|GPU|定位|关键机制|
|---|---:|---|---|
|`EPOC_R5_BRIDGE_CRUSH`|6|桥接接受率压制|`bridge_accept_target=0.01`、`tail_accept_target=0.05`、`proxy_virtual_count=64`、`ow_feat_radius_deg=12`、`phase2_fuse_radius_cap_deg=14`|
|`EPOC_R5_CORE_SHELL_REJECT`|7|core/shell更强拒识|`bridge_accept_target=0.00`、`tail_accept_target=0.03`、`proxy_virtual_count=96`、`ow_feat_radius_deg=10`、`phase2_fuse_radius_cap_deg=12`|

## 本地文件变更

|文件|用途|SHA256|
|---|---|---|
|`E:\type10-7\code\scripts\launch_phase1_epoc_r5_proxy_accept_crush_20260706.sh`|新增R5 source-only proxy accept crush启动器|`2A1D95DE4635BBDADB74C595F7B1032F8CBBBA10DE1442FE70B38F0F913660E6`|
|`E:\type10-7\code\tests\test_phase1_epoc_r5_proxy_accept_crush_launcher.py`|新增R5启动器协议与dry-run测试|`DB115555F6021CD92DDC353F9E1F33B89DC412D1264B6B73E218FB0214DB71F0`|

本地`E:\type10-7`和`E:\type10-7\code`不是Git仓库；已创建快照：

`E:\type10-7\code\snapshots\phase1_epoc_r5_proxy_accept_crush_20260706`

## 本地验证

|命令|结果|
|---|---|
|`conda run -n ssr-gpu python -m pytest -q code\tests\test_phase1_epoc_r5_proxy_accept_crush_launcher.py`|RED：脚本不存在导致`2 failed`；GREEN：脚本实现后`2 passed`，仅`.pytest_cache`权限warning|
|`bash -n code/scripts/launch_phase1_epoc_r5_proxy_accept_crush_20260706.sh`|PASS|
|`conda run -n ssr-gpu python -m py_compile code\tests\test_phase1_epoc_r5_proxy_accept_crush_launcher.py`|PASS|
|`bash code/scripts/launch_phase1_epoc_r5_proxy_accept_crush_20260706.sh --dry-run`|PASS；显示`ManySig.pkl`、GPU6/7、ADV3B02教师、proxy accept crush和LEO源域视图；未出现`ManyTx.pkl`、`--new_wisig_pkl`、`target_unknown`、`--lambda_energy_in`或`--lambda_energy_out`|

## N607同步与远端验证

|项目|证据|
|---|---|
|preflight与容量|2026-07-06 01:48 CST preflight PASS；启动前GPU6/7均约`10 MiB`显存，无训练进程；R2/R3/R4继续monitor-only|
|远端根目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|远端碰撞检查|`runs/phase1_epoc_r5_proxy_accept_crush_20260706`和`logs/phase1_epoc_r5_proxy_accept_crush_20260706`启动前均不存在|
|同步文件|R5启动器、R5测试、本报告、`code/SYNC_MANIFEST.txt`|
|远端hash|启动器`2a1d95de4635bbdadb74c595f7b1032f8cbbba10de1442fe70b38f0f913660e6`；测试`db115555f6021cd92ddc353f9e1f33b89dc412d1264b6b73e218fb0214db71f0`；报告`7c1fdea9fa8a72e97a16b09a035f6459a31074c75bf4e7affb607ab705eff132`；manifest`88354b7fe36ea1f43cf4c650e97d5b704c97c9fe13d2b32db152e4d0dae64926`|
|远端验证|`bash -n` PASS；远端dry-run PASS；远端直接测试函数`direct_r5_launcher_tests=PASS`；dry-run未出现`ManyTx.pkl`、`--new_wisig_pkl`、`target_unknown`、`--lambda_energy_in`或`--lambda_energy_out`|
|SSH/SCP清理|preflight、同步、远端验证、启动前检查、启动命令后均检查本地`ssh.exe`和到`172.31.111.215:22`/`172.31.105.18:22`的ESTABLISHED连接，无残留|

## N607启动记录

启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase1_epoc_r5_proxy_accept_crush_20260706.sh --only=EPOC_R5_BRIDGE_CRUSH,EPOC_R5_CORE_SHELL_REJECT
```

|候选|PID|GPU|日志|输出目录|
|---|---:|---:|---|---|
|`EPOC_R5_BRIDGE_CRUSH`|`3118232`|6|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r5_proxy_accept_crush_20260706/EPOC_R5_BRIDGE_CRUSH.out`|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_r5_proxy_accept_crush_20260706/EPOC_R5_BRIDGE_CRUSH`|
|`EPOC_R5_CORE_SHELL_REJECT`|`3118648`|7|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r5_proxy_accept_crush_20260706/EPOC_R5_CORE_SHELL_REJECT.out`|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_r5_proxy_accept_crush_20260706/EPOC_R5_CORE_SHELL_REJECT`|

## 启动健康检查

2026-07-06 02:01 CST启动后约1分钟检查：

|候选|进程状态|GPU显存|日志进度|配置证据|错误扫描|早期指标|
|---|---|---:|---|---|---|---|
|`EPOC_R5_BRIDGE_CRUSH`|`Rl`，elapsed约`01:37`|GPU6约`2241 MiB`|已到`E006/200`，保存`latest_safe_ssdg.pth`和`latest_ssdg.pth`|出现`[CONFIG-LOSS]`、`[CONFIG-TEACHER]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`；教师为`ADV3B02_CORE90_SOFT_E200`；训练数据为`ManySig.pkl`|未见Traceback、RuntimeError、CUDA OOM、unrecognized arguments；未见`ManyTx`、`target_unknown`、`new_wisig`、`lambda_energy_in/out`|`VAL tx`约98.25%-98.59%；`JOINT-GUARD safe=1`；proxy尚未激活，符合`proxy_unknown_start_epoch=18`前状态|
|`EPOC_R5_CORE_SHELL_REJECT`|`Rl`，elapsed约`01:17`|GPU7约`2045 MiB`|已到`E005/200`，保存`latest_safe_ssdg.pth`和`latest_ssdg.pth`|出现`[CONFIG-LOSS]`、`[CONFIG-TEACHER]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`；教师为`ADV3B02_CORE90_SOFT_E200`；训练数据为`ManySig.pkl`|未见Traceback、RuntimeError、CUDA OOM、unrecognized arguments；未见`ManyTx`、`target_unknown`、`new_wisig`、`lambda_energy_in/out`|`VAL tx`约98.15%-98.67%；`JOINT-GUARD safe=1`；proxy尚未激活，符合`proxy_unknown_start_epoch=18`前状态|

## 观察指标

|阶段|重点指标|判据|
|---|---|---|
|启动|配置标记、epoch开始、无硬错误|启动健康，不等于路线成功|
|E20-E30|`proxy_auc`、`virtual_accept`、`soft_virtual_accept`、`val_tx_acc`|若`virtual_accept`仍约0.8且`proxy_auc<0.55`，R5仍为负证据；若`virtual_accept`明显下降且`val_tx`不崩，继续观察|
|完成后|Phase1 best checkpoint、LEO strict floor、proxy指标、prototype导出|只能作为后续Stage2-C评估输入|
|Stage2-C复评|`old_acc/min_old/seen_new/min_seen/unknown_reject/unknown_FAR`同row|必须同一协同数量行满足目标才可声明推进成功|

## 当前状态

R5已同步N607并启动两个候选。当前证据只支持“source-only底层修复实验已健康启动”，不能声明开集未知拒识目标完成。关键判断点是E20-E30的`proxy_auc`、`virtual_accept`、`soft_virtual_accept`和`val_tx_acc`。
