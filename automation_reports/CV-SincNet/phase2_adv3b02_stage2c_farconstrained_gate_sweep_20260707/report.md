# qKNNV42真实Stage2-C FAR约束中间gate网格

## 基本信息

|字段|内容|
|---|---|
|experiment ID|`phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707`|
|timestamp|2026-07-07|
|operator|Codex|
|objective|在`phase2_adv3b02_stage2c_supportcal_sweep_20260707`发现“旧类可救但FAR失控”后，复用真实Stage2-C特征，扫描中间unknown/accept gate，寻找`unknown_FAR<=0.05`下的最大旧类恢复点|
|status|本地验证通过；待同步和N607运行|

## 协议边界

已读取`E:\type10-7\AGENTS.md`和`E:\type10-7\项目.md`。本轮不修改项目协议：

- 仍使用`R_t=7-14`、K=5/K=10、`target_old`、`target_new`、`target_unknown`互斥划分。
- 复用`phase2_adv3b02_stage2c_normsep_protocol_20260707`导出的LEO特征，不重训模型。
- 固定`support_calibration_mode=leave_one_out`，阈值只来自known support、source proxy或virtual negatives；`Y_unknown`query仍为eval-only。
- 该网格是诊断性sweep，成功标准是找到可进入下一轮正式路线的FAR受控候选，不直接声明部署成功。

## 设计

上轮极端relax可把K10旧类提升到约0.68，但`unknown_FAR≈0.30`。本轮扫描更保守的中间gate：

|参数|取值|
|---|---|
|`unknown_risk_threshold`|0.74、0.78、0.82、0.86、0.90|
|`candidate_set_unknown_reject_risk`|对应0.76、0.80、0.84、0.88、0.92|
|`accept_margin_threshold`|-0.02、0.00|
|variant|`STAGE2C_NORM_SEP`、`STAGE2C_HEAD_SEP`|
|K|5、10|

总计40个诊断组合。目标排序优先满足`unknown_FAR<=0.05`，再比较`old_acc`、`min_old_class_acc`、`seen_new_acc`。

## 预期输出

|类型|路径|
|---|---|
|runs|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707/`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707/`|
|summary|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707/stage2c_farconstrained_gate_sweep_summary.json`|

## 本地验证

|检查|命令|结果|
|---|---|---|
|Bash语法|`bash -n code/scripts/launch_phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707.sh`|通过|
|dry-run任务展开|`env ROOT=/tmp/type10_stage2c_far_gate_dryrun SOURCE_RUNS_ROOT=/tmp/type10_stage2c_normsep_source RUNS_ROOT=/tmp/type10_stage2c_far_gate_dryrun/runs LOG_ROOT=/tmp/type10_stage2c_far_gate_dryrun/logs PYTHON=python bash code/scripts/launch_phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707.sh --dry-run`|通过；展开40个诊断组合|

## 启动前版本与N607上下文

|项目|记录|
|---|---|
|Git基线|`20ba436 Add Stage2-C FAR constrained gate sweep`|
|Git状态|Git承载面在本任务文件提交后仅剩非本任务untracked目录：`local_artifacts/phase2_adv3b02_proxy_mined_20260704/`、`local_artifacts/phase2_adv3b02_smec_ci_20260704/`|
|N607预检|2026-07-07 11:11 CST，`tools\n607_ssh_preflight.ps1`直连通过；项目根和GPU可见|
|远端占用|8张RTX3090均有既有训练负载，约92%到97%GPU利用率；每GPU已有2个`train_ssdg.py`进程。本轮不干预既有训练，只运行已导出特征上的qKNN诊断|
|磁盘|`/home`可用约7.6T|
|同步映射|`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707.sh` -> `N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707.sh`|
