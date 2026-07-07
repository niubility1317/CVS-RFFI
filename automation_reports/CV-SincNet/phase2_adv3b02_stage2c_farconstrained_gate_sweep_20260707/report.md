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

## N607运行与结果

|项目|记录|
|---|---|
|同步校验|远端脚本`sha256=20f6899b8f05bca4973c2a27952c234fc3310a20374bcc19a92e14f0de7ec7b5`，与本地一致|
|远端验证|`bash -n`通过；远端`--dry-run`展开40个诊断组合|
|正式命令|`cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase2_adv3b02_stage2c_farconstrained_gate_sweep_20260707.sh`|
|耗时|约284秒|
|运行状态|完成；未启动训练；输出summary JSON/CSV已拉回到`remote_artifacts/`|

### 全量结果表

|variant|gate|K|old_acc|min_old|seen_new|min_seen|unknown_FAR|coverage|unknown_reject|verdict|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|STAGE2C_HEAD_SEP|U074_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U074_M000|10|0.4167|0.0000|0.0000|0.0000|0.0696|0.2071|0.9250|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U074_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U074_MNEG02|10|0.4167|0.0000|0.0000|0.0000|0.0696|0.2071|0.9250|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U078_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U078_M000|10|0.4262|0.0000|0.0000|0.0000|0.0768|0.2112|0.9179|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U078_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U078_MNEG02|10|0.4262|0.0000|0.0000|0.0000|0.0768|0.2112|0.9179|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U082_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U082_M000|10|0.4381|0.0000|0.0000|0.0000|0.0893|0.2184|0.8911|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U082_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U082_MNEG02|10|0.4381|0.0000|0.0000|0.0000|0.0893|0.2184|0.8911|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U086_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U086_M000|10|0.5357|0.0000|0.0000|0.0000|0.1393|0.2673|0.8500|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U086_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U086_MNEG02|10|0.5357|0.0000|0.0000|0.0000|0.1393|0.2673|0.8500|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U090_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U090_M000|10|0.5548|0.0000|0.0000|0.0000|0.1554|0.2806|0.8321|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|U090_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|U090_MNEG02|10|0.5548|0.0000|0.0000|0.0000|0.1554|0.2806|0.8321|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U074_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U074_M000|10|0.4667|0.0000|0.0000|0.0000|0.0786|0.2276|0.9196|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U074_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U074_MNEG02|10|0.4667|0.0000|0.0000|0.0000|0.0786|0.2276|0.9196|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U078_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U078_M000|10|0.4786|0.0000|0.0000|0.0000|0.0821|0.2357|0.9143|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U078_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U078_MNEG02|10|0.4786|0.0000|0.0000|0.0000|0.0821|0.2357|0.9143|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U082_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U082_M000|10|0.4833|0.0000|0.0000|0.0000|0.0875|0.2378|0.9071|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U082_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U082_MNEG02|10|0.4833|0.0000|0.0000|0.0000|0.0875|0.2378|0.9071|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U086_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U086_M000|10|0.5643|0.0000|0.0000|0.0000|0.1196|0.2776|0.8464|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U086_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U086_MNEG02|10|0.5643|0.0000|0.0000|0.0000|0.1196|0.2776|0.8464|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U090_M000|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U090_M000|10|0.6024|0.0000|0.0000|0.0000|0.1679|0.2990|0.8161|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|U090_MNEG02|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|U090_MNEG02|10|0.6024|0.0000|0.0000|0.0000|0.1679|0.2990|0.8161|旧类恢复但FAR超限|

## 解释与下一步

在`unknown_FAR<=0.05`约束下，没有任何候选同时恢复旧类；20个可行行全部为全拒绝，`known_coverage=0`、`old_acc=0`、`seen_new_acc=0`。放宽到`unknown_FAR<=0.10`时，最佳是`STAGE2C_NORM_SEP/U082/K10`，`old_acc=0.4833`、`unknown_FAR=0.0875`，但`min_old=0`且`seen_new=0`。最高旧类恢复是`STAGE2C_NORM_SEP/U090/K10`，`old_acc=0.6024`、`unknown_FAR=0.1679`，仍不满足FAR约束，且最低旧类和seen-new都为0。

本轮结论是：中间unknown/accept gate不能解决qKNNV42真实Stage2-C的“旧类恢复、FAR控制、seen-new注册”三者冲突。下一轮应转向类内最低类保护和seen-new注册侧，而不是继续单调放宽unknown gate；优先尝试按类support半径/密度自适应的per-class accept floor，配合seen-new专用support-to-query二级注册阈值，并保持unknown query eval-only。
