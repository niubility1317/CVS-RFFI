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
