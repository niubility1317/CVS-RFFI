# qKNNV42真实Stage2-C支持集阈值校准sweep

## 基本信息

|字段|内容|
|---|---|
|experiment ID|`phase2_adv3b02_stage2c_supportcal_sweep_20260707`|
|timestamp|2026-07-07|
|operator|Codex|
|objective|复用`phase2_adv3b02_stage2c_normsep_protocol_20260707`已导出的真实Stage2-C特征，诊断是否可以仅用`target_old/target_new`support校准known接收阈值，缓解K5/K10过拒识、旧类覆盖不足和seen-new全0问题|
|status|本地launcher已创建并通过`bash -n`和dry-run；待同步和N607运行|

## 协议边界

已读取`E:\type10-7\AGENTS.md`和`E:\type10-7\项目.md`。本轮不修改项目协议：

- 仍使用`R_t=7-14`、K=5/K=10、`target_old`、`target_new`、`target_unknown`互斥划分。
- 复用上一轮`features_stage2c_leo_repaired.npz`，不重训backbone或adapter。
- 校准只允许使用已登记的known support和source/virtual proxy信息；`Y_unknown`query仍为eval-only，不参与阈值拟合。
- 结果是诊断性sweep，不得直接写作部署成功。

## 设计

上一轮真实Stage2-C显式路线的主要失败模式是过拒识：K5 known coverage为0，K10 known coverage仅0.1480-0.1786；`seen_new_acc=0`。本轮只调整诊断器的support-calibrated accept策略，验证是否能恢复known侧覆盖。

|profile|核心改动|
|---|---|
|`RELAXED_SUPPORT`|使用`leave_one_out`支持集校准、`score_threshold_combine=min`、更宽松unknown risk与class-set gate|
|`CLASS_SCORE_RELAXED`|在`RELAXED_SUPPORT`基础上启用class score threshold|
|`SUPPORT_CENTER_RELAXED`|在`RELAXED_SUPPORT`基础上启用`support_center`特征中心校准|

预期输出：

|类型|路径|
|---|---|
|runs|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_supportcal_sweep_20260707/`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_supportcal_sweep_20260707/`|
|summary|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_supportcal_sweep_20260707/stage2c_supportcal_sweep_summary.json`|

## 本地验证

|命令|结果|
|---|---|
|`bash -n ./code/scripts/launch_phase2_adv3b02_stage2c_supportcal_sweep_20260707.sh`|PASS|
|`bash -lc 'env ROOT=/tmp/type10_stage2c_supportcal_dryrun ... launch_phase2_adv3b02_stage2c_supportcal_sweep_20260707.sh --dry-run'`|PASS，枚举2个variant×3个profile×2个K，共12个诊断组合|

## 同步计划

|local|remote|
|---|---|
|`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_adv3b02_stage2c_supportcal_sweep_20260707.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_supportcal_sweep_20260707.sh`|
