# qKNNV42真实Stage2-C支持集阈值校准sweep

## 基本信息

|字段|内容|
|---|---|
|experiment ID|`phase2_adv3b02_stage2c_supportcal_sweep_20260707`|
|timestamp|2026-07-07|
|operator|Codex|
|objective|复用`phase2_adv3b02_stage2c_normsep_protocol_20260707`已导出的真实Stage2-C特征，诊断是否可以仅用`target_old/target_new`support校准known接收阈值，缓解K5/K10过拒识、旧类覆盖不足和seen-new全0问题|
|status|N607诊断sweep已完成；support-calibrated relaxed策略可把K10旧类提升到约0.68，但unknown_FAR升至约0.30，seen-new仍接近0，未达目标|

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

## 远端验证与运行

远端同步后验证：

|检查|结果|
|---|---|
|`sha256sum code/scripts/launch_phase2_adv3b02_stage2c_supportcal_sweep_20260707.sh`|`619039389dcaac2bf3a24da045fec07842e1834b1681a7728a93541977492b3f`|
|`bash -n code/scripts/launch_phase2_adv3b02_stage2c_supportcal_sweep_20260707.sh`|PASS|
|远端`--dry-run`|PASS，枚举12个诊断组合|

正式运行：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash code/scripts/launch_phase2_adv3b02_stage2c_supportcal_sweep_20260707.sh
```

运行完成，输出：

|类型|路径|
|---|---|
|runs|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_supportcal_sweep_20260707/`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_supportcal_sweep_20260707/`|
|summary JSON|`remote_artifacts/stage2c_supportcal_sweep_summary.json`|
|summary CSV|`remote_artifacts/stage2c_supportcal_sweep_summary.csv`|

## 结果

按`old_acc`排序的主要结果：

|variant|profile|K|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|known_coverage|verdict|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|`STAGE2C_NORM_SEP`|`SUPPORT_CENTER_RELAXED`|10|0.6810|0.0571|0.0125|0.0000|0.3179|0.3765|旧类显著恢复但FAR失控|
|`STAGE2C_NORM_SEP`|`CLASS_SCORE_RELAXED`|10|0.6786|0.0857|0.0268|0.0000|0.2964|0.3816|旧类显著恢复但FAR失控|
|`STAGE2C_NORM_SEP`|`RELAXED_SUPPORT`|10|0.6786|0.0857|0.0089|0.0000|0.2946|0.3704|旧类显著恢复但FAR失控|
|`STAGE2C_HEAD_SEP`|`SUPPORT_CENTER_RELAXED`|10|0.6714|0.1286|0.0125|0.0000|0.3339|0.3816|旧类显著恢复但FAR失控|
|`STAGE2C_HEAD_SEP`|`CLASS_SCORE_RELAXED`|10|0.6619|0.1429|0.0304|0.0000|0.3214|0.3806|旧类显著恢复但FAR失控|
|`STAGE2C_NORM_SEP`|`SUPPORT_CENTER_RELAXED`|5|0.2952|0.0000|0.0000|0.0000|0.0286|0.1388|FAR合格但旧类仍低|

结论：

1. relaxed support calibration证明上轮主要瓶颈确实包含过拒识：K10旧类从0.3738提升到0.6810，known coverage从0.1786提升到0.3765。
2. 该提升依赖大幅放宽unknown risk gate，导致unknown_FAR升到0.2946-0.3339，违反`unknown_FAR<=0.05`，不能作为成功路线。
3. K5在FAR合格配置下旧类仍不足；K10旧类提升明显但最低旧类仍低，`min_old_class_acc`最高仅0.1429。
4. seen-new仍基本坍塌，最高`seen_new_acc=0.0304`且`min_seen_new_class_acc=0`，说明仅放宽known接收不能解决新类注册。

下一步建议：

- 做FAR约束下的中间gate网格，而不是继续极端relax：固定`support_calibration_mode=leave_one_out`，扫`unknown_risk_threshold`约0.74-0.90、`candidate_set_unknown_reject_risk`约0.76-0.92、`accept_margin_threshold`约-0.02到0.02，目标寻找`unknown_FAR<=0.05`下的最大old_acc。
- 同步增加seen-new专门 rescue 或prototype separation，而不是只共享旧类accept门控；当前seen-new全类最低仍为0。
