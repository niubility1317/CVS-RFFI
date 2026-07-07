# qKNNV42真实Stage2-C seen-new rescue诊断

## 基本信息

|字段|内容|
|---|---|
|experiment ID|`phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707`|
|timestamp|2026-07-07|
|operator|Codex|
|objective|在FAR gate和policy-router均未恢复seen-new后，使用新暴露的seen-new rescue/conformal rescue参数，验证是否能把真实Stage2-C的`seen_new_acc`从0拉起，同时观察旧类和unknown FAR代价|
|status|本地验证通过；待同步和N607运行|

## 协议边界

已读取`E:\type10-7\AGENTS.md`和`E:\type10-7\项目.md`。本轮不修改项目协议：

- 仍使用`R_t=7-14`、K=5/K=10、`target_old`、`target_new`、`target_unknown`互斥划分。
- 复用`phase2_adv3b02_stage2c_normsep_protocol_20260707`导出的LEO特征，不重训模型。
- 新参数只使用manifest里的已注册seen-new标签和support/source阈值；`target_unknown`query仍为eval-only，不进入阈值拟合或选择。
- 该sweep是诊断性，不直接声明部署成功。

## 设计

本轮基于提交`7f22258`暴露的wrapper参数：`seen_new_rescue_enabled`、`seen_new_rescue_risk_scale`、`seen_new_rescue_min_*`和`conformal_rescue_*`。

|profile|核心意图|
|---|---|
|`SCORER_SEEN_RESCUE`|用`scorer_cvs`直接验证seen-new rescue是否能降低known候选风险|
|`SCORER_CENTER_SEEN`|在seen-new rescue上叠加`support_center`适配，观察域偏移是否是主因|
|`SCORER_CONFORMAL_SEEN`|同时打开conformal rescue，验证support质量救援对seen-new是否有正效应|
|`SCG_SEEN_RESCUE`|在原`old_protected_unknown_confirm_cvs`路线上只打开seen-new rescue，验证原安全路由是否能受益|

总计2个variant×2个K×4个profile=16个诊断组合。排序优先`seen_new_acc`，再看`min_seen_new_class_acc`、FAR可行性和旧类指标。

## 预期输出

|类型|路径|
|---|---|
|runs|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707/`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707/`|
|summary|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707/stage2c_seennew_rescue_sweep_summary.json`|

## 本地验证

|检查|命令|结果|
|---|---|---|
|wrapper单元测试|`conda activate ssr-gpu; python -m pytest code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py`|通过；5 passed|
|Bash语法|`bash -n code/scripts/launch_phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707.sh`|通过|
|dry-run任务展开|`env ROOT=/tmp/type10_stage2c_seennew_rescue_dryrun SOURCE_RUNS_ROOT=/tmp/type10_stage2c_normsep_source RUNS_ROOT=/tmp/type10_stage2c_seennew_rescue_dryrun/runs LOG_ROOT=/tmp/type10_stage2c_seennew_rescue_dryrun/logs PYTHON=python bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707.sh --dry-run`|通过；展开16个诊断组合|
|根目录/Git承载面一致性|SHA256脚本`b23b07b86a1f7db672f1152880d9c1209b5cef0353f743353c27255c518ff8e2`；report`0cdc2dc7fd543b9bc2fc6e057eb83875b8a3e6e201e8307abb2da94b85a3cf9e`|一致|
