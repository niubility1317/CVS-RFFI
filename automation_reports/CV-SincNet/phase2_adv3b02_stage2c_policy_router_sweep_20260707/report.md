# qKNNV42真实Stage2-C策略router诊断

## 基本信息

|字段|内容|
|---|---|
|experiment ID|`phase2_adv3b02_stage2c_policy_router_sweep_20260707`|
|timestamp|2026-07-07|
|operator|Codex|
|objective|在中间unknown gate未找到FAR可行旧类恢复后，复用真实Stage2-C LEO特征，比较现有`support_router_cvs`、`candidate_set_cvs`和`known_guarded_rescue_cvs`策略是否能改善旧类最低类和seen-new注册，同时约束unknown FAR|
|status|本地验证通过；待同步和N607运行|

## 协议边界

已读取`E:\type10-7\AGENTS.md`和`E:\type10-7\项目.md`。本轮不修改项目协议：

- 仍使用`R_t=7-14`、K=5/K=10、`target_old`、`target_new`、`target_unknown`互斥划分。
- 复用`phase2_adv3b02_stage2c_normsep_protocol_20260707`导出的LEO特征，不重训模型，不访问unknown query做阈值或选择。
- 所有profile均为诊断性策略组合，成功只表示下一轮可进入正式候选，不声明部署成功。

## 设计

|profile|核心意图|
|---|---|
|`SR_CLASS_SCORE_GUARD`|用`support_router_cvs`和class score阈值，尝试以support确认known、用unknown evidence拒绝unknown|
|`SR_CENTER_GUARD`|在support router上加`support_center`特征适配，观察旧类域适应能否在较低FAR下保留|
|`CS_CLASS_SCORE_BAL`|用`candidate_set_cvs`做candidate集合确认，避免单纯unknown gate放宽|
|`KGR_CLASS_SCORE_BAL`|用`known_guarded_rescue_cvs`尝试在unknown guard下救回known候选|

总计2个variant×2个K×4个profile=16个诊断组合。排序优先`unknown_FAR<=0.05`，再看`old_acc`、`min_old_class_acc`、`seen_new_acc`。

## 预期输出

|类型|路径|
|---|---|
|runs|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_policy_router_sweep_20260707/`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_policy_router_sweep_20260707/`|
|summary|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_policy_router_sweep_20260707/stage2c_policy_router_sweep_summary.json`|

## 本地验证

|检查|命令|结果|
|---|---|---|
|Bash语法|`bash -n code/scripts/launch_phase2_adv3b02_stage2c_policy_router_sweep_20260707.sh`|通过|
|dry-run任务展开|`env ROOT=/tmp/type10_stage2c_policy_router_dryrun SOURCE_RUNS_ROOT=/tmp/type10_stage2c_normsep_source RUNS_ROOT=/tmp/type10_stage2c_policy_router_dryrun/runs LOG_ROOT=/tmp/type10_stage2c_policy_router_dryrun/logs PYTHON=python bash code/scripts/launch_phase2_adv3b02_stage2c_policy_router_sweep_20260707.sh --dry-run`|通过；展开16个诊断组合|
|根目录/Git承载面一致性|SHA256脚本`230b1c68eb54d9c049ef0d36b5167c911435e6f937c03c0f024e0c52a249f512`；report`a11d9b145ee75e573bfa755cf5cc28635bd51af6401dd7975d498506281df9d4`|一致|

## 启动前版本与N607上下文

|项目|记录|
|---|---|
|Git基线|`5231d4c Add Stage2-C policy router sweep`|
|Git状态|Git承载面在本任务文件提交后仅剩非本任务untracked目录：`local_artifacts/phase2_adv3b02_proxy_mined_20260704/`、`local_artifacts/phase2_adv3b02_smec_ci_20260704/`|
|N607预检|2026-07-07 11:28 CST，`tools\n607_ssh_preflight.ps1`直连通过；项目根和GPU可见|
|远端占用|8张RTX3090均有既有训练负载，约95%到99%GPU利用率；`train_ssdg.py`进程族活跃。本轮不干预既有训练，只运行已导出特征上的qKNN诊断|
|磁盘|`/home`可用约7.6T|
|同步映射|`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_adv3b02_stage2c_policy_router_sweep_20260707.sh` -> `N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_policy_router_sweep_20260707.sh`|
