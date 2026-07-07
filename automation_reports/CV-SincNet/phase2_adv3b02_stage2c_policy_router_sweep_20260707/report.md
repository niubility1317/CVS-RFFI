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

## N607运行与结果

|项目|记录|
|---|---|
|同步校验|远端脚本`sha256=230b1c68eb54d9c049ef0d36b5167c911435e6f937c03c0f024e0c52a249f512`，与本地一致|
|远端验证|`bash -n`通过；远端`--dry-run`展开16个诊断组合|
|正式命令|`cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase2_adv3b02_stage2c_policy_router_sweep_20260707.sh`|
|耗时|约130秒|
|运行状态|完成；未启动训练；输出summary JSON/CSV已拉回到`remote_artifacts/`|

### 全量结果表

|variant|profile|K|old_acc|min_old|seen_new|min_seen|unknown_FAR|coverage|unknown_reject|verdict|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|STAGE2C_NORM_SEP|CS_CLASS_SCORE_BAL|10|0.5643|0.0000|0.0000|0.0000|0.1196|0.2776|0.8464|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|KGR_CLASS_SCORE_BAL|10|0.5643|0.0000|0.0000|0.0000|0.1196|0.2776|0.8161|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|CS_CLASS_SCORE_BAL|10|0.5357|0.0000|0.0000|0.0000|0.1393|0.2673|0.8500|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|KGR_CLASS_SCORE_BAL|10|0.5357|0.0000|0.0000|0.0000|0.1393|0.2673|0.8321|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|SR_CENTER_GUARD|10|0.5238|0.0000|0.0000|0.0000|0.1446|0.2663|0.8554|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|SR_CENTER_GUARD|10|0.5143|0.0000|0.0000|0.0000|0.1232|0.2582|0.8768|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|SR_CLASS_SCORE_GUARD|10|0.5024|0.0000|0.0000|0.0000|0.1089|0.2490|0.8911|旧类恢复但FAR超限|
|STAGE2C_NORM_SEP|SR_CLASS_SCORE_GUARD|10|0.4976|0.0000|0.0000|0.0000|0.0929|0.2459|0.9071|旧类恢复但FAR超限|
|STAGE2C_HEAD_SEP|CS_CLASS_SCORE_BAL|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|KGR_CLASS_SCORE_BAL|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|SR_CENTER_GUARD|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_HEAD_SEP|SR_CLASS_SCORE_GUARD|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|CS_CLASS_SCORE_BAL|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|KGR_CLASS_SCORE_BAL|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|SR_CENTER_GUARD|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|SR_CLASS_SCORE_GUARD|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|FAR可行但全拒绝|

## 解释与下一步

policy-router策略没有形成新突破。K5仍全部全拒绝；K10最高为`STAGE2C_NORM_SEP/CS_CLASS_SCORE_BAL`和`STAGE2C_NORM_SEP/KGR_CLASS_SCORE_BAL`，`old_acc=0.5643`、`unknown_FAR=0.1196`，低于supportcal极端relax的旧类恢复，也仍高于FAR约束；所有K10行`min_old=0`且`seen_new=0`。

本轮结论是：仅切换现有fusion policy不能解决qKNNV42真实Stage2-C的新类注册坍塌和最低类过低问题。下一步需要把底层已有的seen-new rescue/支持质量救援参数暴露到`phase2_frozen_manytx_unknown_diagnostic.py`，或新增一个明确区分old保护与seen-new注册的二级决策接口；继续只调`unknown_risk_threshold`或policy名不会产生可用候选。
