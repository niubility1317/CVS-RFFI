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

## 启动前版本与N607上下文

|项目|记录|
|---|---|
|Git基线|`0719b6e Add Stage2-C seen-new rescue sweep`；wrapper参数能力来自`7f22258 Expose seen-new rescue knobs in frozen Stage2-C diagnostic`|
|Git状态|Git承载面在本任务文件提交后仅剩非本任务untracked目录：`local_artifacts/phase2_adv3b02_proxy_mined_20260704/`、`local_artifacts/phase2_adv3b02_smec_ci_20260704/`|
|N607预检|2026-07-07 11:43 CST，`tools\n607_ssh_preflight.ps1`直连通过；项目根和GPU可见|
|远端占用|8张RTX3090均有既有训练负载，约95%到99%GPU利用率。本轮不干预既有训练，只运行已导出特征上的qKNN诊断|
|磁盘|`/home`可用约7.6T|
|同步映射|`code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`和`code/scripts/launch_phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707.sh`同步到N607同名路径|

## N607运行与结果

|项目|记录|
|---|---|
|同步校验|远端wrapper`sha256=2f5ac67cbe1de6f0a9a89e92675091bdf42f10ff461de5402df8ae65747b33c3`；远端launcher`sha256=b23b07b86a1f7db672f1152880d9c1209b5cef0353f743353c27255c518ff8e2`；均与本地一致|
|远端验证|wrapper`--help`可见`--seen_new_rescue_enabled`；launcher`bash -n`通过；远端`--dry-run`展开16个诊断组合|
|正式命令|`cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707.sh`|
|耗时|约140秒|
|运行状态|完成；未启动训练；输出summary JSON/CSV已拉回到`remote_artifacts/`|

### 全量结果表

|variant|profile|K|old_acc|min_old|seen_new|min_seen|unknown_FAR|coverage|unknown_reject|defer|verdict|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|STAGE2C_NORM_SEP|SCORER_CONFORMAL_SEEN|10|0.8024|0.5143|0.3875|0.1857|0.9643|0.9918|0.0000|0.0182|seen-new拉起但FAR失控|
|STAGE2C_NORM_SEP|SCORER_SEEN_RESCUE|10|0.6119|0.0000|0.3875|0.1857|0.6679|0.8133|0.0000|0.2396|seen-new拉起但FAR失控|
|STAGE2C_NORM_SEP|SCORER_CENTER_SEEN|10|0.6310|0.0000|0.3768|0.1571|0.6714|0.8214|0.0000|0.2331|seen-new拉起但FAR失控|
|STAGE2C_HEAD_SEP|SCORER_CONFORMAL_SEEN|10|0.7929|0.6000|0.3536|0.2143|0.9554|0.9959|0.0000|0.0188|seen-new拉起但FAR失控|
|STAGE2C_HEAD_SEP|SCORER_SEEN_RESCUE|10|0.5667|0.0000|0.3536|0.2143|0.6107|0.7704|0.0000|0.2877|seen-new拉起但FAR失控|
|STAGE2C_HEAD_SEP|SCORER_CENTER_SEEN|10|0.5714|0.0143|0.3518|0.2286|0.6268|0.7816|0.0000|0.2747|seen-new拉起但FAR失控|
|STAGE2C_HEAD_SEP|SCORER_CENTER_SEEN|5|0.0000|0.0000|0.3107|0.1429|0.2804|0.4245|0.0000|0.6279|K5 seen-new拉起但旧类/FAR失败|
|STAGE2C_HEAD_SEP|SCORER_CONFORMAL_SEEN|5|0.8071|0.6286|0.2893|0.0714|0.8696|0.9388|0.0000|0.0864|K5旧类强但FAR失控|
|STAGE2C_HEAD_SEP|SCORER_SEEN_RESCUE|5|0.0000|0.0000|0.2893|0.0714|0.2964|0.4122|0.0000|0.6299|K5 seen-new拉起但旧类/FAR失败|
|STAGE2C_NORM_SEP|SCORER_CONFORMAL_SEEN|5|0.8048|0.6286|0.2714|0.0429|0.8696|0.9469|0.0000|0.0812|K5旧类强但FAR失控|
|STAGE2C_NORM_SEP|SCORER_CENTER_SEEN|5|0.0000|0.0000|0.2714|0.0714|0.3143|0.4082|0.0000|0.6260|K5 seen-new拉起但旧类/FAR失败|
|STAGE2C_NORM_SEP|SCORER_SEEN_RESCUE|5|0.0000|0.0000|0.2714|0.0429|0.3214|0.4082|0.0000|0.6234|K5 seen-new拉起但旧类/FAR失败|
|STAGE2C_HEAD_SEP|SCG_SEEN_RESCUE|10|0.5667|0.0000|0.0036|0.0000|0.1679|0.2898|0.8321|0.0000|安全路由仍抑制seen-new|
|STAGE2C_NORM_SEP|SCG_SEEN_RESCUE|10|0.6119|0.0000|0.0000|0.0000|0.1839|0.3031|0.8161|0.0000|安全路由仍抑制seen-new|
|STAGE2C_HEAD_SEP|SCG_SEEN_RESCUE|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|0.0000|FAR可行但全拒绝|
|STAGE2C_NORM_SEP|SCG_SEEN_RESCUE|5|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|1.0000|0.0000|FAR可行但全拒绝|

## 解释与下一步

本轮证明seen-new rescue通道本身有效：`seen_new_acc`从前两轮的0提升到最高0.3875，且`SCORER_CONFORMAL_SEEN`同时把旧类域适应提升到`old_acc≈0.80`、`min_old≈0.51-0.60`。但该收益来自过度accept，`unknown_FAR=0.8696-0.9643`，不能作为部署候选或成功证据。

当前最有信息量的方向是把`SCORER_CONFORMAL_SEEN`的known恢复能力与SCG/unknown二源确认结合：保留seen-new/conformal rescue作为候选生成器，但增加unknown二级veto或pairguard，只对高support质量的seen-new开口，同时恢复unknown拒识。下一轮不应再做单纯救援放宽；应做“rescue候选+unknown veto”混合门控，目标是在`seen_new_acc>0`下把`unknown_FAR`压回可控区间。
