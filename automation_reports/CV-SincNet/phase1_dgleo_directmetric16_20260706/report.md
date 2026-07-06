# phase1_dgleo_directmetric16_20260706

## 基本信息

|字段|内容|
|---|---|
|实验ID|`phase1_dgleo_directmetric16_20260706`|
|时间|2026-07-06|
|操作者|Codex Phase1训练分析/落地agent|
|阶段|Phase1 source-only地面域泛化训练|
|目标|在EPOC concat_sa星地拼接full-loss底座上，直接优化`proxy_vaccept`、`source_overflow`、`bridge_accept_rate`、`low_density_accept_rate`、tail/overflow accept、`radius_to_inter_ratio`、`zid_p50/p95/p99`和`zid_tail_cvar`，同时保护strict UDU、receiver floor和satellite floor。|
|协议边界|训练只使用`ManySig.pkl`源域；不使用`ManyTx`、真实unknown、target receiver、Stage2 query、unknown阈值拟合或Stage2早停。|

## 上轮失败依据

`phase1_dgleo_joint16_20260706`完成16/16个200epoch候选，训练健康和prototype导出完整，但双目标失败：

|维度|现象|含义|
|---|---|---|
|泛化|`J10_BALANCED_B`、`J7_KD_A/B`保住overall、strict UDU、sat floor；弱receiver仍集中在rx11。|EPOC concat_sa+KD能维持闭集DG和星地压力鲁棒性，但没有修复receiver floor短板。|
|拒识代理|final median相对强EPOC参考：`source_overflow`约0.965、`bridge_accept_rate`约1.0、`proxy_vaccept`约0.835、`zid_p95/p99/tail_cvar`明显上升。|known接收域没有收紧，virtual/proxy unknown仍大量落入known尾部或类间低密度区。|
|后期退化|E10到final：`zid_p95`、`zid_p99`、`zid_tail_cvar`、`source_overflow`继续升高。|pseudo+KD保护闭集分类，但把source tail和receiver/channel偏移合法化。|

## 解决策略

本轮不是继续间接调参，而是新增`direct_metric_acceptance_loss`并显式进入总loss：

|目标指标|直接优化方式|保护项|
|---|---|---|
|`zid_p50/p95/p99`、`zid_tail_cvar`|对source known到类中心角度的top-CVaR softplus目标直接反传。|保留CE、KD、SupCon和core accept keep，避免整体塌缩。|
|`source_overflow`|用source domain leave-one-domain episode的soft overflow概率直接压低跨domain溢出。|只用源receiver/day域标签，不接触target receiver。|
|`proxy_vaccept`|从source known构造virtual/shell/bridge/outward negatives，用component-local accept概率直接压低。|virtual unknown只源域合成，不能声明真实unknown FAR。|
|`bridge_accept_rate`|对inter-class bridge hard negative的accept概率做CVaR压制。|保留类间margin和core accept，防止旧类边界样本被整体拒绝。|
|`low_density_accept_rate`|accept概率乘低密度概率后直接压低。|低密度定义来自source core样本邻域。|
|tail/overflow accept|对tail和overflow known样本的自动accept概率做quarantine。|core accept keep保持已知类核心覆盖。|
|`radius_to_inter_ratio`|对样本半径/最近类间角度的CVaR ratio做预算约束。|目标不设到过硬的0.25，本轮主目标0.85，避免泛化崩塌。|
|星地视图|concat_sa clean+sat 2B共同进入所有DG/open损失，并额外优化clean-sat成对`z_id`角度CVaR。|sat CE、sat consistency、domain/ADV/FishR仍同时启用。|

## 本轮矩阵

|组|候选|目标|主要变量|成功标准|失败判据|
|---|---|---|---|---|---|
|P0A|`DGLEO_DM_P0A_CORETAIL_A/B`|压`source_overflow`和tail/overflow accept。|更高`dm_source_w`、`dm_tail_w`、`dm_overflow_w`，B更严格。|`source_overflow<=0.60`，`zid_p99/tail_cvar`较joint16下降，strict UDU/receiver floor下降≤1pp。|p95降但p99/overflow不降；rx11或sat floor崩。|
|P0B|`DGLEO_DM_P0B_BRIDGE_A/B`|压`bridge_accept_rate`和`proxy_vaccept`。|更高`dm_bridge_w`、`dm_proxy_w`和virtual count。|`bridge_accept<0.60`，`proxy_vaccept<=0.65`，泛化floor不显著下降。|bridge仍接近1或proxy仍>0.75。|
|P0D|`DGLEO_DM_P0D_RADIUS_A/B`|压`radius_to_inter_ratio`和known半径。|更高ratio权重和更紧p95/p99目标。|`radius_to_inter_ratio<0.95`且`zid_p95/p99`下降。|ratio下降来自known覆盖塌缩。|
|P1C|`DGLEO_DM_P1C_SATPAIR_A/B`|防止星地增强视图特征散开。|`direct_metric_sat_pair_weight=0.75/1.00`，sat KL/CE更强。|sat floor≥77或相对joint16下降≤0.5pp，同时拒识代理不恶化。|只提升clean，不提升星地视图；sat floor或strict UDU下降。|
|P0C|`DGLEO_DM_P0C_BAL_A/B`|主推进均衡候选。|source/proxy/bridge/tail/overflow/ratio/sat-pair联合。|同时满足泛化floor保护和至少3个拒识代理显著下降。|闭集提升但`source_overflow`、`proxy_vaccept`、bridge仍失败。|
|P1A|`DGLEO_DM_P1A_LATE_A/B`|诊断后期pseudo扩尾。|direct metric晚启动，观察best-final gap。|final不再明显回落，tail指标后期不升。|best强但final退化仍大。|
|P1B|`DGLEO_DM_P1B_FLOOR_A/B`|保护弱receiver floor。|更强domain/FishR/sat损失，拒识权重中等。|rx11和receiver floor不塌，同时拒识代理有下降。|sat mean提升但weak receiver未修复。|
|P0E|`DGLEO_DM_P0E_STRONG_A/B`|拒识强压上界诊断。|最高direct/proxy权重。|若泛化不崩且proxy/bridge显著下降，可作为强约束候选。|非有限梯度、闭集floor崩或known覆盖塌缩。|

## 本地变更

|文件|目的|
|---|---|
|`code/cvsrffi/losses.py`|新增`direct_metric_acceptance_loss`，直接优化source-only拒识代理和星地成对视图。|
|`code/SSDG/train_ssdg.py`|新增CLI参数、loss权重、训练循环接入、stdout和metrics日志字段。|
|`code/scripts/launch_phase1_dgleo_directmetric16_20260706.sh`|新增16候选N607 launcher，每GPU两实验，保留EPOC concat_sa full-loss。|
|`code/tests/test_direct_metric_acceptance_loss.py`|验证direct loss指标、梯度和concat_sa pair约束。|
|`code/tests/test_phase1_dgleo_directmetric16_launcher.py`|验证source-only协议、full concat_sa训练、direct metric参数和GPU排布。|

## 本地验证

|命令|结果|
|---|---|
|`python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py`|通过|
|`bash -n code/scripts/launch_phase1_dgleo_directmetric16_20260706.sh`|通过|
|`bash code/scripts/launch_phase1_dgleo_directmetric16_20260706.sh --dry-run --only=DGLEO_DM_P0C_BAL_A`|通过，输出Phase1 source-only、direct metric、concat_sa full-loss命令|
|`$env:USERPROFILE\.conda\envs\ssr-gpu\python.exe -m pytest -q code/tests/test_direct_metric_acceptance_loss.py`|2 passed；仅`.pytest_cache`写权限警告|
|`$env:USERPROFILE\.conda\envs\ssr-gpu\python.exe -m pytest -q code/tests/test_phase1_dgleo_directmetric16_launcher.py`|3 passed；仅`.pytest_cache`写权限警告|

## N607计划

|字段|内容|
|---|---|
|远端root|`/home/szu2070436088/2510044040/CV-SincNet`|
|运行目录|`runs/phase1_dgleo_directmetric16_20260706/<candidate>`|
|日志目录|`logs/phase1_dgleo_directmetric16_20260706/<candidate>.out`|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|启动命令|`cd /home/szu2070436088/2510044040/CV-SincNet && MAX_ACTIVE_PER_GPU=2 LAUNCH_SETTLE_SECONDS=12 bash code/scripts/launch_phase1_dgleo_directmetric16_20260706.sh`|
|预期输出|`metrics_epoch.csv`、`metrics_epoch.jsonl`、`best_joint_safe_ssdg.pth`、`phase2_zid_prototypes.pt/json`、stdout日志|
|预计时长|参考joint16约6到8小时，P0E强约束候选可能因非有限梯度跳步而变慢或提前暴露失败。|

## 待执行

- 本地非Git目录`E:\type10-7\code`需要创建快照。
- 变更需要镜像到`E:\type10-7\github_publish\CVS-RFFI-repo`并提交。
- N607预检通过后用`scp`同步5个文件，远端做语法/dry-run验证，再启动16候选。
- 启动后4到5分钟检查`[CONFIG-DM-ACCEPT]`、`[CONFIG-CONCAT-SAT]`、`[EPOCH-BEGIN]`、`[EPOCH-END]`、fatal/OOM/NaN和`metrics_epoch.jsonl`写入。
