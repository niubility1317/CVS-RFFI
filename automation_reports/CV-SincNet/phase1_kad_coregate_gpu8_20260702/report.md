# Phase1 KAD Core-Gate Known Accept Domain治理实验

## 协议边界

- run_id:`phase1_kad_coregate_gpu8_20260702`
- timestamp:`2026-07-02`
- operator/agent:Codex Phase1地面训练实验分析与实现agent
- scope:Phase1 source-only地面训练。仅能说明闭集DG能力、星地压力鲁棒性、known特征几何、proxy/virtual unknown风险和prototype导出质量。
- 禁止声明:真实unknown_FAR、FPR95、Stage2 old_acc、seen_new_acc、H_old_new或Stage2成功。
- 数据协议:`tx_rx_day_1_7_2`，`labeled_ratio=0.10`，`unlabeled_ratio=0.70`，`source_val_ratio=0.20`。

## 目标与假设

ADG8暴露的问题不是“proxy unknown不够多”，而是known accept域被source tail、bridge和p95组件半径继续撑宽。KAD8将问题改成known accept域治理:

- 只允许core定义自动接收半径和component gate半径；
- tail仍参与分类保真，但不允许扩大accept半径；
- overflow和低密度accepted样本进入source-safe、tail quarantine、bridge CVaR和energy-margin量化治理；
- source episode从three-sigma安全壳改为core-safe episode；
- prototype导出默认使用`p80`半径，并保留tail sentinel但`tail_auto_accept=false`。

## 本地变更

|文件|变更目的|
|---|---|
|`E:\type10-7\code\cvsrffi\losses.py`|新增`component_radius_mode`和`component_radius_quantile`，让proxy accept gate可用core quantile半径；新增`source_episode_radius_mode`、`source_episode_core_quantile`，让source episode可用core-safe半径；补齐`source_overflow`、`proxy_vaccept`、`vaccept_surrogate_CVaR`、`low_density_accept_rate`、`radius_to_inter_ratio`等报告alias。|
|`E:\type10-7\code\cvsrffi\phase2_prototypes.py`|导出`p50/p80/p90`半径；fusion package记录`keep_tail_sentinel`和`tail_auto_accept`；tail sentinel默认`accept_enabled=false`，避免导出包继承宽边界。|
|`E:\type10-7\code\SSDG\train_ssdg.py`|新增CLI透传:`--proxy_unknown_component_radius_mode`、`--proxy_unknown_component_radius_quantile`、`--source_episode_radius_mode`、`--source_episode_core_quantile`、`--source_episode_min_sigma_deg`、`--phase2_fuse_tail_auto_accept`；metrics_epoch/stdout记录gate半径、source-safe半径、low-density accept和alias字段。|
|`E:\type10-7\code\scripts\launch_phase1_kad_coregate_gpu8_20260702.sh`|新增8GPU KAD8实验矩阵，用于检验known accept域治理，不作为直接promotion证据。|
|`E:\type10-7\code\tests\test_open_world_feature_space_loss.py`|新增source episode core-safe半径RED/GREEN测试和`source_overflow`alias断言。|
|`E:\type10-7\code\tests\test_proxy_unknown_loss.py`|新增core radius gate RED/GREEN测试和`proxy_vaccept`、`vaccept_surrogate_CVaR`alias断言。|
|`E:\type10-7\code\tests\test_phase2_train_cli.py`|验证新CLI存在且透传到loss。|
|`E:\type10-7\code\tests\test_phase2_prototypes.py`|验证prototype导出包含`p50/p80/p90`和tail sentinel不自动接收。|
|`E:\type10-7\code\tests\test_phase2_prototype_fusion_export.py`|修正v2 gate schema期望:core component可接收，tail sentinel默认不可接收。|

## 本地验证

|命令|结果|
|---|---|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -B -m pytest -q -p no:cacheprovider tests/test_open_world_feature_space_loss.py tests/test_proxy_unknown_loss.py tests/test_phase2_train_cli.py tests/test_phase2_prototypes.py tests/test_phase2_prototype_fusion_export.py tests/test_zid_compactness_loss.py tests/test_reject_energy_losses.py tests/test_log_nan_parser.py`|`38 passed in 3.55s`|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile cvsrffi\losses.py cvsrffi\phase2_prototypes.py SSDG\train_ssdg.py`|通过|
|`bash -n scripts/launch_phase1_kad_coregate_gpu8_20260702.sh`|通过|
|`bash scripts/launch_phase1_kad_coregate_gpu8_20260702.sh --dry-run --only=KAD8G0_COREGATE_ANCHOR_E200`|通过，核心CLI展开包含core radius、core-safe episode和tail_auto_accept字段|
|`bash scripts/launch_phase1_kad_coregate_gpu8_20260702.sh --dry-run`|通过，8个候选均可展开|
|Git-backed镜像路径同组pytest、py_compile和`bash -n`|通过|

dry-run输出归档:`E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\dry_run_20260702.txt`

## 候选矩阵

|candidate|泛化假设|拒识潜力机制|主要风险|可否Stage2真实unknown评估|下一步动作|
|---|---|---|---|---|---|
|KAD8G0_COREGATE_ANCHOR_E200|在ADV3B02附近保护strict UDU、receiver floor和satellite floor|p80 component gate、core-safe source episode、tail sentinel不自动接收|可能只降低导出半径，训练期bridge仍高|可以，完成后只作为Stage2-A评估候选，不直接promotion|作为KAD8主锚点，比较ADG8G0/ADV3B02|
|KAD8G1_HOLDOUT_STRESS_E200|检验holdout从1到2是否仍能保持DG|观察extra holdout是否重现ADG source overflow|若overflow升高，说明“更多proxy”是负方向|可以，但预期更偏诊断|判定proxy扩张是否应被限制|
|KAD8G2_BRIDGE_CVAR_E200|尽量不牺牲strict/receiver floor|直接压低same/inter bridge accept|bridge权重过高可能伤害跨receiver困难样本|可以，若bridge明显下降且DG不崩|作为bridge治理主诊断|
|KAD8G3_SOURCE_OVERFLOW_E200|保留分类保真，减少source overflow撑边界|source-safe overflow、tail quarantine、core_quantile episode|source episode过窄可能伤害弱receiver|可以，重点看receiver floor|若overflow下降且floor稳定，进入下一轮主组合|
|KAD8G4_LOW_DENSITY_GATE_E200|避免只改善平均值，治理低密度accept|density gate和shell/outward治理|density过严可能拒收old tail|可以，需同时看old closed-set和low_density_accept_rate|判断低密度gate是否必要|
|KAD8G5_ENERGY_MARGIN_Q05_E200|保持闭集同时提升energy下分位安全边界|energy_margin_q05/q10和vaccept CVaR|可能只修proxy，不修source tail|可以，需看source_overflow和proxy_vaccept是否同降|验证energy quantile是否能补component gate|
|KAD8G6_RADIUS_INTER_BUDGET_E200|保护类间分离和receiver floor|压component_radius_p95/max和radius_to_inter_ratio|半径预算过紧可能降低strict UDU|可以，若p99/tail不膨胀|判断导出半径预算是否可主推|
|KAD8G7_COMBINED_SAT_REPAIR_E200|结合治理并保护satellite weak scenario floor|组合bridge、low-density、energy、radius和satellite stress|组合项可能过强，best/final gap或弱receiver退化|可以，但默认diagnostic|只有同row泛化和拒识代理都改善才推进|

## 监控主表字段

泛化字段:overall_tx、strict_udu、receiver_floor、最弱receiver、satellite mean/floor、best-final gap。

拒识潜力字段:`zid_compact_pos_angle_p50/p95/p99`、`zid_tail_cvar`、`source_overflow`、`source_episode_overflow_rate`、`proxy_unknown_bridge_accept_rate`、`proxy_unknown_proxy_vaccept`、`proxy_unknown_vaccept_surrogate_CVaR`、`proxy_unknown_component_gate_radius_p95_deg`、`proxy_unknown_component_gate_radius_max_deg`、`proxy_unknown_radius_to_inter_ratio`、`proxy_unknown_low_density_accept_rate`、`proxy_unknown_energy_margin_q05/q10`、`component_radius_p95/max`、prototype component数量、tail sentinel数量。

## 成功标准

- Phase1同row同时满足:overall/strict UDU不低于ADV3B02附近，receiver floor不下降，satellite floor不低于ADG8G7/ADV3B30可接受边界；
- `bridge_accept_rate`目标低于0.5，若接近1.0则机制失败；
- `source_episode_overflow_rate`目标不高于0.35，若高于0.45则不能主推；
- `proxy_vaccept`、`vaccept_surrogate_CVaR`、`low_density_accept_rate`、`component_gate_radius_p95/max`、`radius_to_inter_ratio`至少两项同向下降；
- prototype导出必须含`p80`半径、`tail_auto_accept=false`、tail sentinel不自动接收；
- final不能明显劣于best，若final回落只允许作为诊断候选。

## 失败判据

- 闭集增强但`source_overflow`、`bridge_accept_rate`、`proxy_vaccept`或p99扩大；
- p95下降但p99/overflow仍高；
- min_inter或radius_to_inter看似改善但bridge_accept仍接近1.0；
- satellite平均提升但弱receiver或satellite floor未修复；
- best checkpoint强但final退化；
- fusion flag存在但导出包无`p80`、无tail sentinel字段或`tail_auto_accept=true`；
- 任何阈值使用target unknown query调参。

## N607计划

远程root:`/home/szu2070436088/2510044040/CV-SincNet`

N607 preflight:

- direct preflight:`tools\n607_ssh_preflight.ps1`失败，原因是direct TCP/SSH path连接拒绝；SSH config和identity检查通过。
- bridge preflight:通过，`user=szu2070436088`，`host=dell-DSS8440`，project_root存在。
- GPU占用:8张RTX3090均为`0%`，显存约`10/24576MiB`；`nvidia-smi pmon`仅有Xorg，无训练计算进程。
- 目标run/log目录:`runs/phase1_kad_coregate_gpu8_20260702`和`logs/phase1_kad_coregate_gpu8_20260702`均不存在。
- Git-backed commit:`b7b915b Add Phase1 known accept domain governance`

需同步文件:

|local|remote|
|---|---|
|`E:\type10-7\code\cvsrffi\losses.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py`|
|`E:\type10-7\code\cvsrffi\phase2_prototypes.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/phase2_prototypes.py`|
|`E:\type10-7\code\SSDG\train_ssdg.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`|
|`E:\type10-7\code\scripts\launch_phase1_kad_coregate_gpu8_20260702.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_kad_coregate_gpu8_20260702.sh`|

启动命令候选:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/code && bash scripts/launch_phase1_kad_coregate_gpu8_20260702.sh
```

实际启动命令:

```bash
mkdir -p /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_coregate_gpu8_20260702 && cd /home/szu2070436088/2510044040/CV-SincNet/code && nohup bash scripts/launch_phase1_kad_coregate_gpu8_20260702.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_coregate_gpu8_20260702/launcher.out 2>&1 & echo launcher_pid=$!
```

启动前必须完成:N607 direct preflight、GPU占用记录、本地快照、scp同步、远程hash/语法检查、无覆盖检查。

远程备份命令:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p code/snapshots/phase1_kad_coregate_20260702_remote_before_sync/{cvsrffi,SSDG,scripts} && cp -p code/cvsrffi/losses.py code/snapshots/phase1_kad_coregate_20260702_remote_before_sync/cvsrffi/losses.py && cp -p code/cvsrffi/phase2_prototypes.py code/snapshots/phase1_kad_coregate_20260702_remote_before_sync/cvsrffi/phase2_prototypes.py && cp -p code/SSDG/train_ssdg.py code/snapshots/phase1_kad_coregate_20260702_remote_before_sync/SSDG/train_ssdg.py && if [ -f code/scripts/launch_phase1_kad_coregate_gpu8_20260702.sh ]; then cp -p code/scripts/launch_phase1_kad_coregate_gpu8_20260702.sh code/snapshots/phase1_kad_coregate_20260702_remote_before_sync/scripts/launch_phase1_kad_coregate_gpu8_20260702.sh; fi
```

同步命令使用bridge `scp`，逐文件同步上述4个训练文件到对应远程路径。

## 当前状态

- local implementation:完成
- local tests:通过
- dry-run:通过
- local snapshot:完成，路径`E:\type10-7\code\snapshots\phase1_kad_coregate_20260702\SHA256SUMS.txt`
- Git-backed mirror/commit:已提交`b7b915b`
- N607 sync/launch:已启动8个KAD8候选；后续accept gate hardening补丁尚未同步远程

## 2026-07-02 accept gate hardening本地补丁

用户要求针对三个误用风险继续优化:three-sigma半径作为accept gate、tail sentinel自动接收、proxy_vaccept改善即被解释成真实拒识改善。

本补丁只修改本地与Git-backed发布仓库，不同步N607，不影响当前已启动的`phase1_kad_coregate_gpu8_20260702`远程作业。远程作业完成后再决定是否用该补丁生成新run_id重跑。

|风险点|修改|
|---|---|
|three-sigma半径作为accept gate|`proxy_unknown_energy_loss`默认`component_radius_mode`从`three_sigma`改为`core_quantile`；`source_episode_three_sigma_loss`默认`radius_mode`从`three_sigma`改为`min_three_sigma_core`。显式`three_sigma`仍保留为诊断对照，不作为默认路径。|
|tail sentinel自动接收|`fuse_tx_domain_prototypes`中tail sentinel的`accept_enabled`强制为`false`；`tail_auto_accept`仅记录为请求字段，新增`tail_auto_accept_requested`和`tail_auto_accept_effective=false`。|
|proxy_vaccept被当作真实拒识改善|新增`proxy_vaccept_proxy_only`和`proxy_reject_claim_allowed=0`，训练日志新增`train/proxy_unknown_proxy_reject_claim_allowed`，stdout显示`reject_claim=0`。|

本地验证:

|命令|结果|
|---|---|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -B -m pytest -q -p no:cacheprovider tests/test_proxy_unknown_loss.py tests/test_open_world_feature_space_loss.py tests/test_phase2_prototypes.py tests/test_phase2_train_cli.py`|`33 passed in 3.31s`|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -B -m pytest -q -p no:cacheprovider tests/test_open_world_feature_space_loss.py tests/test_proxy_unknown_loss.py tests/test_phase2_train_cli.py tests/test_phase2_prototypes.py tests/test_phase2_prototype_fusion_export.py tests/test_local_component_hard_gate.py tests/test_vacuum_gaussian_prototype_bank.py tests/test_zid_compactness_loss.py tests/test_reject_energy_losses.py tests/test_log_nan_parser.py`|`45 passed in 3.76s`|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile cvsrffi\losses.py cvsrffi\phase2_prototypes.py SSDG\train_ssdg.py`|通过|
|Git-backed镜像路径同组pytest和`py_compile`|通过|

本地快照:`E:\type10-7\code\snapshots\accept_gate_hardening_20260702\SHA256SUMS.txt`
