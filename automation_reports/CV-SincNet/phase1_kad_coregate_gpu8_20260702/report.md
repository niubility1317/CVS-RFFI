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

|candidate|泛化假设|拒识潜力机制|主要风险|后续Stage2-A真实unknown评估资格边界（当前Phase1不产生unknown_FAR/FPR95）|下一步动作|
|---|---|---|---|---|---|
|KAD8G0_COREGATE_ANCHOR_E200|在ADV3B02附近保护strict UDU、receiver floor和satellite floor|p80 component gate、core-safe source episode、tail sentinel不自动接收|可能只降低导出半径，训练期bridge仍高|若完成后双目标达标，仅可列入后续独立Stage2-A评估候选，当前结果不得替代unknown_FAR/FPR95/AUROC|作为KAD8主锚点，比较ADG8G0/ADV3B02|
|KAD8G1_HOLDOUT_STRESS_E200|检验holdout从1到2是否仍能保持DG|观察extra holdout是否重现ADG source overflow|若overflow升高，说明“更多proxy”是负方向|仅诊断proxy扩张风险；当前结果不得替代unknown_FAR/FPR95/AUROC|判定proxy扩张是否应被限制|
|KAD8G2_BRIDGE_CVAR_E200|尽量不牺牲strict/receiver floor|直接压低same/inter bridge accept|bridge权重过高可能伤害跨receiver困难样本|若bridge明显下降且DG不崩，仅可作为后续独立Stage2-A评估候选|作为bridge治理主诊断|
|KAD8G3_SOURCE_OVERFLOW_E200|保留分类保真，减少source overflow撑边界|source-safe overflow、tail quarantine、core_quantile episode|source episode过窄可能伤害弱receiver|若overflow下降且floor稳定，仅可列入后续独立Stage2-A评估候选|若overflow下降且floor稳定，进入下一轮主组合|
|KAD8G4_LOW_DENSITY_GATE_E200|避免只改善平均值，治理低密度accept|density gate和shell/outward治理|density过严可能拒收old tail|需同时看old closed-set和low_density_accept_rate；当前只给Phase1代理证据|判断低密度gate是否必要|
|KAD8G5_ENERGY_MARGIN_Q05_E200|保持闭集同时提升energy下分位安全边界|energy_margin_q05/q10和vaccept CVaR|可能只修proxy，不修source tail|proxy-only风险必须保留；当前结果不得替代unknown_FAR/FPR95/AUROC|验证energy quantile是否能补component gate|
|KAD8G6_RADIUS_INTER_BUDGET_E200|保护类间分离和receiver floor|压component_radius_p95/max和radius_to_inter_ratio|半径预算过紧可能降低strict UDU|若p99/tail不膨胀，仅可作为后续独立Stage2-A评估候选|判断导出半径预算是否可主推|
|KAD8G7_COMBINED_SAT_REPAIR_E200|结合治理并保护satellite weak scenario floor|组合bridge、low-density、energy、radius和satellite stress|组合项可能过强，best/final gap或弱receiver退化|只有同row泛化和拒识代理都改善，才列入后续独立Stage2-A评估候选|只有同row泛化和拒识代理都改善才推进|

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

## 2026-07-02 KAD16每GPU双槽实验设计

用户要求设计16个实验，包括当前已启动的KAD8，并按每张卡两个实验组织。已新增本地矩阵报告:

`E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\kad16_hardening_matrix.md`

当前边界:

- KAD8:`phase1_kad_coregate_gpu8_20260702`，按启动记录已在N607启动8个候选，每GPU一条；该批次仍是accept gate hardening之前的远程代码证据。
- KAD16H:`phase1_kad_hardening_secondlane_gpu8_20260702`，新增8个硬化验证候选，每GPU第二槽一条；本轮只完成本地launcher和矩阵设计，未同步N607、未启动。
- 新launcher:`E:\type10-7\code\scripts\launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`。
- dry-run归档:`E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\dry_run_hardening_secondlane_20260702.txt`。

新增launcher安全语义:

- 非dry-run会创建`${LOG_ROOT}/.launcher.lock`，拒绝重复提交同一second-lane批次。
- 非dry-run会拒绝`RUN_ID`、`RUNS_ROOT`或`LOG_ROOT`指向`phase1_kad_coregate_gpu8_20260702`，避免污染当前KAD8目录。
- 若任一卡已有2条训练进程，launcher只等待，不应继续追加第三条。

KAD16H启动前硬门槛:

1. 先同步本地硬化补丁和新增launcher到N607，至少包括`cvsrffi/losses.py`、`cvsrffi/phase2_prototypes.py`、`SSDG/train_ssdg.py`、`code/scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`。
2. 记录精确scp映射、远端SHA256、远端`py_compile`、远端`bash -n`、远端dry-run8候选计数和非dry-run guard结果。
3. 记录当前KAD8真实compute进程证据:PID、cwd、cmdline、`CUDA_VISIBLE_DEVICES`、GPU号、log路径；不能只看旧preflight或launcher父进程。
4. 确认远端`runs/phase1_kad_hardening_secondlane_gpu8_20260702`和`logs/phase1_kad_hardening_secondlane_gpu8_20260702`不存在；若存在candidate目录、`.out`、`.pid`、`launcher.out`或`.launcher.lock`，停止。
5. 启动后补写远程命令、launcher PID、每候选PID/GPU/log、SSH/SCP断连清理、启动后4-5分钟startup health检查。

本地验证:

|命令|结果|
|---|---|
|`bash -n scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`|通过|
|`bash scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh --dry-run`|通过，8个KAD16H候选均可展开|

新增8行的核心目的不是补proxy unknown数量，而是验证三类修改后的机制:默认不再把three-sigma半径作为accept gate、tail sentinel即使命令请求也不能自动接收、proxy_vaccept改善必须被标记为proxy-only证据且不能直接声明真实unknown拒识改善。`phase2_export_prototypes`和`phase2_fuse_prototypes`只表示源域prototype导出和后续独立Stage2-A评估准备，不表示Stage2已经运行、Stage2成功或部署证据成立。

## 2026-07-02 slot B启动执行记录

用户明确要求启动slot B，即`phase1_kad_hardening_secondlane_gpu8_20260702`的8个KAD16H候选，每GPU第二槽一条。

启动前本地状态:

|项目|证据|
|---|---|
|协议边界|已重新读取`AGENTS.md`、`项目.md`和`cv-sincnet-n607-automation`技能；本次仍为Phase1 source-only，不声明真实`unknown_FAR`、`FPR95`或Stage2成功。|
|Git-backed发布仓库|`E:\type10-7\github_publish\CVS-RFFI-repo`工作区干净；当前HEAD=`084e39e`；KAD16矩阵提交`a2408d3`已在当前历史中。|
|本地验证|`py_compile cvsrffi\losses.py cvsrffi\phase2_prototypes.py SSDG\train_ssdg.py`通过；`bash -n scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`通过；slot B launcher dry-run展开8个KAD16H候选。|
|本地SSH清理|preflight后发现2个旧bridge SSH进程，PID`23492`和`12748`，已关闭；复查`ssh_processes=none`，`n607_established_ssh=none`，`bridge_established_ssh=none`。|

N607 direct preflight:

- 时间:`2026-07-02T17:21:38+08:00`。
- direct target:`N607`，user=`szu2070436088`，host=`dell-DSS8440`。
- project root存在:`/home/szu2070436088/2510044040/CV-SincNet`。
- GPU 0-7均为RTX3090，preflight时显存约`2379-2513MiB`，有KAD8训练占用。

远端slot A/slot B状态:

|GPU|KAD8主compute PID|candidate|slot B状态|
|---|---:|---|---|
|0|4063914|`KAD8G0_COREGATE_ANCHOR_E200`|可追加第二槽|
|1|4059597|`KAD8G1_HOLDOUT_STRESS_E200`|可追加第二槽|
|2|4059609|`KAD8G2_BRIDGE_CVAR_E200`|可追加第二槽|
|3|4059602|`KAD8G3_SOURCE_OVERFLOW_E200`|可追加第二槽|
|4|4059613|`KAD8G4_LOW_DENSITY_GATE_E200`|可追加第二槽|
|5|4059619|`KAD8G5_ENERGY_MARGIN_Q05_E200`|可追加第二槽|
|6|4059603|`KAD8G6_RADIUS_INTER_BUDGET_E200`|可追加第二槽|
|7|4059620|`KAD8G7_COMBINED_SAT_REPAIR_E200`|可追加第二槽|

远端目标目录检查:

- `runs/phase1_kad_hardening_secondlane_gpu8_20260702`:不存在。
- `logs/phase1_kad_hardening_secondlane_gpu8_20260702`:不存在。
- `.launcher.lock`:不存在。
- `code/scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`:启动前尚未同步。

下一步动作:远端备份当前代码文件、同步本地硬化代码和slot B launcher、远端hash/语法/py_compile/dry-run验证，通过后启动slot B。

同步与远端验证:

|项目|证据|
|---|---|
|远端备份|`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/kad16_hardening_secondlane_remote_before_sync_20260702_1725`|
|备份前远端hash|`losses.py=d2efb26e...`；`phase2_prototypes.py=8fee38f0...`；`train_ssdg.py=3b35c90f...`；slot B launcher启动前不存在。|
|同步映射|`E:\type10-7\code\cvsrffi\losses.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py`；`phase2_prototypes.py`同路径映射；`SSDG\train_ssdg.py`同路径映射；`scripts\launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh` -> `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`。|
|同步后远端hash|`losses.py=bf723258...`；`phase2_prototypes.py=8c5a3758...`；`train_ssdg.py=4bad0466...`；launcher=`57b38584...`，与本地SHA256一致。|
|远端guard|`component_radius_default_ok`、`source_episode_default_ok`、`tail_effective_field_ok`、`proxy_claim_boundary_ok`、`launcher_lock_ok`均通过。|
|远端语法/编译|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/cvsrffi/losses.py code/cvsrffi/phase2_prototypes.py code/SSDG/train_ssdg.py`通过；`bash -n code/scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`通过。|
|远端dry-run|`dryrun_candidate_count=8`，`[KAD16H-DONE] run_id=phase1_kad_hardening_secondlane_gpu8_20260702`。|
|远端目标清洁|同步后启动前仍为`runs_missing_ok`、`logs_missing_ok`。|
|SSH状态|远端验证后复查`ssh_processes=none`、`n607_established_ssh=none`。|

slot B启动结果:

- 本地启动SSH命令在30秒超时，不能直接作为成功/失败判据；随后关闭本地残留SSH PID`22844`并复查`n607_established_ssh=none`。
- 远端只读探针确认slot B已启动，launcher日志已写出8个`[KAD16H-LAUNCHED]`。
- 远端launcher父命令曾显示为PID`4153791`，主launcher脚本PID`4153793`，候选子launcher PID`4153802-4153809`。

|GPU|candidate|PID|log|
|---|---|---:|---|
|0|`KAD16H0_HARDENED_DEFAULT_ANCHOR_E200`|4153878|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_hardening_secondlane_gpu8_20260702/KAD16H0_HARDENED_DEFAULT_ANCHOR_E200.out`|
|1|`KAD16H1_THREESIGMA_NEGCTRL_E200`|4153872|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_hardening_secondlane_gpu8_20260702/KAD16H1_THREESIGMA_NEGCTRL_E200.out`|
|2|`KAD16H2_BRIDGE_COREQ75_E200`|4153875|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_hardening_secondlane_gpu8_20260702/KAD16H2_BRIDGE_COREQ75_E200.out`|
|3|`KAD16H3_SOURCE_COREQ75_QUAR_E200`|4153863|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_hardening_secondlane_gpu8_20260702/KAD16H3_SOURCE_COREQ75_QUAR_E200.out`|
|4|`KAD16H4_TAIL_SENTINEL_GUARD_E200`|4153873|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_hardening_secondlane_gpu8_20260702/KAD16H4_TAIL_SENTINEL_GUARD_E200.out`|
|5|`KAD16H5_PROXY_ONLY_BOUNDARY_E200`|4153889|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_hardening_secondlane_gpu8_20260702/KAD16H5_PROXY_ONLY_BOUNDARY_E200.out`|
|6|`KAD16H6_P80_RADIUS_BUDGET_E200`|4153859|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_hardening_secondlane_gpu8_20260702/KAD16H6_P80_RADIUS_BUDGET_E200.out`|
|7|`KAD16H7_HARDENED_COMBINED_SAT_E200`|4153879|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_kad_hardening_secondlane_gpu8_20260702/KAD16H7_HARDENED_COMBINED_SAT_E200.out`|

startup health检查:

- 检查时间:`2026-07-02T17:32:59+08:00`至`2026-07-02T17:34:47+08:00`。
- GPU状态:0-7每卡均有KAD8 slot A和KAD16H slot B各1个compute进程，符合每GPU两个训练实验的上限。
- KAD16H liveness:8个`.pid`文件对应进程均存活，status均为`running`，启动时间均为`2026-07-02T17:26:59+08:00`，`evidence_role=diagnostic`且`promotion_allowed=false`。
- 训练进度:8个候选均出现`[CONFIG-LOSS]`、`[CONFIG-SAT]`和`[EPOCH-BEGIN]`，启动健康窗口内已到约`E011/200`并写出checkpoint。
- clean fatal scan:`Traceback|RuntimeError|unrecognized argument|CUDA out of memory|fatal|Killed|No such file|ImportError|ModuleNotFoundError`未命中，输出`fatal_scan_clean`。
- SSH状态:启动和health探针后均已复查，本地`ssh_process_count=0`，到N607或bridge的`ESTABLISHED:22`连接数为0。

startup health窗口内的同row早期闭集快照只用于确认训练在推进，不能作为最终排榜:

|GPU|candidate|startup epoch|val_tx|test_tx|checkpoint/状态|
|---|---|---:|---:|---:|---|
|0|`KAD16H0_HARDENED_DEFAULT_ANCHOR_E200`|10|91.93%|80.29%|checkpoint已写，进程存活|
|1|`KAD16H1_THREESIGMA_NEGCTRL_E200`|10|90.17%|78.96%|checkpoint已写，进程存活|
|2|`KAD16H2_BRIDGE_COREQ75_E200`|10|90.33%|80.44%|checkpoint已写，进程存活|
|3|`KAD16H3_SOURCE_COREQ75_QUAR_E200`|10|91.62%|80.90%|checkpoint已写，进程存活|
|4|`KAD16H4_TAIL_SENTINEL_GUARD_E200`|10|87.01%|75.85%|checkpoint已写，进程存活|
|5|`KAD16H5_PROXY_ONLY_BOUNDARY_E200`|10|89.89%|80.09%|checkpoint已写，进程存活|
|6|`KAD16H6_P80_RADIUS_BUDGET_E200`|10|89.37%|75.82%|checkpoint已写，进程存活|
|7|`KAD16H7_HARDENED_COMBINED_SAT_E200`|10|92.55%|81.31%|checkpoint已写，进程存活|

注意:`grep -i nan`在早期日志中命中过未激活或零样本评估字段，例如proxy尚未启用、`TEST overall_tx=nan% (0/0)`、`sat_cos=nan`、`aux=nan`和`gate_r95=nandeg`。这些不是startup fatal，但后续分析必须继续检查真实拒识代理、tail、overflow、energy margin和component radius指标，不能把早期闭集快照或proxy字段改善当作真实unknown拒识改善。

## 2026-07-02 20:01完成度只读巡检

用户询问“实验跑完了吗”。按N607 direct preflight后做只读监控，未启动、未停止、未修改远端任务。

总体状态:

- `phase1_kad_coregate_gpu8_20260702`即KAD8 slot A:8个候选均已到`E200/200`，日志fatal scan为OK，pmon中已无KAD8 compute进程；判定训练阶段已跑完，待做完整结果解析和Phase1双目标分析。
- `phase1_kad_hardening_secondlane_gpu8_20260702`即KAD16H slot B:8个候选仍在运行，当前约`E142-E156/200`，日志fatal scan为OK；尚未跑完。
- 当前pmon仅显示KAD16H 8个python compute进程，PID仍为slot B启动时记录的8个PID。
- SSH收尾:只读探针后本地`ssh_process_count=0`，到N607或bridge的`ESTABLISHED:22`连接数为0。

KAD8 slot A完成快照:

|candidate|last epoch|日志错误扫描|best joint快照|
|---|---:|---|---|
|`KAD8G0_COREGATE_ANCHOR_E200`|200|OK|`val_tx=98.29%`、`test_tx=88.09%`@E196|
|`KAD8G1_HOLDOUT_STRESS_E200`|200|OK|`val_tx=97.59%`、`test_tx=84.65%`@E050|
|`KAD8G2_BRIDGE_CVAR_E200`|200|OK|`val_tx=97.30%`、`test_tx=83.34%`@E040|
|`KAD8G3_SOURCE_OVERFLOW_E200`|200|OK|`val_tx=98.15%`、`test_tx=87.47%`@E182|
|`KAD8G4_LOW_DENSITY_GATE_E200`|200|OK|`val_tx=98.32%`、`test_tx=87.09%`@E184|
|`KAD8G5_ENERGY_MARGIN_Q05_E200`|200|OK|`val_tx=98.27%`、`test_tx=87.16%`@E120|
|`KAD8G6_RADIUS_INTER_BUDGET_E200`|200|OK|`val_tx=97.48%`、`test_tx=85.98%`@E050|
|`KAD8G7_COMBINED_SAT_REPAIR_E200`|200|OK|`val_tx=96.49%`、`test_tx=84.52%`@E040|

KAD16H slot B运行中快照:

|candidate|last epoch|日志错误扫描|best joint快照|
|---|---:|---|---|
|`KAD16H0_HARDENED_DEFAULT_ANCHOR_E200`|146|OK|`val_tx=98.14%`、`test_tx=85.60%`@E120|
|`KAD16H1_THREESIGMA_NEGCTRL_E200`|149|OK|`val_tx=98.04%`、`test_tx=86.43%`@E120|
|`KAD16H2_BRIDGE_COREQ75_E200`|155|OK|`val_tx=97.12%`、`test_tx=86.27%`@E040|
|`KAD16H3_SOURCE_COREQ75_QUAR_E200`|142|OK|`val_tx=98.26%`、`test_tx=86.76%`@E140|
|`KAD16H4_TAIL_SENTINEL_GUARD_E200`|144|OK|`val_tx=98.48%`、`test_tx=87.87%`@E130|
|`KAD16H5_PROXY_ONLY_BOUNDARY_E200`|145|OK|`val_tx=98.18%`、`test_tx=84.53%`@E110|
|`KAD16H6_P80_RADIUS_BUDGET_E200`|156|OK|`val_tx=97.03%`、`test_tx=83.08%`@E060|
|`KAD16H7_HARDENED_COMBINED_SAT_E200`|151|OK|`val_tx=95.96%`、`test_tx=84.62%`@E030|

解释边界:以上是完成度/健康巡检，不是Phase1最终排名。KAD8虽已结束，但还需要读取`metrics_epoch.csv`、prototype导出、tail/overflow/proxy/energy/component半径等同row指标后，才能判断泛化与拒识潜力是否同时改善。KAD16H还未结束，不能用当前中途best joint快照给出推进结论。

## 2026-07-02 KAD Phase1全量分析

用户要求按Phase1“可拒识的跨域泛化表征”目标全面分析KAD实验。已完成N607只读preflight、全量stdout扫描、16个`metrics_epoch.csv`全行解析、prototype JSON字段核验和同row指标汇总。分析报告与机器可复查表已落地:

- `E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\kad_phase1_full_analysis_20260702.md`
- `E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\analysis_20260702_2344\kad_full_log_metrics_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\analysis_20260702_2344\kad_generalization_table.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\analysis_20260702_2344\kad_rejection_table.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\analysis_20260702_2344\kad_pair_deltas.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\analysis_20260702_2344\kad_health_table.csv`

核心结论:KAD8G0仍是闭集DG最稳anchor；KAD16H只带来局部代理改善，没有出现“泛化提升且拒识风险下降”的主推进候选。所有候选`bridge_accept_rate=1.0`，proxy_vaccept仍偏高，source_episode_overflow普遍高。KAD16H1的satellite和proxy代理最好，但它是three-sigma负控且final guard失败，只能作为诊断性负例，不能作为Stage2推进证据。

声明边界:本分析仍为Phase1 source-only证据，不声明真实`unknown_FAR`、`FPR95`、Stage2 `old_acc`、`seen_new_acc`或`H_old_new`改善。
