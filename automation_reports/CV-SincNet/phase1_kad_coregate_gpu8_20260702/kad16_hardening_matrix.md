# KAD16 Phase1 known accept域治理实验矩阵

生成时间:2026-07-02  
operator:Codex Phase1地面训练实验分析agent  
状态:本地设计完成；新增8行未同步N607、未启动  
关联运行:
- 已启动/正在跑批次:`phase1_kad_coregate_gpu8_20260702`，8个KAD8候选，每GPU一条；该批次按启动记录使用的是accept gate hardening之前的远程代码，结果只能作为旧code证据。
- 待确认第二卡槽批次:`phase1_kad_hardening_secondlane_gpu8_20260702`，8个KAD16H候选，每GPU追加一条；启动前必须同步本地硬化代码和新launcher。

## 协议边界

这16行仍是Phase1 source-only地面训练验证。可以判断闭集DG能力、星地压力鲁棒性、known特征几何、proxy/virtual unknown风险和prototype导出质量；不能声明真实`unknown_FAR`、`FPR95`、Stage2 old/new/unknown成功、`seen_new_acc`或`H_old_new`已经改善。

KAD8和KAD16H不能合并排名。KAD8用于观察原KAD机制在真实训练中的健康度和弱点；KAD16H用于验证修改后的机制是否真正治理known accept域。若KAD8结果好但KAD16H未启动或未完成，只能说明旧机制的Phase1闭集/代理表现，不能证明accept gate hardening有效。

`phase2_export_prototypes`和`phase2_fuse_prototypes`只表示源域prototype导出和后续独立Stage2-A评估准备，不表示Stage2已经运行、Stage2成功或部署证据成立。

## 每张卡两个实验布局

|GPU|slot A:正在跑/已启动|slot A机制|slot B:新增待启动|slot B机制|启动边界|
|---|---|---|---|---|---|
|0|`KAD8G0_COREGATE_ANCHOR_E200`|core-gated anchor，p80 component gate，core-safe episode|`KAD16H0_HARDENED_DEFAULT_ANCHOR_E200`|硬化默认锚点，依赖默认`core_quantile`和`min_three_sigma_core`|第二槽启动前必须远程guard通过|
|1|`KAD8G1_HOLDOUT_STRESS_E200`|holdout stress，检验更多leave-domain proxy是否扩大overflow|`KAD16H1_THREESIGMA_NEGCTRL_E200`|显式`three_sigma`负控，预期暴露宽半径/tail风险|仅诊断，不得promotion|
|2|`KAD8G2_BRIDGE_CVAR_E200`|bridge CVaR治理|`KAD16H2_BRIDGE_COREQ75_E200`|p75 component gate+更强bridge CVaR|看bridge下降是否伤receiver floor|
|3|`KAD8G3_SOURCE_OVERFLOW_E200`|source overflow和tail quarantine|`KAD16H3_SOURCE_COREQ75_QUAR_E200`|source core q75+source_safe/tail_quarantine加权|看overflow是否下降且weak receiver不崩|
|4|`KAD8G4_LOW_DENSITY_GATE_E200`|低密度accept和shell/outward治理|`KAD16H4_TAIL_SENTINEL_GUARD_E200`|故意请求`tail_auto_accept=true`，验证导出强制`effective=false`|若导出tail可自动接收即失败|
|5|`KAD8G5_ENERGY_MARGIN_Q05_E200`|energy q05/q10和vaccept CVaR|`KAD16H5_PROXY_ONLY_BOUNDARY_E200`|强化proxy_vaccept/energy压力，同时验证proxy-only声明边界|proxy改善不得等价真实拒识改善|
|6|`KAD8G6_RADIUS_INTER_BUDGET_E200`|component radius和radius/inter预算|`KAD16H6_P80_RADIUS_BUDGET_E200`|更严p75训练gate、p80导出、半径预算|若strict UDU/receiver floor明显降则不可主推|
|7|`KAD8G7_COMBINED_SAT_REPAIR_E200`|组合治理+satellite stress|`KAD16H7_HARDENED_COMBINED_SAT_E200`|硬化组合+satellite floor修复|只有泛化和拒识代理同row改善才可推进Stage2-A评估|

并发语义:每张GPU最多两个训练进程。当前KAD8占slot A；KAD16H launcher默认`STAGE2_MAX_ACTIVE_PER_GPU=2`，只允许作为第二槽追加，不允许超过两条/卡。新增launcher在非dry-run时会创建`${LOG_ROOT}/.launcher.lock`，用于拒绝重复提交同一second-lane批次；同时拒绝把KAD16H误指到`phase1_kad_coregate_gpu8_20260702`的run/log目录。

## 16行候选判据

|candidate|泛化判断重点|known accept域判断重点|Phase1同row可推进标准（非Stage2/部署成功）|失败判据|后续Stage2-A真实unknown评估资格边界（当前Phase1不产生unknown_FAR/FPR95）|
|---|---|---|---|---|---|
|`KAD8G0_COREGATE_ANCHOR_E200`|overall、strict UDU、receiver floor、satellite mean/floor相对ADV3B02是否稳定|p95/p99、overflow、component radius、proxy_vaccept是否同步下降|DG不低于ADV3B02，p99/overflow不扩张|闭集好但p99、overflow或proxy_vaccept变差|若完成后双目标达标，仅可列入后续独立Stage2-A评估候选，当前结果不得替代unknown_FAR/FPR95/AUROC|
|`KAD8G1_HOLDOUT_STRESS_E200`|holdout加大后strict和receiver floor是否保持|source overflow是否重现或恶化|extra holdout不牺牲receiver floor且overflow不升|overflow升高或weak receiver退化|主要是诊断负例|
|`KAD8G2_BRIDGE_CVAR_E200`|bridge惩罚下strict UDU是否不掉|`bridge_accept_rate`、low-density accepted bridge、min_inter/p99组合|bridge下降且DG稳定|min_inter高但bridge_accept仍接近1，或receiver floor下降|只可作为bridge治理证据|
|`KAD8G3_SOURCE_OVERFLOW_E200`|source-safe下跨receiver困难样本是否仍可分类|`source_episode_overflow`、tail_frac、zid tail CVaR|overflow下降，p99不扩张，receiver floor稳定|overflow未降或tail继续撑大包络|若同row双目标满足，仅可列入后续独立Stage2-A评估候选|
|`KAD8G4_LOW_DENSITY_GATE_E200`|低密度gate是否伤害old closed-set|`low_density_accept_rate`、shell/outward accept、tail accept|low-density accept下降且overall/strict不回落|old tail被过度拒收或weak receiver更差|诊断低密度gate是否必要|
|`KAD8G5_ENERGY_MARGIN_Q05_E200`|energy下分位压力是否影响闭集|`energy_margin_q05/q10`、`vaccept_surrogate_CVaR`、proxy_vaccept|energy margin改善且source overflow不变差|只修proxy，source tail和p99不动|proxy-only，不等价真实unknown拒识；当前结果不得替代unknown_FAR/FPR95/AUROC|
|`KAD8G6_RADIUS_INTER_BUDGET_E200`|半径预算是否牺牲strict UDU|`component_radius_p95/max`、`radius_to_inter_ratio`、p99|半径/ratio下降且receiver floor稳定|半径收紧但DG显著掉|可作为导出半径预算候选|
|`KAD8G7_COMBINED_SAT_REPAIR_E200`|satellite floor和最弱receiver是否修复|bridge、low-density、energy、radius是否同向改善|同row泛化和accept风险都改善|平均提升但weak receiver未修复，或best-final gap大|仅满足双目标后，列入后续独立Stage2-A评估候选|
|`KAD16H0_HARDENED_DEFAULT_ANCHOR_E200`|硬化默认是否保持KAD8G0级别DG|默认`core_quantile`和`min_three_sigma_core`是否收紧p99/overflow|不显式传radius mode仍使用硬化默认，DG稳定，tail风险下降|默认退回three-sigma或p99/overflow升|硬化默认锚点，完成后优先比较KAD8G0|
|`KAD16H1_THREESIGMA_NEGCTRL_E200`|显式three-sigma是否表面保护闭集|p95/p99、source_overflow、component p95/max是否被长尾抬高|作为负控，应暴露更宽gate或tail风险|若被错误当成promotion，或指标无法区分宽gate|不可promotion，只用于证明three-sigma风险|
|`KAD16H2_BRIDGE_COREQ75_E200`|p75 strict bridge下receiver floor是否保住|`bridge_accept_rate`、same-class bridge、low-density accept|bridge明显下降，p99不扩，weak receiver稳定|bridge仍高或泛化崩|可作为bridge治理主诊断|
|`KAD16H3_SOURCE_COREQ75_QUAR_E200`|source core q75是否伤弱receiver|`source_episode_overflow`、tail quarantine、zid_tail_cvar|overflow/tail下降且strict/receiver floor稳定|source tail继续合法化，或weak receiver下降|若成功，是source episode改造候选；仍需后续独立Stage2-A评估|
|`KAD16H4_TAIL_SENTINEL_GUARD_E200`|tail sentinel守卫不应影响闭集|导出字段`tail_auto_accept_requested=true`且`tail_auto_accept_effective=false`，tail sentinel `accept_enabled=false`|即使命令请求自动接收，导出仍禁止自动接收|任何tail sentinel自动接收或字段缺失|导出守卫验证，不单独作为模型候选|
|`KAD16H5_PROXY_ONLY_BOUNDARY_E200`|强化proxy压力是否扰动DG|proxy_vaccept、proxy_auc、energy q05/q10、`proxy_reject_claim_allowed=0`|proxy指标改善同时日志明确proxy-only边界|proxy变好但source overflow/p99变差，或报告误称真实拒识改善|只可列入后续独立真实unknown评估，不可替代该评估|
|`KAD16H6_P80_RADIUS_BUDGET_E200`|更严半径预算是否仍保strict UDU|`component_radius_p95/max`、`radius_to_inter_ratio`、导出p80半径|radius/ratio下降且satellite/receiver floor不降|半径过紧导致闭集回落|可作为导出gate候选|
|`KAD16H7_HARDENED_COMBINED_SAT_E200`|组合硬化是否修复satellite floor和最弱receiver|bridge、source_overflow、low_density、energy、radius综合|泛化四维和accept风险同row改善|只改善平均值，弱receiver未修复，或tail/overflow反弹|唯一可能作为Phase1主推进候选，但仍需后续独立Stage2-A评估|

## 指标主表要求

每个候选完成后必须同row读取以下字段，不能用不同候选的单项最优拼接结论:

- 泛化:overall_tx、strict UDU、receiver floor、satellite mean/floor、最弱receiver、best-final gap。
- 训练健康:完成epoch、metrics_epoch、stdout fatal/NaN、scheduler、prototype导出、best/final一致性。
- accept域:p95、p99、r3sigma、tail_frac、min_inter、source_episode_overflow、ow_vac_rate、proxy_auc、proxy_vac_rate、proxy_vaccept。
- 新硬化指标:`bridge_accept_rate`、`source_overflow`、`low_density_accept_rate`、`zid_compact_pos_angle_p50/p95/p99`、`zid_tail_cvar`、`component_radius_p95/max`、`radius_to_inter_ratio`、`vaccept_surrogate_CVaR`、`energy_margin_q05/q10`。
- 导出包:component数量、component radius分布、`tail_auto_accept_requested`、`tail_auto_accept_effective`、tail sentinel `accept_enabled`、radius key、fusion/local component字段。

全局矩阵成功标准:

- 16/16行都有同row完整指标、完成epoch/metrics_epoch/stdout/prototype导出证据，且没有fatal、NaN或关键机制字段缺失。
- 所有候选保持`promotion_allowed=false`，所有proxy相关结论保留`proxy_reject_claim_allowed=0`或等价日志证据。
- KAD16H1 three-sigma负控只能作为机制反证或风险对照；若它闭集强但tail/overflow/半径变宽，不得作为主推进候选。
- KAD16H4必须在导出`pt/json`中核对`tail_auto_accept_requested=true`、`tail_auto_accept_effective=false`和tail sentinel `accept_enabled=false`，不能只看训练日志。
- proxy_vaccept、proxy_auc、virtual unknown接收率改善只能写作Phase1 source proxy诊断，不能写成真实unknown拒识改善或部署成功。

## 启动前硬门槛

新增KAD16H启动前必须满足:

1. 本地硬化补丁已同步到N607，至少包含`cvsrffi/losses.py`、`cvsrffi/phase2_prototypes.py`、`SSDG/train_ssdg.py`和`code/scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`。
2. 远程launcher的非dry-run guard通过:`component_radius_mode: str = "core_quantile"`、`radius_mode: str = "min_three_sigma_core"`、`tail_auto_accept_effective`、`proxy_reject_claim_allowed`均存在；同时补远程SHA256、`py_compile`、`bash -n`和远程dry-run8候选计数。
3. N607 preflight、GPU占用、每GPU训练进程数、当前KAD8真实compute进程证据均已记录，需包含PID、cwd、cmdline、`CUDA_VISIBLE_DEVICES`和log路径，并排除launcher/bash父进程。
4. 远程目标`runs/phase1_kad_hardening_secondlane_gpu8_20260702`和`logs/phase1_kad_hardening_secondlane_gpu8_20260702`不存在；若存在任何candidate子目录、`.out`、`.pid`、`launcher.out`或`.launcher.lock`，停止而不是复用。
5. 若KAD8仍在跑，只能在每GPU当前训练进程数小于2时启动第二槽；若某卡已有2条训练则等待。
6. 启动前确认远端没有同run launcher/PID/日志进行中；新增launcher的`.launcher.lock`必须不存在。
7. 新增8行启动后必须更新本报告，记录确切scp映射、远程hash、远程命令、PID、GPU、log路径、SSH/SCP断连清理和启动后4-5分钟startup health检查结果。

## 本地验证

|命令|结果|
|---|---|
|`bash -n scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh`|通过|
|`bash scripts/launch_phase1_kad_hardening_secondlane_gpu8_20260702.sh --dry-run`|通过，8个KAD16H候选均可展开|

dry-run归档:`E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\dry_run_hardening_secondlane_20260702.txt`

## 当前决策

当前16行矩阵的贡献是:把KAD8旧code运行证据和KAD16H硬化机制验证拆开，围绕known accept域治理构造了可证伪对照。当前不能声明的是:真实unknown拒识、Stage2成功、proxy_vaccept改善等价真实拒识、fusion成功。最主要风险是:KAD8仍可能体现旧three-sigma/tail宽边界，KAD16H若未启动则没有硬化实验证据；新增第二槽启动时还存在并发和远程代码版本一致性风险。
