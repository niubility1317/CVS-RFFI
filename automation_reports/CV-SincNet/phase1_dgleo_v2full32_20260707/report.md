# phase1_dgleo_v2full32_20260707

## 基本信息
- 时间:2026-07-07
- operator:Codex
- 目标:设计Phase1地面source-only域泛化训练矩阵,验证v2闭环机制,直接优化open-set几何指标,同时保护跨receiver/date和LEO星地压力泛化。
- 状态:已完成本地矩阵脚本、干跑测试和报告;未连接N607,未sync,未启动训练。
- 协议边界:Phase1 source-only,训练数据限定`ManySig.pkl`;不使用真实unknown类、不使用target receiver样本、不声明真实unknown_FAR/FPR95/Stage2成功。

## 本地文件
|文件|用途|
|---|---|
|`E:\type10-7\code\scripts\launch_phase1_dgleo_v2full32_20260707.sh`|32候选启动脚本,8张GPU每张4个实验,支持`--dry-run`和`--only=`|
|`E:\type10-7\code\tests\test_phase1_dgleo_v2full32_launcher.py`|矩阵协议、资源分配、机制消融和source-only边界测试|
|`E:\type10-7\code\snapshots\phase1_dgleo_v2full32_20260707_20260707_235418\`|非Git根目录变更快照|
|`E:\type10-7\automation_reports\CV-SincNet\phase1_dgleo_v2full32_20260707\report.md`|本报告|

## 设计原则
本矩阵不是把所有候选都做成弱化/稳定/激进/激进保护四档。GPU0保留全机制参照阶梯,用于判断“所有机制全量启动”在不同强度下的泛化/拒识冲突;GPU1-GPU7用于机制消融、单机制增强、交互压力和export gate验证。所有候选仍保留`endpoint_accept_v1`、`loss_gate_exported=false`、`tail_safety_state_machine=true`、`u_tri_state_required=true`和`feasibility_stage=audit`,避免把动态DM软门控当最终拒识边界。

## 矩阵总览
|GPU|候选|组别|类型|验证目标|
|---:|---|---|---|---|
|0|DGLEO_V2FULL32_FULL_WEAK|G0_FULL_LADDER|全机制弱化|低open-set梯度能否保护DG/LEO floor|
|0|DGLEO_V2FULL32_FULL_STABLE|G0_FULL_LADDER|全机制稳定|主参照,验证平衡解|
|0|DGLEO_V2FULL32_FULL_AGGR|G0_FULL_LADDER|全机制激进|验证加大几何loss是否真实压低p99/proxy_vaccept|
|0|DGLEO_V2FULL32_FULL_AGGR_SAFE|G0_FULL_LADDER|全机制激进保护|验证KD/sat保护能否缓解激进几何损失对strict UDU和sat floor的伤害|
|1|DGLEO_V2FULL32_DM_OFF|G1_DIRECT_LOSS_ABLATION|消融|关闭`direct_metric_accept`,仅保留endpoint评估,验证DM loss真实贡献|
|1|DGLEO_V2FULL32_DM_RELAXED|G1_DIRECT_LOSS_ABLATION|探索|放松DM目标,避免“罚很多但推不动”|
|1|DGLEO_V2FULL32_DM_PROXY_ALIGNED|G1_DIRECT_LOSS_ABLATION|探索|让DM目标与旧proxy风险代理更一致|
|1|DGLEO_V2FULL32_DM_HARD|G1_DIRECT_LOSS_ABLATION|压力|直接压p95/p99/source_overflow/proxy_vaccept上限|
|2|DGLEO_V2FULL32_SOURCE_OFF|G2_KNOWN_GEOMETRY|消融|关闭source episode收紧,验证source_overflow是否恶化|
|2|DGLEO_V2FULL32_SOURCE_FOCUS|G2_KNOWN_GEOMETRY|隔离|重点收紧known core,弱化proxy/DM干扰|
|2|DGLEO_V2FULL32_SOURCE_STRICT|G2_KNOWN_GEOMETRY|压力|强压source_episode_overflow和tail expansion|
|2|DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE|G2_KNOWN_GEOMETRY|保护|receiver-aware局部component方向,兼顾DG floor|
|3|DGLEO_V2FULL32_PROXY_OFF|G3_PROXY_BRIDGE|消融|关闭proxy unknown,验证旧proxy_vaccept/bridge风险来源|
|3|DGLEO_V2FULL32_PROXY_VACCEPT|G3_PROXY_BRIDGE|探索|集中压proxy_vaccept和proxy accept CVaR|
|3|DGLEO_V2FULL32_BRIDGE_LOW_DENSITY|G3_PROXY_BRIDGE|压力|集中压bridge_accept和low_density_accept|
|3|DGLEO_V2FULL32_SHELL_TAIL_PROXY|G3_PROXY_BRIDGE|压力|集中压shell/tail/overflow accept|
|4|DGLEO_V2FULL32_U_OFF|G4_U_TRISTATE|消融|关闭无标签loss,但保留三态审计,验证U_s是否真实贡献|
|4|DGLEO_V2FULL32_U_DOMAIN_SAT|G4_U_TRISTATE|消融|只用U_s域/星地一致性,不直接做open-set U loss|
|4|DGLEO_V2FULL32_U_DIRECT_QUAR|G4_U_TRISTATE|隔离|重点验证U_s direct/quarantine是否能参与几何收紧|
|4|DGLEO_V2FULL32_U_TRISTATE_FULL|G4_U_TRISTATE|探索|三态U_s全开,观察trusted_core/ambiguous_tail/outside_reject分布|
|5|DGLEO_V2FULL32_SAT_WEAK|G5_SAT_DG_STRESS|弱化|星地增强弱约束,观测sat floor和open-set几何基线|
|5|DGLEO_V2FULL32_SAT_STRONG|G5_SAT_DG_STRESS|探索|增强sat consistency/KD,验证LEO floor提升|
|5|DGLEO_V2FULL32_SAT_DOMAIN_ADV|G5_SAT_DG_STRESS|压力|域分类/ADV与星地一致性同时增强|
|5|DGLEO_V2FULL32_SAT_OPEN_PAIR|G5_SAT_DG_STRESS|压力|强化sat pair open-set约束,观察星地视图下p99/proxy_vaccept|
|6|DGLEO_V2FULL32_BUDGET_CLOSED|G6_GRADIENT_BUDGET|消融|闭集/KD/sat占优,验证几何loss被压制时open-set指标是否不动|
|6|DGLEO_V2FULL32_BUDGET_BALANCED|G6_GRADIENT_BUDGET|稳定|平衡预算参照|
|6|DGLEO_V2FULL32_BUDGET_OS_HIGH|G6_GRADIENT_BUDGET|压力|提高open-set预算,验证直接指标响应|
|6|DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE|G6_GRADIENT_BUDGET|保护|高open-set预算叠加KD/sat保护|
|7|DGLEO_V2FULL32_EXPORT_LOCAL|G7_EXPORT_GATE|探索|local component export严格化|
|7|DGLEO_V2FULL32_EXPORT_TAIL_GATE|G7_EXPORT_GATE|探索|tail delta gate稳定性|
|7|DGLEO_V2FULL32_EXPORT_FEASIBILITY|G7_EXPORT_GATE|压力|严格目标下的feasibility/audit行为|
|7|DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE|G7_EXPORT_GATE|保护|promotion-safe综合导出候选|

## 共同机制
- 数据:`--wisig_pkl .../ManySig.pkl`,`--split_mode tx_rx_day_1_7_2`,`labeled_ratio=0.10`,`unlabeled_ratio=0.70`,`source_val_ratio=0.20`。
- 星地视图:所有候选使用`--use_concat_sat_channel_aug`和`--no_concat_sat_ce_only`;不允许退化为只用TX CE。星地损失同时包含sat CE、sat consistency、teacher sat KL、U_s sat consistency和direct metric sat pair。
- 最终拒识边界:保留`endpoint_accept_v1`,并固定`--loss_gate_exported false`;DM软门控只作训练loss和审计代理,不作为最终拒识边界。
- tail safety:`p99_delta>2.0`阻断final export,`p99_delta>3.5`阻断promotion;同时启用CVaR delta gate。
- U_s三态:保留`trusted_core/ambiguous_tail/outside_reject`审计要求;即使`U_OFF`也必须产出三态审计并fail-closed。
- known几何:保留`source_episode_density_gate=true`和`source_episode_min_local_components=4`;source消融只关闭loss权重,不关闭审计门控。
- prototype导出:`phase1_source_zid_prototypes.pt`,local component accept,禁止global ball和tail auto accept。

## 主指标与成功标准
|类别|指标|成功标准|失败判据|
|---|---|---|---|
|DG泛化|overall_tx、strict_udu、receiver_floor、per-receiver最弱点|不低于OSFIX稳定候选,且best-final gap收窄|overall提升但strict UDU/floor下降|
|星地压力|sat mean/floor、sat strict floor、sat pair z_id距离|sat floor不低于EPOC/OSFIX参照,星地视图下open-set指标同步改善|只改善clean,星地p99/proxy_vaccept恶化|
|known收紧|zid_p50/p95/p99、zid_tail_cvar、tail_frac、r3sigma|p95/p99/CVaR同步下降,且final不扩tail|p95下降但p99/CVaR/overflow仍高|
|拒识代理|proxy_vaccept、proxy_vac_rate、proxy_auc、bridge_accept_rate、low_density_accept_rate|旧proxy_unknown指标同步下降,不是只让dm_*变好|dm_*下降但旧proxy_vaccept仍高|
|source episode|source_episode_overflow/source_overflow|SOURCE_FOCUS/STRICT相对SOURCE_OFF明显下降|source_overflow仍约0.97或比参照更差|
|边界比例|radius_to_inter_ratio、tail/overflow accept|ratio下降且tail/overflow accept下降|min_inter提升但proxy/bridge仍接收|
|U_s利用|U_s active epoch、trusted_core/ambiguous_tail/outside_reject、U direct/quarantine loss非零|U_TRISTATE_FULL优于U_OFF和U_DOMAIN_SAT_ONLY|U_s selected有数但direct/quarantine为0|
|导出安全|endpoint parity、reason code、artifact字段、p99_delta gate|best checkpoint和final export均有完整字段|final export绕过tail gate或prototype缺字段|

## 本地验证
|命令|结果|
|---|---|
|`conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_dgleo_v2full32_launcher.py -q`|4 passed;仅有`.pytest_cache`权限warning|
|`bash -n code/scripts/launch_phase1_dgleo_v2full32_20260707.sh`|通过|
|`bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh --dry-run --only=DGLEO_V2FULL32_FULL_STABLE`|确认全机制稳定候选包含DM/source/proxy/U_s/sat/domain/ADV和`phase1_source_zid_prototypes.pt`|
|`bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh --dry-run --only=DGLEO_V2FULL32_DM_OFF,DGLEO_V2FULL32_SOURCE_OFF,DGLEO_V2FULL32_PROXY_OFF,DGLEO_V2FULL32_U_OFF`|确认对应消融实际置零loss权重,不是只改候选名|

## N607启动边界
尚未连接N607。若后续启动,必须先执行`tools\n607_ssh_preflight.ps1`,记录GPU占用和server时间,再sync本地脚本/测试/报告。用户本轮指定每张卡4个实验,脚本默认`MAX_ACTIVE_PER_GPU=4`;启动前仍需确认当前GPU占用,避免覆盖已有日志或超过用户允许的并发边界。
