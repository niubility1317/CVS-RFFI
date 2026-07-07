# Phase1 DG-LEO DirectMetric16与UOPT24联合分析

## 协议边界

本报告只分析Phase1 source-only地面训练证据。训练数据边界是ManySig源接收机L_s/U_s和源域派生LEO增强视图；不得声称真实unknown_FAR、FPR95、Stage2 old_acc、seen_new_acc或H_old_new已经改善。proxy/virtual unknown只作为Phase1代理拒识风险。

## 证据完整性

- `phase1_dgleo_directmetric16_20260706`：16/16候选完成200epoch，16/16有`metrics_epoch.csv`、stdout、`phase2_zid_prototypes.json`和`.pt`。
- `phase1_dgleo_uopt24_20260707`：24/24候选完成200epoch，24/24有`metrics_epoch.csv`、stdout、`phase2_zid_prototypes.json`和`.pt`。
- 两组fatal=0，无Traceback/OOM/argparse失败；final metrics中test nonfinite count为0。训练日志存在固定非有限统计字段，DirectMetric16为train=5/val=1，UOPT24为train=16/val=1，且`train_skipped_nonfinite_grad=1`，应作为健康风险记录，但不是崩溃。
- best epoch集中在10-40，说明后期pseudo/几何压缩阶段普遍不能继续提升strict UDU。

## 分组主表

|run_id|n|epochs|strict_mean|strict_max|receiver_floor_mean|receiver_floor_max|sat_floor_mean|sat_floor_max|p99_mean|p99_min|proxy_vaccept_mean|source_overflow_mean|best_final_gap_mean|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|phase1_dgleo_directmetric16_20260706|16|200/200 all|85.294|86.435|73.479|78.067|75.254|75.957|84.772|83.982|0.155|0.743|0.505|
|phase1_dgleo_uopt24_20260707|24|200/200 all|84.962|86.127|72.919|78.625|75.372|75.786|83.818|82.262|0.146|0.769|0.891|

## 联合候选Top12

joint_score只用于排序辅助，不替代四象限判定。

|run_id|candidate|best_epoch|final_test_tx|final_strict_udu|final_receiver_floor|final_sat_floor|best_minus_final_strict|final_dm_p95|final_dm_p99|final_dm_source_overflow|final_dm_proxy_vaccept|final_dm_radius_inter|final_source_episode_overflow|final_proxy_vaccept|joint_score|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|phase1_dgleo_uopt24_20260707|DGLEO_UOPT_P0_CORE_C|40|89.928|86.127|78.625|75.255|0.000|72.195|82.377|0.769|0.148|1.131|0.974|0.630|84.233|
|phase1_dgleo_directmetric16_20260706|DGLEO_DM_P0D_RADIUS_A|30|89.982|86.435|78.067|75.424|0.000|74.655|84.819|0.739|0.152|1.134|0.967|0.626|83.366|
|phase1_dgleo_uopt24_20260707|DGLEO_UOPT_P0_GATE_B|10|89.620|85.222|73.617|75.105|1.373|71.851|82.262|0.765|0.146|1.103|0.972|0.638|82.199|
|phase1_dgleo_uopt24_20260707|DGLEO_UOPT_P0_CORE_A|30|89.566|85.402|73.658|75.641|0.630|72.616|84.135|0.771|0.142|1.090|0.971|0.639|82.181|
|phase1_dgleo_uopt24_20260707|DGLEO_UOPT_P1_QUOTA_C|30|89.242|84.355|71.233|75.491|0.777|73.170|83.468|0.762|0.144|1.086|0.972|0.639|82.027|
|phase1_dgleo_uopt24_20260707|DGLEO_UOPT_P0_SAT_A|30|89.550|85.638|75.075|75.028|0.785|72.530|83.645|0.765|0.145|1.155|0.972|0.639|81.880|
|phase1_dgleo_directmetric16_20260706|DGLEO_DM_P1B_FLOOR_A|20|89.648|85.448|73.192|75.497|0.157|73.920|84.400|0.743|0.157|1.115|0.972|0.621|81.612|
|phase1_dgleo_uopt24_20260707|DGLEO_UOPT_P0_SAT_C|10|89.701|85.747|73.758|75.729|0.120|72.547|83.646|0.769|0.145|1.149|0.970|0.645|81.545|
|phase1_dgleo_uopt24_20260707|DGLEO_UOPT_P1_STRONG_C|10|88.904|84.297|73.083|74.672|2.157|73.168|83.179|0.769|0.148|1.125|0.972|0.654|81.483|
|phase1_dgleo_directmetric16_20260706|DGLEO_DM_P0E_STRONG_B|20|90.075|86.013|77.775|75.146|0.000|74.673|84.621|0.746|0.153|1.171|0.973|0.619|81.450|
|phase1_dgleo_directmetric16_20260706|DGLEO_DM_P0A_CORETAIL_B|20|89.882|85.840|73.800|75.543|0.000|76.705|85.439|0.744|0.155|1.106|0.971|0.641|81.379|
|phase1_dgleo_uopt24_20260707|DGLEO_UOPT_P1_ADV_C|10|89.297|84.837|73.142|75.607|1.105|73.624|83.844|0.771|0.146|1.112|0.973|0.649|81.333|

## UOPT24无标签利用机制族

|group|u_active_epochs|w_u_domain_mean_pseudo|w_u_adv_mean_pseudo|w_u_sat_mean_pseudo|w_u_dm_mean_active|final_strict_udu|final_receiver_floor|final_sat_floor|final_dm_proxy_vaccept|final_dm_source_overflow|final_dm_p95|final_dm_p99|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|P0_BAL|15.667|0.127|0.191|0.276|0.006|85.101|71.914|75.634|0.146|0.768|73.985|84.276|
|P0_CORE|15.333|0.130|0.199|0.198|0.004|85.365|75.247|75.334|0.145|0.771|72.941|83.248|
|P0_GATE|0.000|0.129|0.208|0.249||85.058|73.675|75.260|0.148|0.768|73.363|83.719|
|P0_SAT|32.000|0.115|0.147|0.312|0.005|85.547|73.400|75.350|0.144|0.767|72.654|83.755|
|P1_ADV|14.333|0.129|0.312|0.258|0.005|85.026|73.294|75.496|0.145|0.770|74.409|84.400|
|P1_LATE|32.000|0.143|0.234|0.299|0.004|84.251|70.922|75.142|0.149|0.769|74.449|83.958|
|P1_QUOTA|33.667|0.129|0.190|0.268|0.005|84.347|71.147|75.477|0.145|0.768|73.253|83.598|
|P1_STRONG|31.667|0.142|0.243|0.304|0.009|85.006|73.753|75.278|0.145|0.770|73.815|83.593|

结论：单候选最强是`DGLEO_UOPT_P0_CORE_C`。机制上，最有效的不是强U_s直接几何或强ADV，而是温和U_s domain/ADV、较弱U_s satellite consistency、低权重U_s direct metric的组合。P0_SAT族平均strict UDU最高，说明U_s星地一致性有价值；但P0_SAT未同步修复receiver floor。强ADV、late和quota族没有形成稳定收益，部分候选后期回落和receiver floor下降明显。

## DirectMetric16机制判断

DirectMetric16最重要的正例是`DGLEO_DM_P0D_RADIUS_A`：final strict UDU=86.435，receiver floor=78.067，sat floor=75.424，best-final gap=0。它说明直接指标loss可以在不牺牲闭集DG的情况下得到可用Phase1表征。

但DirectMetric16没有解决open-set代理风险：组均p99=84.772，source_overflow=0.743，source_episode_overflow=0.971，direct metric proxy_vaccept=0.155，proxy branch vaccept=0.635，radius_to_inter=1.128。也就是说，p95/p99有一定约束，proxy/direct metric accept有所下降，但known尾部和source episode overflow仍然过大，proxy branch仍然大量接收proxy。

## 主要失败模式

- 闭集强但open-set危险：`DGLEO_DM_P0E_STRONG_B`、`DGLEO_UOPT_P1_STRONG_A`闭集较强，但source_overflow和radius/inter仍高。
- p95下降但p99/overflow仍高：UOPT24均值p95从DirectMetric16的75.088降到73.609，p99从84.772降到83.818，但source_overflow从0.743升到0.769。
- min_inter高但proxy接受仍高：prototype min_inter约88.7deg，local component字段存在，但proxy branch vaccept仍约0.63-0.65。
- source_episode_overflow升高：DirectMetric16均值0.971，UOPT24均值0.972，基本未改善。
- best checkpoint强但final退化：UOPT24 strict平均best-final gap=0.891pp，高于DirectMetric16的0.505pp；10/24个UOPT候选回落超过1pp。
- satellite平均不等于弱receiver修复：P1_ADV_B/P0_BAL_C/P0_SAT_C有高sat floor，但receiver floor并非最强。
- U_s直接几何loss不稳定：UOPT24中只有11/24候选出现U_s direct metric active epoch，且日志没有单独记录U_s p50/tail_cvar，说明无标签直接几何利用仍是半生效机制。

## 候选决策表

|candidate|泛化结论|拒识潜力|主要风险|可否Stage2真实unknown评估|下一步动作|
|---|---|---|---|---|---|
|DGLEO_DM_P0A_CORETAIL_A|泛化可用|无明显改善；风险:p99高、dm_proxy_vaccept偏高、source_episode_overflow高、radius/inter>1|source_overflow=0.745;p99=85.003;proxy=0.162;gap=0.000|保留为机制诊断|保留同口径对照|
|DGLEO_DM_P0A_CORETAIL_B|泛化可用|无明显改善；风险:p99高、dm_proxy_vaccept偏高、source_episode_overflow高、radius/inter>1|source_overflow=0.744;p99=85.439;proxy=0.155;gap=0.000|保留为机制诊断|保留同口径对照|
|DGLEO_DM_P0B_BRIDGE_A|泛化弱/不稳，后期回落1.11pp|无明显改善；风险:dm_proxy_vaccept偏高、source_episode_overflow高、radius/inter>1|source_overflow=0.748;p99=84.737;proxy=0.156;gap=1.105|保留为机制诊断|缩短pseudo阶段或冻结best评估|
|DGLEO_DM_P0B_BRIDGE_B|泛化弱/不稳，后期回落1.91pp，receiver floor低，sat floor低|无明显改善；风险:source_episode_overflow高、radius/inter>1|source_overflow=0.739;p99=84.202;proxy=0.151;gap=1.905|不建议Stage2；先修复弱receiver/后期回落|缩短pseudo阶段或冻结best评估|
|DGLEO_DM_P0C_BAL_A|泛化可用|无明显改善；风险:dm_proxy_vaccept偏高、source_episode_overflow高、radius/inter>1|source_overflow=0.740;p99=84.326;proxy=0.164;gap=0.610|保留为机制诊断|保留同口径对照|
|DGLEO_DM_P0C_BAL_B|泛化可用|无明显改善；风险:p99高、source_episode_overflow高、radius/inter>1|source_overflow=0.746;p99=85.118;proxy=0.153;gap=0.448|保留为机制诊断|保留同口径对照|
|DGLEO_DM_P0D_RADIUS_A|泛化强|无明显改善；风险:radius/inter>1|source_overflow=0.739;p99=84.819;proxy=0.152;gap=0.000|可进入Stage2真实unknown评估，但只作候选验证|DirectMetric16基准推进：Stage2对照并检验overflow|
|DGLEO_DM_P0D_RADIUS_B|泛化弱/不稳，后期回落1.18pp|无明显改善；风险:dm_proxy_vaccept偏高、source_episode_overflow高、radius/inter>1|source_overflow=0.738;p99=84.928;proxy=0.160;gap=1.180|保留为机制诊断|缩短pseudo阶段或冻结best评估|
|DGLEO_DM_P0E_STRONG_A|泛化弱/不稳，sat floor低|无明显改善；风险:p99高、source_episode_overflow高、radius/inter>1|source_overflow=0.749;p99=85.391;proxy=0.150;gap=0.252|保留为机制诊断|保留同口径对照|
|DGLEO_DM_P0E_STRONG_B|泛化强|无明显改善；风险:source_episode_overflow高、radius/inter>1|source_overflow=0.746;p99=84.621;proxy=0.153;gap=0.000|保留为机制诊断|保留同口径对照|
|DGLEO_DM_P1A_LATE_A|泛化弱/不稳，后期回落1.09pp|无明显改善；风险:dm_proxy_vaccept偏高、source_episode_overflow高、radius/inter>1|source_overflow=0.747;p99=84.950;proxy=0.155;gap=1.088|保留为机制诊断|缩短pseudo阶段或冻结best评估|
|DGLEO_DM_P1A_LATE_B|泛化弱/不稳，receiver floor低|无明显改善；风险:source_episode_overflow高、radius/inter>1|source_overflow=0.739;p99=84.421;proxy=0.152;gap=0.000|不建议Stage2；先修复弱receiver/后期回落|弱receiver定向satellite stress修复|
|DGLEO_DM_P1B_FLOOR_A|泛化可用|无明显改善；风险:dm_proxy_vaccept偏高、source_episode_overflow高、radius/inter>1|source_overflow=0.743;p99=84.400;proxy=0.157;gap=0.157|保留为机制诊断|保留同口径对照|
|DGLEO_DM_P1B_FLOOR_B|泛化弱/不稳，receiver floor低|无明显改善；风险:source_episode_overflow高、radius/inter>1|source_overflow=0.743;p99=84.454;proxy=0.153;gap=0.552|不建议Stage2；先修复弱receiver/后期回落|弱receiver定向satellite stress修复|
|DGLEO_DM_P1C_SATPAIR_A|泛化弱/不稳，sat floor低|无明显改善；风险:source_episode_overflow高、radius/inter>1|source_overflow=0.744;p99=83.982;proxy=0.153;gap=0.838|保留为机制诊断|保留同口径对照|
|DGLEO_DM_P1C_SATPAIR_B|泛化可用|无明显改善；风险:p99高、dm_proxy_vaccept偏高、source_episode_overflow高、radius/inter>1|source_overflow=0.743;p99=85.555;proxy=0.158;gap=-0.053|保留为机制诊断|保留同口径对照|
|DGLEO_UOPT_P0_BAL_A|泛化可用|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.768;p99=84.215;proxy=0.147;gap=0.543|保留为机制诊断|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_BAL_B|泛化可用|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.768;p99=84.145;proxy=0.143;gap=0.700|保留为机制诊断|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_BAL_C|泛化弱/不稳，receiver floor低|无明显改善；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.769;p99=84.467;proxy=0.149;gap=0.212|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_CORE_A|泛化可用|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.771;p99=84.135;proxy=0.142;gap=0.630|保留为机制诊断|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_CORE_B|泛化弱/不稳，后期回落1.32pp|p99收紧；dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.773;p99=83.231;proxy=0.145;gap=1.317|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_CORE_C|泛化强|p99收紧；dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.769;p99=82.377;proxy=0.148;gap=0.000|可进入Stage2真实unknown评估，但只作候选验证|主推进：补U_s日志后做真实unknown query评估|
|DGLEO_UOPT_P0_GATE_A|泛化弱/不稳|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.769;p99=84.279;proxy=0.144;gap=0.955|保留为机制诊断|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_GATE_B|泛化可用，后期回落1.37pp|p99收紧；dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.765;p99=82.262;proxy=0.146;gap=1.373|可做Stage2诊断评估|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_GATE_C|泛化可用|无明显改善；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.771;p99=84.617;proxy=0.152;gap=0.572|保留为机制诊断|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_SAT_A|泛化可用|dm_proxy_vaccept较低；风险:source_episode_overflow高、radius/inter>1|source_overflow=0.765;p99=83.645;proxy=0.145;gap=0.785|可做Stage2诊断评估|保留同口径对照|
|DGLEO_UOPT_P0_SAT_B|泛化弱/不稳，receiver floor低|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.768;p99=83.974;proxy=0.142;gap=0.643|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|
|DGLEO_UOPT_P0_SAT_C|泛化可用|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.769;p99=83.646;proxy=0.145;gap=0.120|可做Stage2诊断评估|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_ADV_A|泛化弱/不稳，后期回落1.20pp|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.766;p99=84.424;proxy=0.147;gap=1.203|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_ADV_B|泛化可用|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.774;p99=84.933;proxy=0.143;gap=0.437|保留为机制诊断|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_ADV_C|泛化弱/不稳，后期回落1.10pp|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.771;p99=83.844;proxy=0.146;gap=1.105|保留为机制诊断|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_LATE_A|泛化弱/不稳，后期回落1.17pp|无明显改善；风险:source_episode_overflow高、radius/inter>1|source_overflow=0.763;p99=83.722;proxy=0.151;gap=1.168|保留为机制诊断|缩短pseudo阶段或冻结best评估|
|DGLEO_UOPT_P1_LATE_B|泛化弱/不稳，后期回落1.10pp，receiver floor低|无明显改善；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.769;p99=84.005;proxy=0.149;gap=1.102|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_LATE_C|泛化弱/不稳，后期回落1.67pp，receiver floor低|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.776;p99=84.147;proxy=0.148;gap=1.675|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_QUOTA_A|泛化弱/不稳，后期回落2.07pp，receiver floor低|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.769;p99=83.692;proxy=0.144;gap=2.067|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_QUOTA_B|泛化弱/不稳|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.773;p99=83.633;proxy=0.147;gap=0.548|保留为机制诊断|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_QUOTA_C|泛化弱/不稳，receiver floor低|p99收紧；dm_proxy_vaccept较低；风险:source_episode_overflow高、radius/inter>1|source_overflow=0.762;p99=83.468;proxy=0.144;gap=0.777|不建议Stage2；先修复弱receiver/后期回落|弱receiver定向satellite stress修复|
|DGLEO_UOPT_P1_STRONG_A|泛化可用|p99收紧；dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.775;p99=83.215;proxy=0.145;gap=0.000|可做Stage2诊断评估|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_STRONG_B|泛化弱/不稳，后期回落1.29pp|dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.766;p99=84.385;proxy=0.142;gap=1.288|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|
|DGLEO_UOPT_P1_STRONG_C|泛化弱/不稳，后期回落2.16pp，sat floor低|p99收紧；dm_proxy_vaccept较低；风险:source_overflow高、source_episode_overflow高、radius/inter>1|source_overflow=0.769;p99=83.179;proxy=0.148;gap=2.157|不建议Stage2；先修复弱receiver/后期回落|加source episode density gate/quarantine|

## 下一轮实验

|组|目标|变量|成功标准|失败判据|
|---|---|---|---|---|
|hard gate/local component dry-run|验证local component是否真能降低accept|固定P0_CORE_C和P0D_RADIUS_A，离线扫accept radius key、component margin、global ball off|proxy_vaccept、tail/overflow accept下降且receiver floor不降|proxy branch vaccept仍>0.60或receiver floor下降>1pp|
|真实Stage2 unknown query评估|把Phase1代理风险落到真实unknown|用ManyTx互斥Y_new/Y_unknown，只评估不训练|old/seen-new/unknown同row可解释，unknown FAR不恶化|任何unknown query参与阈值拟合即无效|
|shell/inter-class/bridge negative|直接打same-class bridge和类间低密度区|构造shell negative、inter-class negative、same-class bridge negative loss|bridge/low-density accept下降，p99不升|strict UDU下降>1pp或source_overflow升高|
|core/tail/outside quarantine|分离core/tail/outside样本|tail样本不参与扩张accept radius，outside进入quarantine|source_episode_overflow下降到<0.90|p95下降但p99/overflow不降|
|source episode density gate|修source episode overflow|按receiver-day密度和component density gate episode|source_episode_overflow明显下降，proxy branch vaccept同步下降|只降低direct metric proxy_vaccept但proxy branch不变|
|弱receiver satellite stress修复|补receiver floor与sat floor冲突|针对最弱rx的LEO增强、quota和teacher权重|receiver floor提升且sat floor不降|只提升sat平均，最弱receiver仍低|
|U_s日志/机制修复|证明U_s直接指标是否真生效|补`u_dm_p50/tail_cvar/selected`日志；mask交`valid_u_mask`；clean forward传`domain_labels=d_u`|U_s active epoch稳定，U_s p95/p99/source_overflow可解释|U_s direct metric大面积inactive或selected为空|
