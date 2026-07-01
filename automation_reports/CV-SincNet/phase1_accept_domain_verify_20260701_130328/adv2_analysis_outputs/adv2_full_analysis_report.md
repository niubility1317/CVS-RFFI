# ADV2实验全面机制审计报告

生成时间：2026-07-01 17:18:29 Asia/Hong_Kong

分析对象：`phase1_accept_domain_verify_20260701_130328`

证据根目录：`E:\type10-7\automation_reports\CV-SincNet\phase1_accept_domain_verify_20260701_130328\adv2_full_analysis_artifacts`

远端来源：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_accept_domain_verify_20260701_130328`与`runs/phase1_accept_domain_verify_20260701_130328`

## Executive Summary

1. 闭集DG相对lateopt有小幅提升，但相对vacuum有效32和四个父候选没有闭集均值优势，unknown拒识也没有被真实验证。ADV2的`final_overall_tx`均值为86.92%，相对lateopt变化0.48pp，相对vacuum32变化-0.31pp；`final_strict_udu`均值为80.09%，相对lateopt变化-0.15pp，相对vacuum32变化-0.62pp。这里仍是Phase1 source-only地面训练，不含真实`Y_unknown` query。
2. `proxy_vaccept≈1`仍是关键失败信号。14个候选`final_proxy_vaccept`均值0.9995、最小0.9981、最大1.0000，说明virtual unknown几乎仍被旧类能量/接收规则接收。
3. fusion导出问题在ADV2中已被修复到可审计层面。14/14个`phase2_zid_prototypes.json`包含`fusion_components`、`fused_tx_prototypes`和`fusion_config`，`fusion_accept_policy=local_component`且`global_ball_accept=False`。但这只证明导出字段存在，不证明local component gate已经让真实unknown拒识成功。
4. known中心和常规尾部更紧，但不是完整accept-domain收紧。ADV2`final_p95`均值52.20deg，低于vacuum32的52.42deg；但`final_p99`均值75.78deg，source overflow均值0.3442，仍显示极端尾部与跨域越界风险。
5. 最强闭集候选是`ADV2_SRCLOW_R17_E260`：`best_joint_test_tx=90.00%`、`final_strict_udu=82.75%`、`receiver_floor=74.63%`、`sat_strict_floor=69.47%`。它适合进入真实Stage2 unknown评估，但不能直接宣称拒识达标。
6. 主推进候选应按本轮同排数据重选，而不是沿用父候选角色。`ADV2_SRCLOW_R17_E260`是闭集主候选，`ADV2_FUSE5_R20_E260`是闭集+最低proxy_vac的融合候选，`ADV2_R28_FUSE6_E260`是p95最低且satellite floor较强的机制候选；`ADV2_R20_SAT70_E260`本轮`receiver_floor`只有63.81%，不能按“稳定主线”直接推进。
7. 下一步优先级不是继续盲扫参，而是先做不重训hard gate dry-run、shell/inter/bridge negative评估、真实Stage2-A/C unknown评估，再决定是否重训negative-space filling。

## Protocol Boundary

本轮是Phase1 source-only地面训练/弱标注半监督DG实验。训练协议为`split_mode=tx_rx_day_1_7_2`、`labeled_ratio=0.10`、`unlabeled_ratio=0.70`、`source_val_ratio=0.20`，训练期不使用target receiver数据。ADV2可说明闭集DG、特征空间几何、proxy unknown遥测、source episode风险和Phase2原型/fusion导出状态。

禁止声明：`unknown_FAR`下降、`FPR95`改善、真实unknown AUROC改善、Stage2-C成功、真实新类注册成功、local component gate已经部署成功。允许声明：闭集DG小幅改善、p95/min_inter等训练期几何指标改善、proxy unknown证据仍弱、fusion字段已导出但需要hard gate dry-run和真实unknown验证。

## Data Integrity and Run Health

有效候选数：14。排除候选：无。未发现`T16-T31`或纠偏前artifact混入ADV2统计。14/14完成260个epoch，14/14有`[PHASE2-EXPORT]`，14/14有`metrics_epoch.csv/jsonl`、`phase2_zid_prototypes.json/pt`。fatal模式`Traceback/RuntimeError/OOM/Killed/unrecognized arguments/FATAL`为0。

NaN分类：早期`[TEST] overall_tx=nan% (0/0)`属于`NAN_SKIPPED_TEST_PLACEHOLDER`；`[GRAD] aux=nan`和禁用辅助项的`sat_cos=nan/cons_cos=nan`属于`NAN_AUX_GRAD_TELEMETRY`或辅助遥测缺省；最终指标有限且训练完成。未发现`NAN_REAL_LOSS`、`NAN_REAL_METRIC`或`NAN_FATAL`影响本轮有效性。

| candidate_id | completed | expected_epochs | last_epoch | phase2_export | fatal_count | skipped_test_placeholder_nan_lines | nan_aux_grad_lines | inf_grad_lines | health_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADV2_FUSE5_R20_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 5 | PASS_WITH_TELEMETRY_NAN |
| ADV2_FUSE6_R17_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 5 | PASS_WITH_TELEMETRY_NAN |
| ADV2_R17_CORESTRICT_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 4 | PASS_WITH_TELEMETRY_NAN |
| ADV2_R17_PROXYHI_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 6 | PASS_WITH_TELEMETRY_NAN |
| ADV2_R20_SAT70_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 7 | PASS_WITH_TELEMETRY_NAN |
| ADV2_R20_VACMID_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 5 | PASS_WITH_TELEMETRY_NAN |
| ADV2_R28_FUSE6_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 6 | PASS_WITH_TELEMETRY_NAN |
| ADV2_R28_PROXYLOW_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 6 | PASS_WITH_TELEMETRY_NAN |
| ADV2_SOURCECAP32_R20_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 6 | PASS_WITH_TELEMETRY_NAN |
| ADV2_SRCLOW_R17_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 6 | PASS_WITH_TELEMETRY_NAN |
| ADV2_T13_CONSERVE_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 5 | PASS_WITH_TELEMETRY_NAN |
| ADV2_T13_TAILGUARD_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 4 | PASS_WITH_TELEMETRY_NAN |
| ADV2_TAILCV_R17_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 4 | PASS_WITH_TELEMETRY_NAN |
| ADV2_TAILCV_R20_E260 | True | 260 | 260 | True | 0 | 226 | 260 | 3 | PASS_WITH_TELEMETRY_NAN |

## Aggregate Metrics vs Baseline

ADV2相对lateopt有`final_overall_tx`提升，但相对vacuum32有效候选和四个父候选在`final_overall_tx`、`final_strict_udu`、`receiver_floor`、`sat_strict_floor`上均值回退。它的价值更偏向fusion导出修复、p95略降和source overflow均值下降，而不是闭集峰值压倒上一轮。

| metric | lateopt16 | vacuum_effective32 | vacuum_parent4 | ADV2_14 | delta_vs_vacuum_effective32 | delta_vs_vacuum_parent4 |
| --- | --- | --- | --- | --- | --- | --- |
| best_joint_epoch | 234.3750 | 257.1875 | 268.5000 | 246.4286 | -10.7589 | -22.0714 |
| best_joint_test_tx | 88.0031 | 88.3578 | 89.3575 | 88.0272 | -0.3306 | -1.3303 |
| final_min_inter | 82.7719 | 83.5369 | 84.5550 | 83.3107 | -0.2262 | -1.2443 |
| final_overall_tx | 86.4375 | 87.2347 | 88.7075 | 86.9211 | -0.3135 | -1.7864 |
| final_ow_vac_rate |  | 0.0179 | 0.0138 | 0.0178 | -0.0001 | 0.0041 |
| final_p95 | 53.4769 | 52.4153 | 51.4250 | 52.1951 | -0.2202 | 0.7701 |
| final_p99 |  |  |  | 75.7844 |  |  |
| final_pos_angle | 32.0063 | 31.6581 | 31.3275 | 31.6906 | 0.0325 | 0.3631 |
| final_proxy_auc |  | 0.5693 | 0.5675 | 0.5679 | -0.0014 | 0.0005 |
| final_proxy_vac_rate |  | 0.3552 | 0.3320 | 0.3488 | -0.0064 | 0.0168 |
| final_proxy_vaccept |  | 0.9998 | 0.9998 | 0.9995 | -0.0003 | -0.0003 |
| final_receiver_floor | 69.4350 | 70.2641 | 72.8125 | 69.0714 | -1.1926 | -3.7411 |
| final_sat_mean | 76.0056 | 76.9278 | 77.5475 | 75.9799 | -0.9479 | -1.5676 |
| final_sat_strict_floor | 68.0031 | 68.8772 | 69.7825 | 67.8337 | -1.0435 | -1.9488 |
| final_source_overflow | 0.3312 | 0.3650 | 0.2904 | 0.3442 | -0.0208 | 0.0538 |
| final_strict_udu | 80.2381 | 80.7047 | 82.9125 | 80.0867 | -0.6180 | -2.8258 |

关键解释：相对lateopt的`final_overall_tx`提升不能掩盖相对vacuum32的闭集回退；`final_p95`略降和`source_overflow`均值下降是好信号；`proxy_vaccept`没有改善到可用拒识水平；`final_proxy_auc`仍接近0.56-0.57的弱判别区间；候选间差异大，不能用均值掩盖`SOURCECAP32`、`TAILCV`等高风险样本。

## Group Ablation

| group_type | group | n | best_joint_test_tx_mean | final_strict_udu_mean | final_receiver_floor_mean | final_sat_strict_floor_mean | final_p95_mean | final_p99_mean | final_min_inter_mean | final_proxy_auc_mean | final_proxy_vac_rate_mean | final_proxy_vaccept_mean | final_source_overflow_mean | prototype_radius_p95_mean_deg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| family | R | 12 | 88.0844 | 80.3119 | 69.8500 | 68.0665 | 52.3009 | 75.9617 | 83.2513 | 0.5684 | 0.3492 | 0.9995 | 0.3576 | 5.8081 |
| family | T | 2 | 87.6843 | 78.7350 | 64.4000 | 66.4367 | 51.5604 | 74.7206 | 83.6673 | 0.5655 | 0.3462 | 0.9990 | 0.2638 | 5.3302 |
| base | R17 | 5 | 88.7507 | 80.2170 | 70.4283 | 68.5213 | 52.6315 | 76.2183 | 83.0208 | 0.5672 | 0.3637 | 0.9996 | 0.3666 | 5.6922 |
| base | R20 | 5 | 87.3903 | 80.2857 | 67.9100 | 67.5283 | 52.3283 | 75.8873 | 83.4511 | 0.5690 | 0.3447 | 0.9992 | 0.3571 | 5.9955 |
| base | R28 | 2 | 88.1537 | 80.6150 | 73.2542 | 68.2750 | 51.4062 | 75.5061 | 83.3276 | 0.5697 | 0.3241 | 1.0000 | 0.3364 | 5.6295 |
| base | T13 | 2 | 87.6843 | 78.7350 | 64.4000 | 66.4367 | 51.5604 | 74.7206 | 83.6673 | 0.5655 | 0.3462 | 0.9990 | 0.2638 | 5.3302 |
| mechanism_tag | CONSERVE | 1 | 87.7647 | 79.1967 | 66.9250 | 66.7667 | 50.7631 | 74.6628 | 83.7785 | 0.5670 | 0.3210 | 0.9990 | 0.2234 | 5.2949 |
| mechanism_tag | CORESTRICT | 1 | 88.3672 | 81.5550 | 69.0667 | 66.6817 | 53.1563 | 75.9832 | 83.3016 | 0.5744 | 0.3454 | 1.0000 | 0.3776 | 5.2926 |
| mechanism_tag | FUSE5 | 1 | 88.6951 | 82.4850 | 71.3417 | 67.8133 | 51.7161 | 74.4287 | 84.3229 | 0.5704 | 0.3050 | 1.0000 | 0.2990 | 5.7383 |
| mechanism_tag | FUSE6 | 2 | 88.0542 | 81.4717 | 71.2833 | 70.5908 | 51.0158 | 75.8656 | 83.4245 | 0.5681 | 0.3519 | 1.0000 | 0.3791 | 5.5512 |
| mechanism_tag | PROXYHI | 1 | 87.0255 | 76.3067 | 67.7083 | 66.6417 | 53.0107 | 75.8844 | 81.2207 | 0.5710 | 0.3923 | 1.0000 | 0.3905 | 6.8594 |
| mechanism_tag | PROXYLOW | 1 | 89.3221 | 79.4733 | 75.1833 | 65.2183 | 53.2421 | 75.8464 | 82.5457 | 0.5638 | 0.3272 | 1.0000 | 0.3035 | 5.4459 |
| mechanism_tag | SAT70 | 1 | 86.8221 | 78.1900 | 63.8083 | 64.7800 | 52.0785 | 76.0883 | 82.8360 | 0.5656 | 0.3434 | 1.0000 | 0.3016 | 5.2490 |
| mechanism_tag | SOURCECAP32 | 1 | 86.1569 | 80.1200 | 69.0500 | 67.1567 | 51.3267 | 72.6236 | 83.2714 | 0.5671 | 0.3234 | 0.9990 | 0.4738 | 6.9162 |
| mechanism_tag | SRCLOW | 1 | 89.9985 | 82.7533 | 74.6333 | 69.4667 | 52.5451 | 77.3754 | 84.4580 | 0.5630 | 0.3279 | 0.9981 | 0.2108 | 5.6552 |
| mechanism_tag | TAILCV | 2 | 88.2934 | 79.8550 | 69.5542 | 68.6692 | 53.2861 | 77.0025 | 82.7776 | 0.5671 | 0.3846 | 0.9995 | 0.4433 | 5.6958 |
| mechanism_tag | TAILGUARD | 1 | 87.6039 | 78.2733 | 61.8750 | 66.1067 | 52.3577 | 74.7783 | 83.5561 | 0.5639 | 0.3714 | 0.9990 | 0.3041 | 5.3656 |
| mechanism_tag | VACMID | 1 | 87.9299 | 80.2067 | 65.7333 | 70.5200 | 51.9321 | 77.5743 | 84.6547 | 0.5746 | 0.3526 | 0.9981 | 0.2899 | 6.0467 |

分组结论：

- R组均值闭集更强，T13组更像保守尾部对照。T组只有2个候选，结论不能过度泛化。
- R17系更偏闭集峰值和强推进；R20系内部差异大，只有`FUSE5_R20`达到主推进门槛；R28系更多是proxy/fusion机制候选，其中`R28_FUSE6`优于`R28_PROXYLOW`的satellite floor；T13系用于保守对照但本轮闭集不足。
- `FUSE5/FUSE6`标签组不能简单解释为性能增益，因为所有ADV2候选都已导出fusion字段，差异来自组件数、radius cap、source/vacuum配置的组合。
- `TAILCV/TAILGUARD`没有把proxy_vaccept压下来，说明单纯tail压力不足以形成拒识面。

## Candidate-Level Deep Dive

候选级同排指标如下，排序优先展示综合推进分可计算候选。`composite_score`仅在`final_strict_udu>=82`、`receiver_floor>=70`、`best_joint_test_tx>=88`的闭集门槛内计算。

| candidate_id | mechanism_tag | best_joint_test_tx | final_overall_tx | final_strict_udu | final_receiver_floor | final_sat_strict_floor | final_p95 | final_p99 | final_min_inter | final_proxy_auc | final_proxy_vac_rate | final_proxy_vaccept | final_source_overflow | prototype_radius_p95_mean_deg | tx_domain_components_mean | closed_score | geometry_score | proxy_score | pareto_front |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADV2_SRCLOW_R17_E260 | SRCLOW | 89.9985 | 88.8108 | 82.7533 | 74.6333 | 69.4667 | 52.5451 | 77.3754 | 84.4580 | 0.5630 | 0.3279 | 0.9981 | 0.2108 | 5.6552 | 1.3333 | 1.8000 | 3.7000 | 7.5500 | True |
| ADV2_FUSE5_R20_E260 | FUSE5 | 88.6951 | 88.8623 | 82.4850 | 71.3417 | 67.8133 | 51.7161 | 74.4287 | 84.3229 | 0.5704 | 0.3050 | 1.0000 | 0.2990 | 5.7383 | 2.1667 | 3.8500 | 3.9500 | 4.3000 | True |
| ADV2_FUSE6_R17_E260 | FUSE6 | 89.1230 | 87.4603 | 81.1867 | 71.2417 | 69.8500 | 52.4613 | 76.5652 | 82.7394 | 0.5606 | 0.3827 | 1.0000 | 0.3889 | 5.2894 | 2.6667 | 4.5000 | 10.4000 | 12.4000 | True |
| ADV2_R17_CORESTRICT_E260 | CORESTRICT | 88.3672 | 88.0691 | 81.5550 | 69.0667 | 66.6817 | 53.1563 | 75.9832 | 83.3016 | 0.5744 | 0.3454 | 1.0000 | 0.3776 | 5.2926 | 1.5000 | 6.4000 | 9.3000 | 6.7500 | True |
| ADV2_R17_PROXYHI_E260 | PROXYHI | 87.0255 | 85.3926 | 76.3067 | 67.7083 | 66.6417 | 53.0107 | 75.8844 | 81.2207 | 0.5710 | 0.3923 | 1.0000 | 0.3905 | 6.8594 | 1.3333 | 11.7000 | 12.3500 | 9.3500 | True |
| ADV2_R20_SAT70_E260 | SAT70 | 86.8221 | 85.6127 | 78.1900 | 63.8083 | 64.7800 | 52.0785 | 76.0883 | 82.8360 | 0.5656 | 0.3434 | 1.0000 | 0.3016 | 5.2490 | 1.5000 | 13.1500 | 7.5500 | 8.7500 | False |
| ADV2_R20_VACMID_E260 | VACMID | 87.9299 | 87.0990 | 80.2067 | 65.7333 | 70.5200 | 51.9321 | 77.5743 | 84.6547 | 0.5746 | 0.3526 | 0.9981 | 0.2899 | 6.0467 | 1.5000 | 7.2500 | 2.6000 | 5.0500 | True |
| ADV2_R28_FUSE6_E260 | FUSE6 | 86.9853 | 86.9853 | 81.7567 | 71.3250 | 71.3317 | 49.5703 | 75.1659 | 84.1096 | 0.5756 | 0.3210 | 1.0000 | 0.3692 | 5.8130 | 2.8333 | 6.0500 | 4.2500 | 3.3500 | True |
| ADV2_R28_PROXYLOW_E260 | PROXYLOW | 89.3221 | 86.5686 | 79.4733 | 75.1833 | 65.2183 | 53.2421 | 75.8464 | 82.5457 | 0.5638 | 0.3272 | 1.0000 | 0.3035 | 5.4459 | 1.3333 | 5.5500 | 10.0500 | 8.5500 | True |
| ADV2_SOURCECAP32_R20_E260 | SOURCECAP32 | 86.1569 | 86.4990 | 80.1200 | 69.0500 | 67.1567 | 51.3267 | 72.6236 | 83.2714 | 0.5671 | 0.3234 | 0.9990 | 0.4738 | 6.9162 | 1.8333 | 10.3000 | 9.4000 | 5.1500 | True |
| ADV2_T13_CONSERVE_E260 | CONSERVE | 87.7647 | 85.5054 | 79.1967 | 66.9250 | 66.7667 | 50.7631 | 74.6628 | 83.7785 | 0.5670 | 0.3210 | 0.9990 | 0.2234 | 5.2949 | 1.6667 | 9.6500 | 3.1500 | 5.0500 | True |
| ADV2_T13_TAILGUARD_E260 | TAILGUARD | 87.6039 | 85.4441 | 78.2733 | 61.8750 | 66.1067 | 52.3577 | 74.7783 | 83.5561 | 0.5639 | 0.3714 | 0.9990 | 0.3041 | 5.3656 | 1.6667 | 11.3500 | 6.8000 | 9.7000 | False |
| ADV2_TAILCV_R17_E260 | TAILCV | 89.2392 | 87.5863 | 79.2833 | 69.4917 | 69.9667 | 51.9841 | 75.2834 | 83.3845 | 0.5669 | 0.3701 | 1.0000 | 0.4653 | 5.3645 | 1.5000 | 5.9000 | 8.5500 | 9.7500 | True |
| ADV2_TAILCV_R20_E260 | TAILCV | 87.3475 | 87.0005 | 80.4267 | 69.6167 | 67.3717 | 54.5880 | 78.7217 | 82.1707 | 0.5674 | 0.3991 | 0.9990 | 0.4213 | 6.0272 | 1.8333 | 7.5500 | 12.9500 | 9.3000 | True |

闭集推进分：

| candidate_id | best_joint_test_tx | final_strict_udu | final_receiver_floor | final_sat_strict_floor | closed_score |
| --- | --- | --- | --- | --- | --- |
| ADV2_SRCLOW_R17_E260 | 89.9985 | 82.7533 | 74.6333 | 69.4667 | 1.8000 |
| ADV2_FUSE5_R20_E260 | 88.6951 | 82.4850 | 71.3417 | 67.8133 | 3.8500 |
| ADV2_FUSE6_R17_E260 | 89.1230 | 81.1867 | 71.2417 | 69.8500 | 4.5000 |
| ADV2_R28_PROXYLOW_E260 | 89.3221 | 79.4733 | 75.1833 | 65.2183 | 5.5500 |
| ADV2_TAILCV_R17_E260 | 89.2392 | 79.2833 | 69.4917 | 69.9667 | 5.9000 |
| ADV2_R28_FUSE6_E260 | 86.9853 | 81.7567 | 71.3250 | 71.3317 | 6.0500 |
| ADV2_R17_CORESTRICT_E260 | 88.3672 | 81.5550 | 69.0667 | 66.6817 | 6.4000 |
| ADV2_R20_VACMID_E260 | 87.9299 | 80.2067 | 65.7333 | 70.5200 | 7.2500 |
| ADV2_TAILCV_R20_E260 | 87.3475 | 80.4267 | 69.6167 | 67.3717 | 7.5500 |
| ADV2_T13_CONSERVE_E260 | 87.7647 | 79.1967 | 66.9250 | 66.7667 | 9.6500 |

几何安全分：

| candidate_id | final_p95 | final_p99 | final_min_inter | final_source_overflow | final_ow_vac_rate | geometry_score |
| --- | --- | --- | --- | --- | --- | --- |
| ADV2_R20_VACMID_E260 | 51.9321 | 77.5743 | 84.6547 | 0.2899 | 0.0087 | 2.6000 |
| ADV2_T13_CONSERVE_E260 | 50.7631 | 74.6628 | 83.7785 | 0.2234 | 0.0111 | 3.1500 |
| ADV2_SRCLOW_R17_E260 | 52.5451 | 77.3754 | 84.4580 | 0.2108 | 0.0104 | 3.7000 |
| ADV2_FUSE5_R20_E260 | 51.7161 | 74.4287 | 84.3229 | 0.2990 | 0.0123 | 3.9500 |
| ADV2_R28_FUSE6_E260 | 49.5703 | 75.1659 | 84.1096 | 0.3692 | 0.0111 | 4.2500 |
| ADV2_T13_TAILGUARD_E260 | 52.3577 | 74.7783 | 83.5561 | 0.3041 | 0.0168 | 6.8000 |
| ADV2_R20_SAT70_E260 | 52.0785 | 76.0883 | 82.8360 | 0.3016 | 0.0202 | 7.5500 |
| ADV2_TAILCV_R17_E260 | 51.9841 | 75.2834 | 83.3845 | 0.4653 | 0.0174 | 8.5500 |
| ADV2_R17_CORESTRICT_E260 | 53.1563 | 75.9832 | 83.3016 | 0.3776 | 0.0178 | 9.3000 |
| ADV2_SOURCECAP32_R20_E260 | 51.3267 | 72.6236 | 83.2714 | 0.4738 | 0.0219 | 9.4000 |

拒识代理分：

| candidate_id | final_proxy_auc | final_proxy_vac_rate | final_proxy_vaccept | proxy_score |
| --- | --- | --- | --- | --- |
| ADV2_R28_FUSE6_E260 | 0.5756 | 0.3210 | 1.0000 | 3.3500 |
| ADV2_FUSE5_R20_E260 | 0.5704 | 0.3050 | 1.0000 | 4.3000 |
| ADV2_R20_VACMID_E260 | 0.5746 | 0.3526 | 0.9981 | 5.0500 |
| ADV2_T13_CONSERVE_E260 | 0.5670 | 0.3210 | 0.9990 | 5.0500 |
| ADV2_SOURCECAP32_R20_E260 | 0.5671 | 0.3234 | 0.9990 | 5.1500 |
| ADV2_R17_CORESTRICT_E260 | 0.5744 | 0.3454 | 1.0000 | 6.7500 |
| ADV2_SRCLOW_R17_E260 | 0.5630 | 0.3279 | 0.9981 | 7.5500 |
| ADV2_R28_PROXYLOW_E260 | 0.5638 | 0.3272 | 1.0000 | 8.5500 |
| ADV2_R20_SAT70_E260 | 0.5656 | 0.3434 | 1.0000 | 8.7500 |
| ADV2_TAILCV_R20_E260 | 0.5674 | 0.3991 | 0.9990 | 9.3000 |

## Geometry and Acceptance-Domain Analysis

| risk_region | observed_metric | current_value_or_evidence | why_it_matters | next_test |
| --- | --- | --- | --- | --- |
| known core | final_overall mean 86.92% | 闭集中心仍强 | 支撑Phase1闭集DG有效，但不等于unknown拒识 | Stage2 old/query真实评估 |
| known soft tail | p95 mean 52.20deg | 比vacuum32低 | p95只覆盖常规尾部 | p95+p99+CVaR联合门控 |
| known extreme tail | p99 mean 75.78deg | 已可量化 | p99仍长，可能污染accept半径 | tail quarantine/CVaR97 |
| source cross-domain overflow | overflow mean 0.3442 | 低于vacuum32 | 跨域query仍有较大比例越界 | source_episode_density_gate |
| inter-class low-density zone | min_inter mean 83.31deg | 类中心分离很高 | 中心角不覆盖类间低密度带 | inter-class slerp negative |
| same-class multi-mode bridge | fusion components mean 1.76/class | local component已导出 | 多组件之间空洞可能被错误接收 | same-class bridge negative |
| old-class shell outside r_accept | prototype p95 mean 5.74deg,p99 40.27deg | local accept字段可审计 | p99远大于p95时自动接收必须限制 | shell accept dry-run |
| proxy unknown near tail | proxy_vac_rate mean 0.3488 | 部分候选下降 | proxy点仍不能代表真实unknown | tail-outward proxy sampler |
| virtual unknown accepted by energy | proxy_vaccept mean 0.9995 | 暴露失败面 | 几乎全接收，拒识面未形成 | energy/density/geometric hard gate |
| unknown not evaluated | unknown_FAR/FPR95/real AUROC缺失 | 边界清楚 | 不能声明部署拒识成功 | 真实Y_unknown Stage2-A/C |

最终判断：ADV2更像是“known中心与p95常规包络略收紧+fusion导出可审计”，不是“known接收域真正变紧”。真正的automatic accept域必须通过local component hard gate、density gate、NLL/Mahalanobis gate、geo margin gate和energy/reject gate重新定义，并且tail/outside样本不能自动计入known accept。

## Proxy Unknown and Vacuum Mechanism Analysis

`ow_vac_rate`低只能说明旧类foreign-tail intrusion在训练代理面减少，不等于unknown被拒绝。`proxy_vac_rate`下降只说明virtual/proxy点较少进入某些真空带，不代表不会被known energy/softmax接收。`proxy_vaccept≈1`是最严重失败信号，说明最终接收规则仍把几乎所有virtual unknown当known接收。

| metric | interpretation | good_sign | bad_sign | current_result | conclusion |
| --- | --- | --- | --- | --- | --- |
| `ow_vac_rate` | 旧类特征进入真空带的代理率 | 下降 | 不能代表真实unknown | 均值0.0178 | 只能作为几何辅助 |
| `proxy_auc` | proxy/known能量区分度 | 越高越好 | 0.56级很弱 | 均值0.5679 | 不足以支撑拒识声明 |
| `proxy_vac_rate` | proxy进入真空带比例 | 下降 | 与真实unknown不同分布 | 均值0.3488 | 可做机制筛选 |
| `proxy_vaccept` | virtual unknown被接收比例 | 应显著低于1 | 接近1表示拒识面失败 | 均值0.9995 | 当前最大失败面 |
| `source_overflow` | source episode越过3σ比例 | 越低越好 | 高值扩大known包络 | 均值0.3442 | 需density gate |
| `p95/p99` | known半径常规尾/极端尾 | p95下降 | p99长尾污染accept | p95均值52.20deg,p99均值75.78deg | 不能只看p95 |
| `min_inter` | 类中心分离 | 越高越好 | 不覆盖低密度/尾部风险 | 均值83.31deg | 不等价于拒识成功 |

proxy unknown设计缺陷仍在：virtual outlier可能太靠近known manifold；leave-one-TX-out不等价于真实unknown；没有shell negative、tail-outward negative、inter-class slerp negative和same-class bridge negative；energy surface没有被训练成open-space reject地形。

## Source Episode Risk Analysis

source episode的目标不能是“所有跨域query都回到known 3σ内”。ADV2需要把query分为core/uncertain/outside：只有高密度core query才自动known拉近，低密度query进入uncertain/risk/reject。否则闭集提升可能来自扩大known包络，反而损害真实unknown拒识。

| candidate_id | best_joint_epoch | final_epoch | final_minus_best_test | final_minus_best_strict | p95_min_epoch | final_minus_min_p95 | min_inter_max_epoch | final_minus_max_min_inter | proxy_vaccept_final | source_overflow_final | source_overflow_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADV2_FUSE5_R20_E260 | 250 | 260 | 0.1672 | -1.2567 | 241 | 3.0091 | 241 | -0.3540 | 1.0000 | 0.2990 | -0.0523 |
| ADV2_FUSE6_R17_E260 | 256 | 260 | -1.6627 | -2.2650 | 258 | 3.3840 | 225 | -1.7390 | 1.0000 | 0.3889 | -0.0436 |
| ADV2_R17_CORESTRICT_E260 | 242 | 260 | -0.2980 | -0.2133 | 248 | 4.6940 | 245 | -1.6422 | 1.0000 | 0.3776 | -0.0331 |
| ADV2_R17_PROXYHI_E260 | 246 | 260 | -1.6328 | -4.1167 | 254 | 3.9482 | 254 | -3.4681 | 1.0000 | 0.3905 | -0.0075 |
| ADV2_R20_SAT70_E260 | 230 | 260 | -1.2093 | -2.7800 | 252 | 2.8445 | 252 | -1.9983 | 1.0000 | 0.3016 | -0.0163 |
| ADV2_R20_VACMID_E260 | 258 | 260 | -0.8309 | -1.9783 | 247 | 3.1794 | 256 | -0.3289 | 0.9981 | 0.2899 | -0.0335 |
| ADV2_R28_FUSE6_E260 | 260 | 260 | 0.0000 | 0.0000 | 260 | 0.0000 | 246 | -0.9912 | 1.0000 | 0.3692 | -0.0238 |
| ADV2_R28_PROXYLOW_E260 | 256 | 260 | -2.7534 | -4.1433 | 251 | 4.0246 | 251 | -2.6040 | 1.0000 | 0.3035 | -0.0269 |
| ADV2_SOURCECAP32_R20_E260 | 250 | 260 | 0.3422 | -0.5600 | 240 | 2.0415 | 247 | -1.4219 | 0.9990 | 0.4738 | -0.0165 |
| ADV2_SRCLOW_R17_E260 | 254 | 260 | -1.1877 | -2.5317 | 248 | 1.2883 | 249 | -0.5199 | 0.9981 | 0.2108 | -0.0343 |
| ADV2_T13_CONSERVE_E260 | 230 | 260 | -2.2593 | -3.2317 | 225 | 0.0239 | 255 | -1.3868 | 0.9990 | 0.2234 | -0.0336 |
| ADV2_T13_TAILGUARD_E260 | 252 | 260 | -2.1598 | -2.9983 | 233 | 1.0650 | 253 | -1.3817 | 0.9990 | 0.3041 | -0.0181 |
| ADV2_TAILCV_R17_E260 | 246 | 260 | -1.6529 | -2.9117 | 257 | 4.0763 | 235 | -2.1181 | 1.0000 | 0.4653 | -0.0549 |
| ADV2_TAILCV_R20_E260 | 220 | 260 | -0.3471 | -0.1950 | 258 | 3.9652 | 234 | -1.8291 | 0.9990 | 0.4213 | -0.0257 |

## Prototype Export and Fusion Audit

ADV2的fusion审计结论不同于vacuum：本轮14/14原型JSON都存在`fusion_components`、`fused_tx_prototypes`、`fusion_config`，且`fusion_config.enabled=True`、`accept_policy=local_component`、`global_ball_accept=False`、`tail_auto_accept=False`。这说明导出路径已经执行，`[PHASE2-EXPORT] fused=1`与JSON字段一致。

但本轮仍不能写“local component gate部署成功”，原因是：导出字段只定义组件和半径，没有在真实Stage2 unknown query上给出`unknown_FAR/FPR95/AUROC`；也没有shell/inter/bridge synthetic hard gate dry-run的accept率。

| candidate_id | json_exists | pt_exists | n_classes | n_domains | samples | active_domain_prototypes_mean | tx_domain_components_mean | has_fusion_components | has_fused_tx_prototypes | has_fusion_config | fusion_accept_policy | fusion_global_ball_accept | fusion_tail_auto_accept | prototype_radius_p95_mean_deg | prototype_radius_p99_mean_deg | prototype_min_inter_deg | prototype_tail_frac_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADV2_FUSE5_R20_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 2.1667 | True | True | True | local_component | False | False | 5.7383 | 29.7673 | 88.8937 | 0.1073 |
| ADV2_FUSE6_R17_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 2.6667 | True | True | True | local_component | False | False | 5.2894 | 38.2926 | 89.2775 | 0.0868 |
| ADV2_R17_CORESTRICT_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.5000 | True | True | True | local_component | False | False | 5.2926 | 42.6986 | 89.2555 | 0.0977 |
| ADV2_R17_PROXYHI_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.3333 | True | True | True | local_component | False | False | 6.8594 | 41.9863 | 88.6240 | 0.0886 |
| ADV2_R20_SAT70_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.5000 | True | True | True | local_component | False | False | 5.2490 | 39.6328 | 88.6963 | 0.0729 |
| ADV2_R20_VACMID_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.5000 | True | True | True | local_component | False | False | 6.0467 | 41.5859 | 89.1172 | 0.0970 |
| ADV2_R28_FUSE6_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 2.8333 | True | True | True | local_component | False | False | 5.8130 | 48.2760 | 89.0821 | 0.0871 |
| ADV2_R28_PROXYLOW_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.3333 | True | True | True | local_component | False | False | 5.4459 | 35.4175 | 89.2704 | 0.0806 |
| ADV2_SOURCECAP32_R20_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.8333 | True | True | True | local_component | False | False | 6.9162 | 35.3842 | 88.7271 | 0.1013 |
| ADV2_SRCLOW_R17_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.3333 | True | True | True | local_component | False | False | 5.6552 | 43.0880 | 89.2372 | 0.0763 |
| ADV2_T13_CONSERVE_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.6667 | True | True | True | local_component | False | False | 5.2949 | 38.7685 | 89.3834 | 0.0865 |
| ADV2_T13_TAILGUARD_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.6667 | True | True | True | local_component | False | False | 5.3656 | 43.2367 | 89.4939 | 0.0613 |
| ADV2_TAILCV_R17_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.5000 | True | True | True | local_component | False | False | 5.3645 | 39.6953 | 89.5630 | 0.0808 |
| ADV2_TAILCV_R20_E260 | True | True | 6 | 26 | 8320 | 14.0000 | 1.8333 | True | True | True | local_component | False | False | 6.0272 | 45.9117 | 88.7673 | 0.0902 |

注意区分训练日志中的`final_ow_p95/final_ow_min_inter`与导出包中的`prototype_radius_p95_mean_deg/prototype_min_inter_deg`。前者是训练期特征损失遥测，后者是导出原型包的接收半径/类间角审计面。

## Correlation and Trade-off Analysis

| pair | pearson | spearman | n |
| --- | --- | --- | --- |
| final_strict_udu vs final_source_overflow | -0.1508 | -0.2000 | 14 |
| final_strict_udu vs final_p95 | -0.1603 | -0.0418 | 14 |
| final_strict_udu vs final_proxy_vac_rate | -0.4755 | -0.3538 | 14 |
| final_strict_udu vs final_proxy_auc | 0.1115 | 0.1341 | 14 |
| final_receiver_floor vs final_source_overflow | 0.0100 | -0.0286 | 14 |
| final_p95 vs final_proxy_vac_rate | 0.6150 | 0.6440 | 14 |
| final_p95 vs final_proxy_auc | -0.3005 | -0.3011 | 14 |
| final_p95 vs final_source_overflow | 0.1425 | 0.1824 | 14 |
| final_min_inter vs final_proxy_vac_rate | -0.6432 | -0.5341 | 14 |
| final_min_inter vs final_proxy_auc | 0.2271 | 0.2088 | 14 |
| final_min_inter vs final_source_overflow | -0.4691 | -0.6352 | 14 |
| final_ow_vac_rate vs final_proxy_vac_rate | 0.6848 | 0.5341 | 14 |
| final_ow_vac_rate vs final_source_overflow | 0.6121 | 0.7187 | 14 |
| best_joint_test_tx vs final_strict_udu | 0.4477 | 0.4022 | 14 |
| best_joint_test_tx vs final_receiver_floor | 0.5894 | 0.5780 | 14 |
| expected_epochs vs final_overall_tx |  |  | 14 |
| expected_epochs vs final_strict_udu |  |  | 14 |
| prototype_radius_p95_mean_deg vs final_p95 | 0.0153 | -0.1077 | 14 |
| tx_domain_components_mean vs prototype_radius_p95_mean_deg | -0.0579 | 0.0246 | 14 |

相关性解释只作为提示，不作为因果结论。需要关注三点：第一，`min_inter`高不必然带来proxy unknown改善；第二，`p95`低不必然压低source overflow；第三，proxy_vac_rate低的候选如果闭集不足，只能作为机制候选，不能越过闭集门槛成为主推进。

## Failure Modes and Root Causes

| failure_mode | evidence | symptoms | likely_root_cause | impact_on_unknown_rejection | how_to_test_next | fix_priority |
| --- | --- | --- | --- | --- | --- | --- |
| proxy_vaccept接近1 | 均值0.9995，min0.9981 | virtual unknown几乎全被接收 | energy阈值和接收规则没有形成open-space reject地形 | 真实unknown FAR大概率仍高 | 真实unknown+shell/inter-slerp/same-class bridge accept dry-run | P0 |
| p95改善但p99长尾仍在 | p95均值52.20deg，p99均值75.78deg | 常规包络收紧，极端尾部仍长 | 3sigma/accept半径仍被tail污染 | unknown可能落入旧类尾部 | CVaR97、tail quarantine、tail_auto_accept=false断言 | P0 |
| min_inter高但拒识弱 | min_inter均值83.31deg，proxy_auc均值0.5679 | 类中心远离但proxy判别弱 | 开放空间风险由tail/低密度/桥接区决定 | 继续单推min_inter收益有限 | inter-class slerp negative评估 | P1 |
| source_episode_overflow仍有风险 | 均值0.3442，最高ADV2_SOURCECAP32_R20_E260=0.4738 | 部分source约束扩大known包络 | 跨域query被强拉为known | unknown误接收风险增加 | source_episode_query_core_only/density_gate | P0 |
| fusion已导出但未验证拒识收益 | 14/14 JSON有fusion字段，global_ball_accept均False | 实现路径已通，但只是导出成功 | 缺少local component hard gate dry-run和真实unknown | 不能说local component gate部署成功 | 不重训hard gate dry-run | P0 |
| proxy unknown覆盖不足 | proxy_auc均值0.5679 | proxy区分度弱 | virtual/leave-one-TX-out未覆盖shell/bridge/tail-outward | 真实unknown风险未被训练面覆盖 | 四类negative sampler | P1 |
| E260缩短训练不是万能 | best epoch均值246.4，final-best test均值-1.11pp | 部分候选final低于best | 后期joint guard仍需更细 | 不能默认final checkpoint最佳 | joint early-stop/hard gate checkpoint selection | P1 |
| NaN遥测污染健康判断 | 每候选skipped-test NaN均226.0行，aux grad NaN均260.0行 | 占位/辅助NaN混杂 | parser若不分类会误报 | 自动化健康判断不稳定 | NaN分类parser落地 | P1 |

## Promotion Decision

| candidate_id | category | promote_to_stage2 | use_for_mechanism | reject_or_risk_reason | required_followup |
| --- | --- | --- | --- | --- | --- |
| ADV2_FUSE5_R20_E260 | Stage2真实unknown评估主推进候选 | True | False | 闭集/receiver floor/对照价值较强，但必须真实unknown验证。proxy_vaccept仍接近1。 | local hard gate dry-run+真实Stage2 unknown评估 |
| ADV2_FUSE6_R17_E260 | 机制诊断候选 | False | True | 用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。proxy_vaccept仍接近1。 | 机制隔离或负例分析 |
| ADV2_R17_CORESTRICT_E260 | 机制诊断候选 | False | True | 用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。proxy_vaccept仍接近1。receiver floor不足。 | 机制隔离或负例分析 |
| ADV2_R17_PROXYHI_E260 | 不建议直接推进 | False | False | proxy_vaccept仍接近1。receiver floor不足。 | 机制隔离或负例分析 |
| ADV2_R20_SAT70_E260 | 不建议直接推进 | False | False | proxy_vaccept仍接近1。receiver floor不足。 | 机制隔离或负例分析 |
| ADV2_R20_VACMID_E260 | 不建议直接推进 | False | False | proxy_vaccept仍接近1。receiver floor不足。 | 机制隔离或负例分析 |
| ADV2_R28_FUSE6_E260 | Stage2真实unknown评估主推进候选 | True | False | 闭集/receiver floor/对照价值较强，但必须真实unknown验证。proxy_vaccept仍接近1。 | local hard gate dry-run+真实Stage2 unknown评估 |
| ADV2_R28_PROXYLOW_E260 | 机制诊断候选 | False | True | 用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。proxy_vaccept仍接近1。 | 机制隔离或负例分析 |
| ADV2_SOURCECAP32_R20_E260 | 高风险机制/负例候选 | False | True | 用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。proxy_vaccept仍接近1。source_overflow过高。receiver floor不足。 | 机制隔离或负例分析 |
| ADV2_SRCLOW_R17_E260 | Stage2真实unknown评估主推进候选 | True | False | 闭集/receiver floor/对照价值较强，但必须真实unknown验证。proxy_vaccept仍接近1。 | local hard gate dry-run+真实Stage2 unknown评估 |
| ADV2_T13_CONSERVE_E260 | 机制诊断候选 | False | True | 用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。proxy_vaccept仍接近1。receiver floor不足。 | 机制隔离或负例分析 |
| ADV2_T13_TAILGUARD_E260 | 机制诊断候选 | False | True | 用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。proxy_vaccept仍接近1。receiver floor不足。 | 机制隔离或负例分析 |
| ADV2_TAILCV_R17_E260 | 高风险机制/负例候选 | False | True | 用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。proxy_vaccept仍接近1。source_overflow过高。receiver floor不足。 | 机制隔离或负例分析 |
| ADV2_TAILCV_R20_E260 | 机制诊断候选 | False | True | 用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。proxy_vaccept仍接近1。receiver floor不足。 | 机制隔离或负例分析 |

主推进池建议：`ADV2_SRCLOW_R17_E260`、`ADV2_FUSE5_R20_E260`、`ADV2_R28_FUSE6_E260`。机制诊断池建议：`ADV2_R28_PROXYLOW_E260`、`ADV2_R17_CORESTRICT_E260`、`ADV2_T13_CONSERVE_E260`、`ADV2_SOURCECAP32_R20_E260`、`ADV2_FUSE6_R17_E260`、`ADV2_TAILCV_R17_E260`、`ADV2_TAILCV_R20_E260`。`ADV2_R20_SAT70_E260`和`ADV2_T13_TAILGUARD_E260`因receiver floor不足，不应直接推进。高风险或负例候选按`source_overflow`、receiver floor不足和proxy_vaccept未改善处理。

## Next Experiment Matrix

| experiment_group | candidates | mechanism | metrics | success_criteria |
| --- | --- | --- | --- | --- |
| A:不重训fusion+local hard gate dry-run | SRCLOW_R17,FUSE5_R20,R28_FUSE6,R17_CORESTRICT,T13_CONSERVE | local component distance+density+NLL+geo margin+energy gate | known_core_accept,known_tail_review,proxy_vaccept,shell/inter/bridge accept,reject_reason_counts | proxy_vaccept显著低于1；shell/inter/bridge accept<0.05；known_core_accept>=0.90；tail不自动接收 |
| B:negative space filling训练 | SRCLOW_R17/FUSE5_R20主线+R28机制候选 | shell negative,tail-outward negative,inter-class slerp negative,same-class bridge negative | proxy_vaccept,proxy_vac_rate,shell_accept,inter_slerp_accept,source_overflow,closed metrics | unknown-risk accept下降且closed-set不崩 |
| C:core/tail/outside quarantine | SRCLOW_R17,FUSE5_R20,TAILCV_R17 | core强CE,soft tail低权重,extreme tail quarantine,p99/CVaR约束,tail_auto_accept=false | p99,CVaR95/97,source_overflow,known_core_accept,known_tail_review,proxy_vaccept | p99和overflow下降，不只降p95 |
| D:unlabeled unknown-risk mining | SRCLOW_R17,R17_CORESTRICT,R20_VACMID,PROXYHI | unlabeled分pseudo-known core/unknown-risk buffer/ignore | unl_pseudo_core_count,unl_risk_count,risk_energy_out_loss,proxy_vaccept | 低density/低margin高softmax样本不再污染known compactness |
| E:source episode safe gate | SRCLOW_R17,SOURCECAP32_R20,FUSE5_R20 | source query仅core+density pass才拉近，tail/outside进入uncertain/risk | source_ep_known_query_frac,source_ep_uncertain_query_frac,source_overflow,closed metrics | source_overflow下降且closed-set不大幅下降 |
| F:真实Stage2-A/C unknown评估 | SRCLOW_R17,FUSE5_R20,R28_FUSE6+T13/R17对照 | 使用真实Y_unknown query和Stage2 support/query权限 | unknown_FAR,FPR95,AUROC_energy,old_acc,seen_new_acc,H_old_new,reject_reason_counts | unknown_FAR<=0.05且old/seen-new可接受时才谈拒识达标 |

## Final Verdict

ADV2是一次有效的Phase1 source-only实验：14/14完成、14/14导出、fatal为0、fusion字段真实落地到JSON。它相对lateopt有闭集总体提升，但相对vacuum32有效候选和父候选闭集均值回退；它真正推进的是fusion可审计、p95略降和source overflow均值下降，而不是unknown拒识成功。

但是ADV2不能作为unknown拒识成功证据。`proxy_vaccept≈1`、`proxy_auc≈0.56-0.57`、p99长尾和source episode overflow共同说明accept-domain还没有真正收紧。下一步必须先做fusion/local component hard gate dry-run和真实Stage2 unknown评估，再考虑negative-space filling、tail quarantine和source episode safe gate重训。
