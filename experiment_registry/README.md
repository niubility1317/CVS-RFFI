# 实验总索引

更新：2026-09-26T18:28:33+00:00

先按方法/问题检索，再用run ID读取精确配置与证据。历史组数量不等于独立实验数量。

- [管理规范与常用命令](../docs/EXPERIMENT_MANAGEMENT.md)
- [完整目录CSV](catalog.csv) · [结构化目录](catalog.jsonl) · [覆盖与缺项](coverage.json)
- 通过show的`--section artifacts/facts`读取该条目的路径或配置出处；细目按记录压缩保存于details/，不扫描全库。

## 登记规模

|记录类型|数量|
|---|---:|
|managed_run|53|
|legacy_evidence_group|4028|
|legacy_log_record|7778|
|remote_directory_locator|3608|

历史状态统一为HISTORICAL_UNVERIFIED；RUNNING等原文声明只供查证，不能证明此刻仍在运行。

## 按方法与用途查找

|路径标签（定位提示）|证据组数|
|---|---:|
|[centralized](by_method/centralized.md)|4168|
|[other](by_method/other.md)|2245|
|[comparison_baseline](by_method/comparison_baseline.md)|1784|
|[adv3b02](by_method/adv3b02.md)|1463|
|[cvs](by_method/cvs.md)|1214|
|[bex02](by_method/bex02.md)|1132|
|[satellite_ablation](by_method/satellite_ablation.md)|1109|
|[scheduler](by_method/scheduler.md)|1098|
|[federated](by_method/federated.md)|742|
|[phase1](by_method/phase1.md)|671|
|[riei](by_method/riei.md)|626|
|[qknn](by_method/qknn.md)|624|
|[drift](by_method/drift.md)|616|
|[stage2](by_method/stage2.md)|600|
|[fsdg](by_method/fsdg.md)|540|
|[cvcnn](by_method/cvcnn.md)|479|
|[phase2](by_method/phase2.md)|466|
|[receiver_agnostic](by_method/receiver_agnostic.md)|466|
|[d92](by_method/d92.md)|463|
|[fl82](by_method/fl82.md)|376|
|[fedcvs/vmb](by_method/fedcvs_vmb.md)|344|
|[bex](by_method/bex.md)|237|
|[core90](by_method/core90.md)|208|
|[fasttrust](by_method/fasttrust.md)|145|
|[shot](by_method/shot.md)|113|
|[source](by_method/source.md)|89|
|[smoke](by_method/smoke.md)|70|
|[repair](by_method/repair.md)|68|
|[nm fdu](by_method/nm_fdu.md)|49|
|[response](by_method/response.md)|45|
|[ecrs](by_method/ecrs.md)|45|
|[STAR](by_method/STAR.md)|37|
|[daot](by_method/daot.md)|35|
|[Loo](by_method/Loo.md)|31|
|[mopc](by_method/mopc.md)|30|
|[split_bex02](by_method/split_bex02.md)|30|
|[csil](by_method/csil.md)|29|
|[protonet](by_method/protonet.md)|29|
|[comparison](by_method/comparison.md)|24|
|[RX1](by_method/RX1.md)|24|
|[Smooth-E](by_method/Smooth-E.md)|22|
|[hcfdg](by_method/hcfdg.md)|20|
|[dryrun](by_method/dryrun.md)|19|
|[plain-pseudo-label](by_method/plain-pseudo-label.md)|13|
|[diagnostic](by_method/diagnostic.md)|13|
|[rc4](by_method/rc4.md)|10|
|[practical](by_method/practical.md)|10|
|[fedcvs](by_method/fedcvs.md)|9|
|[residual](by_method/residual.md)|7|
|[special-SNN](by_method/special-SNN.md)|7|
|[static-Loo](by_method/static-Loo.md)|7|
|[SG-only](by_method/SG-only.md)|7|
|[concat](by_method/concat.md)|6|
|[full](by_method/full.md)|5|
|[zf](by_method/zf.md)|5|
|[mmse](by_method/mmse.md)|5|
|[ordinary-pseudo-label](by_method/ordinary-pseudo-label.md)|5|
|[stop](by_method/stop.md)|5|
|[CVCNN](by_method/CVCNN.md)|4|
|[RIEI](by_method/RIEI.md)|4|
|[residual_noeq](by_method/residual_noeq.md)|4|
|[rafl](by_method/rafl.md)|4|
|[poster](by_method/poster.md)|3|
|[radionet](by_method/radionet.md)|3|
|[phase3](by_method/phase3.md)|3|
|[sixscene](by_method/sixscene.md)|2|
|[evaluation](by_method/evaluation.md)|2|
|[baseline](by_method/baseline.md)|2|
|[source_only](by_method/source_only.md)|2|
|[three_seed](by_method/three_seed.md)|2|
|[E](by_method/E.md)|2|
|[fucl](by_method/fucl.md)|2|
|[tifs2025](by_method/tifs2025.md)|2|
|[original_leo](by_method/original_leo.md)|1|
|[PL100](by_method/PL100.md)|1|
|[ratio](by_method/ratio.md)|1|
|[warm10](by_method/warm10.md)|1|
|[no-unlabeled](by_method/no-unlabeled.md)|1|
|[data-builder](by_method/data-builder.md)|1|
|[STAR-comparison](by_method/STAR-comparison.md)|1|
|[provided-code](by_method/provided-code.md)|1|
|[Original-PairSNN](by_method/Original-PairSNN.md)|1|
|[ablation](by_method/ablation.md)|1|
|[no-SNN](by_method/no-SNN.md)|1|
|[threshold](by_method/threshold.md)|1|
|[fedfa](by_method/fedfa.md)|1|

## 最近记录入口

|名称|类型|报告/原目录|
|---|---|---|
|ADV3B02＋DAOT＋FastTrust-RC4原LEO拼接增强|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-original-leo-manysig-s392005-r01/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r01/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r02/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验r03|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r03/report.md)|
|practical四组最新保存权重测试|managed_run|[打开](../automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r01/report.md)|
|practical四组最新保存权重测试r02|managed_run|[打开](../automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r02/report.md)|
|指定9月17日CVCNN版本重跑|managed_run|[打开](../automation_reports/CV-SincNet/20260920-cvcnn-original-pl100-s299-e200-r02/report.md)|
|DAOT＋FastTrust-RC4 residual环境比例六组|managed_run|[打开](../automation_reports/CV-SincNet/20260920-phase1-daot-rc4-residual-ratios-manysig-s392005-r01/report.md)|
|旧Practical四组checkpoint统一六环境测试|managed_run|[打开](../automation_reports/CV-SincNet/20260920-phase1-practical4-sixscene-eval-s392005-r01/report.md)|
|原RIEI优化算法+普通固定星地拼接+硬伪标签|managed_run|[打开](../automation_reports/CV-SincNet/20260920-riei-original-concat-pl-s299-e200-r03/report.md)|
|RIEI简单暖启动：100轮、有/无伪标签对照|managed_run|[打开](../automation_reports/CV-SincNet/20260920-riei-warm10-pl20-e100-s299-r04/report.md)|
|最后一轮权重：clean与三种residual_noeq目标域评估|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase1-baselines-final-clean-satellite-m5-r01/report.md)|
|外部对比方法：统一残差信道、五模型种子源域训练|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase1-baselines-practical-manysig-m5-r01/report.md)|
|Phase2外部对比：固定DG、support NCM及POSTER/RadioNet原生微调|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase2-baselines-practical-manytx-m5-r01/report.md)|
|Phase2残差信道共享数据：7RX、6旧类+20新类|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase2-practical-data-manytx-s2026092705-r01/report.md)|
|提供代码CORAL/MMMD/CNN/RIEI及PL六方法十种子|managed_run|[打开](../automation_reports/CV-SincNet/baselines_rx1_s10_e100_20260921_r01/report.md)|
|STAR同配置CVCNN/RIEI普通伪标签对照|managed_run|[打开](../automation_reports/CV-SincNet/cvcnn_riei_pl_loo_s299_t90_20260920_r01/report.md)|
|旧PairSNN：RX1九种子八阈值|managed_run|[打开](../automation_reports/CV-SincNet/pairsnn_rx1_s9_t8_e100_20260922_r01/report.md)|
|原生DAOT＋RC4与仅动力博弈三seed对照|managed_run|[打开](../automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r1/report.md)|
|原生DAOT＋RC4与仅动力博弈三seed对照|managed_run|[打开](../automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r2/report.md)|
|响应博弈矩阵测试评估：VERIFIED|managed_run|[打开](../automation_reports/CV-SincNet/response_matrix_eval_20260917/report.md)|
|RIEI优化版原系数+等权星地CE修正版|managed_run|[打开](../automation_reports/CV-SincNet/riei_pl_loo_ce11_s299_t90_20260920_r02/report.md)|
|真实标签CE下限0.05＋梯度裁剪10：八种子六阈值|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_anchor05_pl9_rx1_s8_t6_e100_20260924_r01/report.md)|
|源校准严格伪标签：3seed × 6阈值|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_calibrated_plainpl_rx1_s3_t6_e100_20260924_r01/report.md)|
|冻结E5教师：探索性比较，不是同方法阈值敏感性|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_frozen5_pl9_rx1_s8_t5_e100_20260925_r01/report.md)|
|冻结门控自伪标签：统一 CE 下限 0.10|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_frozengate_anchor10_rx1_s8_t5_e100_20260925_r01/report.md)|
|冻结门控自伪标签：有标签CE下限0.20|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_frozengate_anchor20_rx1_s8_t5_e100_20260925_r01/report.md)|
|最终状态：ANALYZED，40/40完成，期望趋势未达到|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_frozengate_selfpl9_rx1_s8_t5_e100_20260925_r01/report.md)|
|类别支持联合置信度，统一 CE 下限 0.10|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_jointconf_anchor10_rx1_s8_t5_e100_20260926_r01/report.md)|
|Smooth E STAR＋普通置信度伪标签|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_plainpl_rx1_s299_t8_e100_20260923_r01/report.md)|
|Smooth E普通伪标签移除延迟渐增与SG投影：RX1单种子八阈值|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_plainpl_unprotected_rx1_s299_t8_e100_20260923_r01/report.md)|
|Smooth E unprotected RX1新增四种子六阈值|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_plainpl_unprotected_rx1_s4_t6_e100_20260923_r01/report.md)|
|无校准＋弱真实标签锚定：PL9压力实验|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_weakanchor_pl9_rx1_s3_t6_e100_20260924_r01/report.md)|
|弱真实标签PL9：新增八种子六阈值|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_weakanchor_pl9_rx1_s8_t6_e100_20260924_r01/report.md)|
|零有标签后期监督＋无梯度裁剪PL9压力实验|managed_run|[打开](../automation_reports/CV-SincNet/smooth_e_zeroanchor_noclip_pl9_rx1_s8_t6_e100_20260924_r01/report.md)|
|当前STAR E机制与配对消融|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_ablation_s10_t90_20260920_r01/report.md)|
|Smooth STAR E纯Loo去SNN配对对照|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_nosnn_loo_s10_t4_20260920_r01/report.md)|
|Comment3 RX1 w45单seed六阈值敏感性|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_comment3_s299_e100_20260921_r01/report.md)|
|Comment3 RX1 w45单seed六阈值敏感性|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_comment3_s299_e100_20260921_r02/report.md)|
|原Smooth E：原seed299加8种子的六阈值实验|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_comment3_s9_t6_e100_20260921_r01/report.md)|
|RX1伪标签噪声压力实验：取消渐增与监督增权|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_plstress_s299_e100_20260921_r01/report.md)|
|RX1 Smooth E SG增权与推理EMA精调9项|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_refine_s3_e100_20260921_r01/report.md)|
|Smooth STAR E RX1目标域三种子100轮|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_s3_e100_20260920_r01/report.md)|
|RX1严格阈值与提前伪标签实验|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_strict_early_s299_e100_20260921_r01/report.md)|
|RX1 Smooth E SG权重0.45十种子验证|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_w45_s10_e100_20260921_r01/report.md)|
|RX1 Smooth E SG伪标签权重两档三种子|managed_run|[打开](../automation_reports/CV-SincNet/smooth_star_e_rx1_wsg_s3_e100_20260920_r01/report.md)|
|SNN源域均衡和完整覆盖：2方案各24seed|managed_run|[打开](../automation_reports/CV-SincNet/star_snn_balanced_s24_e100_20260920_r01/report.md)|
|SNN与Loo增强捆绑：3方案3seed100轮|managed_run|[打开](../automation_reports/CV-SincNet/star_snn_loo_bridge_s3_e100_20260920_r01/report.md)|
|稳定SNN星地方案：3调度3seed100轮|managed_run|[打开](../automation_reports/CV-SincNet/star_snn_schedule_s3_e100_20260920_r01/report.md)|
|SNN星地分类权重二维搜索：8新配置3seed100轮|managed_run|[打开](../automation_reports/CV-SincNet/star_snn_sgweight_search_s3_e100_20260920_r01/report.md)|
|稳定SNN方案24种子验证：3复用21新增|managed_run|[打开](../automation_reports/CV-SincNet/star_snn_stable_s24_e100_20260920_r01/report.md)|
|旧STAR配对特别SNN：4候选+2对照，各10seed|managed_run|[打开](../automation_reports/CV-SincNet/star_special_snn_s10_t90_20260920_r01/report.md)|
|旧STAR配对特别SNN：4候选+2对照，各3seed，100轮|managed_run|[打开](../automation_reports/CV-SincNet/star_special_snn_s3_t90_e100_20260920_r01/report.md)|
|CVS项目场景与数据协议|legacy_evidence_group|[打开](../github_publish/CVS-RFFI-repo/docs/PROJECT_PROTOCOL.md)|
|实验管理与快速定位|legacy_evidence_group|[打开](../github_publish/CVS-RFFI-repo/docs/EXPERIMENT_MANAGEMENT.md)|
|N607操作|legacy_evidence_group|[打开](../github_publish/CVS-RFFI-repo/docs/workflows/n607.md)|
|实验管理与快速定位|legacy_evidence_group|[打开](../docs/EXPERIMENT_MANAGEMENT.md)|
|CVS项目场景与数据协议|legacy_evidence_group|[打开](../docs/PROJECT_PROTOCOL.md)|
|N607操作|legacy_evidence_group|[打开](../docs/workflows/n607.md)|
|已完成实验测试集完整数据|legacy_evidence_group|[打开](../automation_reports/CV-SincNet/response_matrix_eval_20260916/results/report.md)|
|response_matrix_eval_20260916_r1|legacy_evidence_group|[打开](../local_artifacts/response_matrix_eval_20260916_r1)|
|已完成实验测试集完整数据|legacy_evidence_group|[打开](../automation_reports/CV-SincNet/response_matrix_eval_20260915/results/report.md)|
|response_matrix_eval_20260915_r1|legacy_evidence_group|[打开](../local_artifacts/response_matrix_eval_20260915_r1)|
|POSTER与RadioNet：保留核心的服务器适配|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/reproduction/li_author_server_compat.md)|
|phase2_adv3b02_collab_open_set_qknn_full_20260703|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/remote_artifacts/phase2_adv3b02_collab_open_set_qknn_full_20260703)|
|scripts|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/scripts)|
|DADDA目录改名记录|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/reports/dadda_directory_rename_20260713.md)|
|Paper Original Matrix|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/paper_original_matrix.md)|
|正交空间约束FSCIL-SEI论文复现清单|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/orthogonal_incremental_sei/paper_checklist.md)|
|MoPC-HR Non-Exemplar CIL SEI|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/mopc_hr_non_exemplar_cil_sei/README.md)|
|mitigating_receiver_impact_da|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/mitigating_receiver_impact_da)|
|configs|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/configs)|
|adv3b02_official_newcount_scale_20260724_v7_release|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v7_release)|
|adv3b02_official_newcount_scale_20260724_v6_release|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v6_release)|
|adv3b02_official_newcount_scale_20260724_v2_release|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v2_release)|
|adv3b02_official_newcount_scale_20260724_v1_release|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v1_release)|
|Paper Reproduction Modules|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/README.md)|
|DADDA Paper-to-Code Audit Checklist|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/DADDA/paper_checklist.md)|
|CSIL Class-Incremental IoT Reproduction|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/paper_reproduction/CSIL/README.md)|
|receiver_agnostic_twostage_uda_n607_formal_dry_run_20260708|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/receiver_agnostic_twostage_uda_n607_formal_dry_run_20260708.json)|
|receiver_agnostic_twostage_uda_dry_run_20260708|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/receiver_agnostic_twostage_uda_dry_run_20260708.json)|
|qknnv42_strict_dual125_20260714_183556|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/qknnv42_strict_dual125_20260714_183556)|
|qknnv42_jg020_matched_stage2b_k10_20260716_summary|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/qknnv42_jg020_matched_stage2b_k10_20260716_summary)|
|qknnv42_full_nonoracle125_20260715_094615_summary|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/qknnv42_full_nonoracle125_20260715_094615_summary)|
|qknn_ground_effective8_v14_protocol_plan_repair6_20260715|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/qknn_ground_effective8_v14_protocol_plan_repair6_20260715)|
|phase2_r8_r9_r10_qknn8_collab_budget_20260706|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/phase2_r8_r9_r10_qknn8_collab_budget_20260706)|
|phase1_dgleo_p0factorial8_matrix_20260714|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/phase1_dgleo_p0factorial8_matrix_20260714.json)|
|mitigating_receiver_impact_da_dry_run_20260708|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/mitigating_receiver_impact_da_dry_run_20260708.json)|
|M5 support-only sparse key-layer delta|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/m5_support_only_sparse_delta_k10_new5_20260717/results.md)|
|D21 M6 support-fold低秩投影诊断|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/d21_m6_support_fold_lowrank/final/report.md)|
|M6独立query不可达审计|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/d21_m6_query_unreachable_audit/report.md)|
|D21 M5-lite极轻norm-affine诊断|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/d21_m5lite_norm_affine/final/report.md)|
|D21 M1b软旧状态融合开发结果|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/d21_m1b_soft_fusion/final/report.md)|
|D21 M1双状态竞争保护开发实验|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/d21_m1_dual_state/final/report.md)|
|D21 floor-aware极轻量数学描述符开发实验|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/d21_floor_explore/final4arm_smetric/report.md)|
|D21 K10/new5胶囊轻型适应与注册开发结果|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/d21_capsule_fast_adapt_dev_20260717/report.md)|
|ADV3B02、qKNN、域适应与类增量方法完整对比报告|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/adv3b02_full_comparison_20260714_191356/report.md)|
|adv3b02_direct_old_strict_20260714_181100|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/local_artifacts/adv3b02_direct_old_strict_20260714_181100)|
|CVS-RFFI/CV-SincNet项目介绍|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/项目介绍.md)|
|CVS-RFFI Phase1伪标签优化周报（2026-08-17—2026-08-23）|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN域适应与类增量结果补充版_截至20260731.docx)|
|EPOC-ADV3B02 Distill Implementation Plan|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/superpowers/plans/2026-07-05-epoc-adv3b02-distill.md)|
|Project Instructions|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/source_controls/AGENTS.full.md)|
|release_manifest_latest|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/release_manifest_latest.json)|
|release_manifest_20260701_003040|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/release_manifest_20260701_003040.json)|
|release_manifest_20260701_002915|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/release_manifest_20260701_002915.json)|
|release_manifest_20260701_002420|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/release_manifest_20260701_002420.json)|
|release_manifest_20260626_154938|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/release_manifest_20260626_154938.json)|
|qKNNV42技术汇报构建说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/qknnv42_report_20260714/report.html)|
|本地项目资产清理记录：L-D1、L-C1、L-C2|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/project_governance/local_asset_cleanup_20260823_ld1_lc1_lc2.md)|
|FastTrust-QB3有界域混淆与伪标签利用设计|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/plans/2026-08-24-fasttrust-qb3-bounded-confusion-design.md)|
|Stage2-C sparse key-layer delta protocol mirror|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/cvs_stage2c_sparse_key_layer_delta_20260715.md)|
|CVS Phase2 Stage2-B/Stage2-C极轻型快速适应目标模式|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/cvs_stage2c_extreme_light_goal_20260714.md)|
|TD-HTRC模块二改造实施追踪|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/TD_HTRC_MODULE2_IMPLEMENTATION_TRACE_20260819.md)|
|TD-HTRC M2.2模块二改造说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/TD_HTRC_M22_MODULE2_UPGRADE.md)|
|TD-HTRC-M2.1：模块二目标域共享传输—稳健中心升级说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/TD_HTRC_M21_MODULE2_UPGRADE.md)|
|CVS Stage2研发快照发布记录|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/STAGE2_RELEASE_20260722.md)|
|Stage2轻型域适应与新类注册研发目标|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/STAGE2_METHOD_RESEARCH_GOAL.md)|
|CVS GitHub发布快照|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/RELEASE_SNAPSHOT.md)|
|对比方法归档位置|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/REPRODUCTION_ARCHIVE.md)|
|CVS-only发布范围|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PUBLISH_SCOPE.md)|
