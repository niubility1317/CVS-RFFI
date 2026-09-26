# 实验总索引

更新：2026-09-26T18:14:15+00:00

先按方法/问题检索，再用run ID读取精确配置与证据。历史组数量不等于独立实验数量。

- [管理规范与常用命令](../docs/EXPERIMENT_MANAGEMENT.md)
- [完整目录CSV](catalog.csv) · [结构化目录](catalog.jsonl) · [覆盖与缺项](coverage.json)
- 通过show的`--section artifacts/facts`读取该条目的路径或配置出处；细目按记录压缩保存于details/，不扫描全库。

## 登记规模

|记录类型|数量|
|---|---:|
|managed_run|8|
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
|[adv3b02](by_method/adv3b02.md)|1462|
|[cvs](by_method/cvs.md)|1214|
|[bex02](by_method/bex02.md)|1132|
|[satellite_ablation](by_method/satellite_ablation.md)|1109|
|[scheduler](by_method/scheduler.md)|1098|
|[federated](by_method/federated.md)|742|
|[phase1](by_method/phase1.md)|665|
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
|[core90](by_method/core90.md)|205|
|[fasttrust](by_method/fasttrust.md)|139|
|[shot](by_method/shot.md)|113|
|[source](by_method/source.md)|89|
|[smoke](by_method/smoke.md)|70|
|[repair](by_method/repair.md)|68|
|[nm fdu](by_method/nm_fdu.md)|49|
|[ecrs](by_method/ecrs.md)|45|
|[response](by_method/response.md)|42|
|[mopc](by_method/mopc.md)|30|
|[split_bex02](by_method/split_bex02.md)|30|
|[csil](by_method/csil.md)|29|
|[protonet](by_method/protonet.md)|29|
|[daot](by_method/daot.md)|25|
|[comparison](by_method/comparison.md)|24|
|[hcfdg](by_method/hcfdg.md)|20|
|[dryrun](by_method/dryrun.md)|19|
|[diagnostic](by_method/diagnostic.md)|13|
|[fedcvs](by_method/fedcvs.md)|9|
|[STAR](by_method/STAR.md)|5|
|[Loo](by_method/Loo.md)|5|
|[ordinary-pseudo-label](by_method/ordinary-pseudo-label.md)|5|
|[stop](by_method/stop.md)|5|
|[CVCNN](by_method/CVCNN.md)|4|
|[RIEI](by_method/RIEI.md)|4|
|[rafl](by_method/rafl.md)|4|
|[residual_noeq](by_method/residual_noeq.md)|3|
|[phase3](by_method/phase3.md)|3|
|[practical](by_method/practical.md)|2|
|[poster](by_method/poster.md)|2|
|[radionet](by_method/radionet.md)|2|
|[fucl](by_method/fucl.md)|2|
|[tifs2025](by_method/tifs2025.md)|2|
|[PL100](by_method/PL100.md)|1|
|[warm10](by_method/warm10.md)|1|
|[no-unlabeled](by_method/no-unlabeled.md)|1|
|[baseline](by_method/baseline.md)|1|
|[source_only](by_method/source_only.md)|1|
|[data-builder](by_method/data-builder.md)|1|
|[fedfa](by_method/fedfa.md)|1|

## 最近记录入口

|名称|类型|报告/原目录|
|---|---|---|
|指定9月17日CVCNN版本重跑|managed_run|[打开](../automation_reports/CV-SincNet/20260920-cvcnn-original-pl100-s299-e200-r02/report.md)|
|原RIEI优化算法+普通固定星地拼接+硬伪标签|managed_run|[打开](../automation_reports/CV-SincNet/20260920-riei-original-concat-pl-s299-e200-r03/report.md)|
|RIEI简单暖启动：100轮、有/无伪标签对照|managed_run|[打开](../automation_reports/CV-SincNet/20260920-riei-warm10-pl20-e100-s299-r04/report.md)|
|外部对比方法：统一残差信道、五模型种子源域训练|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase1-baselines-practical-manysig-m5-r01/report.md)|
|Phase2外部对比：固定DG、support NCM及POSTER/RadioNet原生微调|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase2-baselines-practical-manytx-m5-r01/report.md)|
|Phase2残差信道共享数据：7RX、6旧类+20新类|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase2-practical-data-manytx-s2026092705-r01/report.md)|
|STAR同配置CVCNN/RIEI普通伪标签对照|managed_run|[打开](../automation_reports/CV-SincNet/cvcnn_riei_pl_loo_s299_t90_20260920_r01/report.md)|
|RIEI优化版原系数+等权星地CE修正版|managed_run|[打开](../automation_reports/CV-SincNet/riei_pl_loo_ce11_s299_t90_20260920_r02/report.md)|
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
|qKNN量化记忆、目标域适应与新类注册研究报告|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/QKNN_INT8_PROTOTYPE_RESEARCH_REPORT_20260720.md)|
|CVS项目场景与数据协议|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PROJECT_PROTOCOL.md)|
|Phase2近期改动联合实验验证计划|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE2_RECENT_CHANGES_INTEGRATED_VALIDATION_PLAN.md)|
|Phase2开放世界特征空间优化实验验证设计|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE2_OW_FEATURE_EXPERIMENT_VALIDATION_PLAN.md)|
|Phase2 LEO_weak-only数据可达性变更追踪|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE2_LEO_WEAK_ONLY_TRACEABILITY_20260715.md)|
|Phase2本地适配矩阵|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE2_LOCAL_FIT_MATRIX.md)|
|Phase2本地实现计划|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE2_IMPLEMENTATION_PLAN_CODEX.md)|
|Phase2地面特征空间优化设计与实现|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE2_FEATURE_SPACE_OPTIMIZATION.md)|
|Phase2数据一次验证实现附录|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE2_DATA_VALIDATION_APPENDIX.md)|
|PHASE1 SHORT195_S3 z_id特征空间验证计划|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE1_SHORT195S3_ZID_FEATURE_SPACE_VALIDATION.md)|
|Phase1半监督Baseline固定协议（2026-07-13）|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE1_SSL_BASELINE_PROTOCOL_20260713.md)|
|Phase1 ADVB02方法核对版：网络、损失、训练配置与星地信道增强|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/PHASE1_ADVB02_METHOD_RECHECK_20260819.md)|
|Local Workspace Cleanup 2026-06-26|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/LOCAL_CLEANUP_20260626.md)|
|地面训练双路线|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/GROUND_TRAINING.md)|
|ERBT-IDR M2.3／D92 E1-RFGuard实现追踪|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/ERBT_IDR_M23_RFGUARD_TRACE_20260820.md)|
|在轨部署Phase|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/DEPLOYMENT_PHASES.md)|
|D92 E0完整技术报告：identity160＋FFT96的256维注册方法|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/D92_METHOD_COMPLETE_REPORT_20260727.md)|
|D92 E0技术报告替换追踪表|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/D92_E0_REPORT_REPLACEMENT_TRACE_20260817.md)|
|D92_E0_METHOD_COMPLETE_REPORT_20260727|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/D92_E0_METHOD_COMPLETE_REPORT_20260727.html)|
|D92 E0全量消融实验数据汇总与详细分析|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/D92_E0_ALL_ABLATION_EXPERIMENTS_REPORT_20260819.md)|
|CVS Stage2-C TCSR-CI算法说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_TCSR_CI_ALGORITHM_20260704.md)|
|CVS Stage2-C RMD-CI算法说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_RMD_CI_ALGORITHM_20260704.md)|
|CVS Stage2-C OPC-MECR协同拒识算法说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_OPC_MECR_ALGORITHM_20260704.md)|
|CVS Stage2-C OSPR-CI Algorithm|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_OSPR_CI_ALGORITHM_20260705.md)|
|OF-HNFR-CI:旧类 floor 约束的硬负样本协同推理诊断|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_OF_HNFR_CI_ALGORITHM_20260704.md)|
|CVS Stage2-C KERA-CI算法说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_KERA_CI_ALGORITHM_20260704.md)|
|CVS Stage2-C DMG-CI算法设计|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_DMG_CI_ALGORITHM_20260704.md)|
|CVS Stage2-C CRISP-C协同推理算法设计|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_CRISP_C_ALGORITHM_20260704.md)|
|CVS Stage2-C C3R-HNFR算法设计|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_C3R_HNFR_ALGORITHM_20260704.md)|
|CVS Stage2-C APACE-CI算法说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_APACE_CI_ALGORITHM_20260704.md)|
|CVS Stage2-C AOR-Adapter-CI算法说明|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_STAGE2C_AOR_ADAPTER_CI_ALGORITHM_20260704.md)|
|CVS论文级对比实验协议|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md)|
|CVS项目相关约定与数据协议|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_PROJECT_CONVENTIONS_AND_DATA_PROTOCOL_20260715.md)|
|Phase1 FastTrust方法设计、实现与实验结果报告|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_PHASE1_FASTTRUST_METHOD_REPORT_20260827.md)|
|Phase1挑战条件化PA算子辨识需求追踪表|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_PHASE1_CCOI_PA_V1_TRACE_20260824.md)|
|从内容控制到条件系统辨识：Phase1挑战条件化PA算子辨识详细结合设计|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_PHASE1_CCOI_PA_V1_DESIGN_20260824.md)|
|ADV3B02-NMFDU-GATE-V1设计追踪表|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_PHASE1_ADV3B02_NMFDU_GATE_V1_TRACE_20260901.md)|
|ADV3B02物理因子化交叉重构设计追踪表|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md)|
|ADV3B02-ECRS-V1设计追溯表|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_PHASE1_ADV3B02_ECRS_V1_TRACE_20260901.md)|
|CVS_META_ADAPTER_TRI_R4_V1设计追踪表|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md)|
|CVS_META_ADAPTER_TRI_R4_V1设计规格|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_META_ADAPTER_TRI_R4_V1_DESIGN_20260824.md)|
|CVS阶段性成果技术报告：`ADV3B02_CORE90_SOFT_E200`与`qKNNV42`|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS_ADV3B02_QKNNV42_TECHNICAL_REPORT_20260709.md)|
|CVS-RFFI_Phase2详细复现报告_图表与场景优化版_修复版|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS-RFFI_Phase2详细复现报告_图表与场景优化版_修复版.docx)|
|CVS-RFFI_Phase2详细复现报告_图表与场景优化版|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CVS-RFFI_Phase2详细复现报告_图表与场景优化版.docx)|
|持续学习、开放世界学习与CVS-RFFI持续注册：联网调研与项目映射报告|legacy_evidence_group|[打开](../github_publish/rffi-labeled-da-three-20260915/docs/CONTINUAL_LEARNING_OPEN_WORLD_REGISTRATION_RESEARCH_20260809.md)|
