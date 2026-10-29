# G10 原 F0 预测身份条件局部支持

Run `20261030-phase3-star-conditional-support-s2026092701-r01`，仅源端单种子证伪。用户 v3 指标保持跨未见 RX 星地 C_K/R_U≥90%、A_K 距同输入原 F0 闭集下降≤2pp；主报 true-TX 宏 A_K/R_U/H。完整 Phase3 目标继续 ACTIVE。前轮局部支持主方法已 NO_PROMOTION；本轮不复用其失败主规则或按 novel 调 k。

## 唯一差异与数据

原 F0 E50/11100/default6head/232993 及 DURA v1.0 保持冻结。本轮复用 `20261030-phase3-star-local-support-s2026092701-r01` 的 original_F0/predict：bank L6300/OE6134、known V27000、novel3150 的单位 fused160 和原默认头 logits。前轮 code `84cdc012f7d9641b283c407f6048bfe4e6e21cf6`。bank 来自 jointOVA r02 的原 F0 teacher，而非更新后的 candidate。真实输入祖先全部同 G10 MIX01 P1；本轮无新增真实 IQ 值消费或神经前向，只做实际原 F0 checkpoint 的2条 synthetic G10 P0 无 query smoke。

旧六类按 `14-10、14-7、20-15、20-19、6-15、8-20`，L 各1050；V 各4500。source RX1/3/4/6/8及2021-03-08/15/23三日已见；novel19-15/19-16/7-1各1050已有历史源开发曝光，非新 RX/TX 独立盲测。只新增读取已合法 source L 的 y 元数据6300条，并按相同 L IDs 对齐，不能读 OE TX/K 或 target 标签。原 U_s56700、目标U/Q及正式RX24-18均不消费。

固定 k=8、eps=1e-6、FP64逐行再单位化、距离clip(2−2cos,0,4)，沿用前轮 core 的固定 einsum/最小8值排序均值，保证逐行/分块/支持库置换语义一致。两个方法顺序固定为：

1. `pooled_relative`（匹配控制）：log(d_OE+eps)−log(d_L+eps)，两bank整体取8邻居。原始分数、原六τ和接收集合精确复现前轮控制，不重新拟合阈值。
2. **`predicted_identity_relative`（唯一主）**：query 的 c=原F0 default logits argmax，d_Lc只在 source L 真实旧类 y=c 的子bank内取8邻居，d_OE仍是全部OE的8邻居；分数 log(d_OE+eps)−log(d_Lc+eps)。不接收 query 真 TX/RX/场景、不按近邻替换身份、不更新bank或估计query群体统计。

主方法用既有 `star_class_risk.calibrate_known`，仅 V 标签和同输入原F0 baseline，在42组 true-known 宏 C_K≥90%、closed−A_K≤2pp的 complete-tie约束下校准六个预测类τ。支持库在 V 前冻结，主六τ和控制六τ在 novel 特征打开前全部固定。known baseline/freshclosed均复制父原F0，不用candidate自身closed重设。所有prediction完成后独立CPU scorer才连接原source novel真值。只有固定主known42及unknown总体/五RX/六scene12组均过90%才有资格研究无源样本部署导出；否则NO_PROMOTION，不能换主/调k/扩seed。协议禁止部署侧回读逐样本source特征，因此本轮全bank仅作源端机制诊断，不是可部署状态，也不能直接推进target。通过后须另记只含冻结聚合锚点或参数的source-free候选；这不是本轮新增实现或额外发布门槛。

## 实现合同与分工

- `star_conditional_support_core.py`：常量METHODS/K/EPS/BLOCK128。`fit(L_features,L_y,OE_features)->state` exact keys L/L_y/OE，L/OE为FP64单位工作特征，L_y为int64且六类各至少8条。`score(query_features,closed,state)->FP64[N,2]`，closed是原默认头的预测类，不是真标签；无RX/scene参数。复用前轮固定数值距离归约，不修改旧核心。测试手算条件距离、控制一致、默认预测路由、完整ties、行独立/置换、bank不变及非法输入。
- `star_conditional_support_run.py`：exact7配置 run_id/code_commit/source_config/parent_prediction_root/cache_root/output_root/batch_size128；source_config沿runner.COMMON加source_cache，row_id=`star-conditional-support-s2026092701`，其他原F0与源契约不变。可复用local_support_run的_backend/_smoke/_snapshot/_check_models/_identity/_same/_write/_npz，禁止调用其run/role IQ加载。顺序为actualcheckpoint smoke→父marker/bank+cacheL ids/metadata y→冻结bank→父known单位/logits→固定分数与控制精确复现→仅主τ校准→12τ/cal/known写全→父novel单位/logits打开→固定分数→rootcomplete。父控制raw与margin容差1e-12，控制τ逐值相等；不重复父三方法评分或数据重验。
- `star_conditional_support_score.py`：prediction complete和全部公有来源/ID/分数/τ/defaultargmax/父控制复现检验后才能读私有novel truth。固定新核心重建两个分数，不重拟合τ；sourceL y与当前cache元数据和bankIDs对齐，OE数据不含TX/K。沿用common._masks/_goals/metrics/CI，主报两方法overall、42组、逐TX/RX/scene、paired、macroAUROC/OSCR、成本；A_N/H_OW=N/A，fullgoal ACTIVE。
- `star_conditional_support_launch.py`：沿单row local launcher 12fields runtime，rows只original_F0、max_concurrent_rows1，minfree4096MB，每GPU最多2新增，foreign不动；source→独立CPU score→ANALYZED_SOURCE_ONLY，来源/非有限/冲突/技术失败保留partial，不因性能停止或重跑。

导出RUN_ID/SCHEMA=`pamf_phase3_star_conditional_support_prediction_v1`/STATUS=`STAR_CONDITIONAL_SUPPORT_PREDICTIONS_COMPLETE`/CALIBRATION_STATUS=`STAR_CONDITIONAL_SUPPORT_THRESHOLDS_FROZEN`/METHODS/PRIMARY_METHOD/COUNTS/KNOWN_FIELDS/NOVEL_FIELDS/BANK_FIELDS。

KNOWN_FIELDS exact沿前轮 ids/y/rx_i/day_i/scenario/unit_features/default_logits/closed/baseline_closed/raw_scores/scores，但两个score矩阵均FP64[N,2]；NOVEL_FIELDS删y/scenario/baseline_closed。所有identity单位/logits/IDs/meta复制父结果；unit FP32[N160]、logits FP32[N6]。BANK_FIELDS为前轮六fields L_ids/L_unit_features/L_rx_i/OE_ids/OE_unit_features/OE_rx_i再加L_y，Rx只用于来源/报告，不进新评分。thresholds.npz exact conditional FP64[2,6]。

cal complete含schema/status/run/code、method_names/conditional_thresholds/methods（每项known42报告），group_count42、primary_threshold_fit_known_only=true、control_thresholds_reused=true、source_only/target_reads0/source_L_TX_labels_used=true/source_OE_TX_or_K_used=false。root complete含同身份/来源、parent_run_id/code_commit/parent_prediction_root/cache_root、teacher_run_id/code_commit、cache_identities、model_identity/dura_identity、计数known27000/novel3150/bankL6300/OE6134、source_only/target_reads0/source_novel_truth_reads0/source_U_reads0、F_forward_rows2/synthetic_P0_forward_rows2/G_forward_rows0/new_V_forward_rows0/new_novel_forward_rows0/source_L_forward_rows0/source_OE_forward_rows0/neural_updates0/G_updates0/real_IQ_values_read0/source_L_metadata_labels_read6300、classification_pathP1、parent_public_features_reused=true、control_scores_thresholds_acceptance_reproduced=true、source_bank_frozen_before_V/source_bank_unchanged/model_parameters_buffers_unchanged/all_thresholds_frozen_before_novel全true、CPUreadout_seconds/rows30150、fresh_closed_vs_cached_baseline_changed_count0、final_phase3_complete=false/A_N/H_OW=null。

根输出为source_bank.npz/known_predictions.npz/novel_predictions.npz/thresholds.npz/smoke.json/complete.json及calibration/complete.json。评分status=`STAR_CONDITIONAL_SUPPORT_SCORES_COMPLETE`，字段source_gate_passed/arm_source_gate_passed、primary_summary/primary_goals和next_step（SOURCE_CONDITIONAL_SUPPORT_READY_FOR_SOURCE_FREE_EXPORT或NO_PROMOTION）供launcher实际消费。eligible_primary固定false，source_free_export_required=true、source_bank_is_deployment_state=false；source gate通过只支持源端机制，不代表目标准入。launcher同样保留target eligible=false。

## 执行与成本边界

默认N607、一个source predictor、一个随后CPU scorer，新immutable root；只校验父已完成合法句柄和sourceL标签对齐，不重新读取真实IQ或生成观测。固定首步实际checkpoint smoke PASS立即继续、一次聚焦协议负测/独立P0P1、实际code/config commit、最小登记、单preflight/archive SHA/remotecompile、实时PID/CWD/cmd/GPU/log及完整预测后独立评分。无额外封印、重复审查或报告审批。

无神经/G更新、无新真实IQ/F/G，额外F2仅smoke。NN部署参数F232993+G266951=499944、无aux；bank原FP32特征7957760B/工作FP64特征15915520B，另有L_y50400B、RX/IDs/索引/距离workspace。生产CPU核心score耗时和独立复核分别报告，不冒充B1端到端时延。父F30152与历史P1恢复成本单列祖先，不把复用称零成本。
