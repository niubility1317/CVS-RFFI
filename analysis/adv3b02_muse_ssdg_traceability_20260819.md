# ADV3B02-MUSE-SSDG需求追踪

设计权威：`docs/superpowers/specs/2026-08-19-adv3b02-muse-ssdg-design.md`

科学与数据协议权威：`E:/type10-7/项目.md`

Task 8审计日期：2026-08-20

## Final fix wave追踪（2026-08-20）

|ID|来源|规范化要求|实现目标|状态|验证|备注|
|---|---|---|---|---|---|---|
|FFR-1|final-fix brief|三路证据必须为全局头、source-domain局部头和由`z_id`打分的真实分类prototype；主调用链在routing前执行只由`L_s`统计得到的prior alignment|`code/cvsrffi/muse_ssdg.py`；`code/SSDG/train_ssdg.py`；MUSE单测|verified|`test_classification_prototype_probabilities_are_normalized_with_explicit_missing_classes`；`test_m2_fusion_uses_global_local_prototype_and_l_s_prior_alignment`|不读取`U_s` TX truth，不更新开集geometry|
|FFR-2|final-fix brief|schedule `proto_momentum`控制prototype EMA且与0.05–0.10未标注贡献分离；S3C冻结memory、statistics、U prototype和local teacher|MUSE核心、训练集成与训练头测试|verified|`test_proto_momentum_boundary_is_095_then_099_at_s3b_and_s3c`；`test_prototype_momentum_and_unlabeled_contribution_are_distinct_controls`；`test_epoch_181_freezes_muse_statistics_prior_and_local_teacher_state`；`test_frozen_local_teacher_survives_train_calls_and_optimizer_steps`|边界epoch 160/161/180/181与checkpoint回环均覆盖|
|FFR-3|final-fix brief|M3依稳定SHA mask逐样本从strong或satellite/nuisance视图中唯一选择identity logits和`z_id`，M1/M2不启用satellite identity|`code/SSDG/train_ssdg.py`；satellite/integration测试|verified|`test_m3_sha_mask_selects_exactly_one_identity_student_per_row`；`test_m1_m2_never_enable_satellite_identity_student`|配对nuisance前向仍可覆盖全batch|
|FFR-4|final-fix brief|MUSE launcher禁用训练入口内部target held-out final eval，只保留launcher一次canonical联合评测|`code/SSDG/train_ssdg.py`；MUSE launcher及行为测试|verified|`test_muse_can_delegate_final_target_eval_without_changing_legacy`；`test_fake_joint_evaluator_runs_once_and_writes_four_semantic_metrics_before_complete`|MUSE内部只写`external_final_eval_pending.json`，非MUSE默认行为不变|
|FFR-5|final-fix brief|正式评测器strict模式使用`strict=True`、禁止fallback；失败时不写metrics；launcher拒绝非严格metadata|`eval_ssdg_sat_per_rx.py`；MUSE launcher；strict/launcher测试|verified|`test_strict_checkpoint_loader_uses_torch_strict_true_and_reports_zero_mismatch`；`test_strict_reconstruction_failure_exits_before_metrics_are_written`；`test_launcher_rejects_non_strict_or_fallback_reconstruction_metadata`|非strict legacy evaluator行为保留|
|FFR-6|final-fix brief|两个活动Phase1 factory迁移到`final_only`并能被共享入口解析|`full_ablation_spec.py`；`phase1_ablation_factory.py`；相应测试|verified|`test_active_phase1_row_factories_emit_parser_valid_final_only_selection`；`test_active_ablation_configs_pass_shared_checkpoint_parser`|其他legacy路径不改|
|FFR-7|final-fix brief|追踪和正式报告引用新真实调用链证据，保持M0–M3真实训练/四场景性能未运行|traceability；正式run report；final-fix report|verified|`test_final_fix_traceability_names_real_call_chain_evidence`；`test_formal_report_keeps_runtime_performance_pending_after_final_fix`|不编造性能结果；发布OID由final-fix report独立回读记录|

## 审计口径

- 实现基线：`0e1019beb8f9c3217b4ae84f1a56a4be6dd5ba9e`；Tasks 1–7代码审计HEAD：`4c66489ea058f5fe8401c29a237a58708bd7451f`。
- `verified`表示需求已经由聚焦测试、协议负测、真实checkpoint smoke或artifact回读直接验证；`implemented`表示实现和聚焦测试已经闭合，但规范要求的真实训练期输出仍须由后续M0–M3运行生成；`pending`表示尚无实现或验证证据。
- final fix完整聚焦测试使用`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`运行16个MUSE、launcher、strict evaluator、Phase1协议、factory和文档测试文件，共收集175项；退出码0，175项全部通过。
- 真实checkpoint无query smoke使用`E:/type10-7/automation_reports/CV-SincNet/qknnv42_strict_dual125_20260714_183556/artifacts/best_joint_safe_ssdg.pth`，SHA-256为`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。严格重建结果为0 missing、0 unexpected；单batch验证真实global/local/prototype三头、`L_s` prior alignment、逐样本SHA identity选择、optimizer step和S3C strict状态回环；`query_input_count=0`、`target_truth_read_count=0`、`target_evaluation_count=0`。
- final fix smoke artifact：`E:/type10-7/local_artifacts/adv3b02_muse_ssdg_final_fix_20260820/m3_true_prototype_identity_strict_state_no_query_smoke.pt`，279,773字节，独立`weights_only=True`回读通过。稳定SHA mask选择2条strong与2条satellite；输入是确定性构造的source-shaped张量，不读取dataset、support、target query或truth；该smoke不形成准确率或性能证据。

## 正向追踪

|ID|来源|规范化要求|实现目标|状态|验证|证据边界|
|---|---|---|---|---|---|---|
|MUSE-001|Spec 1、2.1|以ADV3B02双表征、`160D z_id`、`z_dom`和部署接口为底座|`code/SSDG/train_ssdg.py`；`code/cvsrffi/muse_ssdg.py`|verified|真实ADV3B02 checkpoint严格重建；M3单batch前后向、optimizer step和state回环|smoke验证兼容性与可训练性，不是完整训练|
|MUSE-002|Spec 1；`项目.md` 4|固定`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`、物理ID互斥、source/target receiver不相交且零target进入|`_enforce_muse_source_protocol`；`_MUSEUnlabeledDatasetView`；launcher|implemented|`test_muse_enablement_resolves_and_validates_exact_four_role_source_protocol`；`test_muse_unlabeled_dataset_view_removes_tx_truth_before_collation`|当前没有实际loader receipt证明四角色物理ID互斥、`R_s∩R_t=∅`和target计数为0；synthetic smoke不构成该证据|
|MUSE-003|Spec 2.1、4|Epoch 1起让全部`U_s`参与domain/GRL/self/nuisance，S1无身份分类梯度|日程、epoch配对和M1训练路径|verified|日程5段测试；`test_muse_epoch_pairs_use_every_unlabeled_batch_and_cycle_labeled_batches`；S1参数化梯度负测|验证训练路径，不是200 epoch完成证据|
|MUSE-004|Spec 3、4|Epoch 17起启用EMA三证据融合，教师只读weak/source视图；routing前执行`L_s`全局/source-domain prior alignment|EMA调度；真实global/local/prototype融合和prior接线|verified|EMA边界测试；`test_m2_fusion_uses_global_local_prototype_and_l_s_prior_alignment`；`test_train_reaches_all_muse_integration_boundaries`|prior只由有标签`L_s`更新，未用target或`U_s` TX truth|
|MUSE-005|Spec 3|全局、source-domain局部和分类prototype按`0.50/0.25/0.25`几何融合|`geometric_fuse_probabilities`；`MUSEClassificationPrototypeBank.class_probabilities`；训练期局部头|verified|`test_classification_prototype_probabilities_are_normalized_with_explicit_missing_classes`；`test_m2_fusion_uses_global_local_prototype_and_l_s_prior_alignment`；finite/归一化回归|缺失类prototype概率显式为0；局部头仅训练期存在|
|MUSE-006|Spec 2.2、3|以置信、margin、JS、prototype距离和时间稳定性计算可靠度|`compute_muse_reliability`；训练路径|verified|五轴单调性、JS大于1、非有限输入和缺prototype证据测试|`_compute_muse_unlabeled_losses`不接受`y_u`|
|MUSE-007|Spec 3|将`U_s`互斥且完备地路由到`U_H/U_M/U_L`|`route_muse_reliability`；`MUSERoute`|verified|路由partition、阈值边界、M2/M3集成partition测试|空集合由加权损失零图处理|
|MUSE-008|Spec 3|`U_H`硬CE、`U_M`soft CE、`U_L`候选集或无身份梯度；按有效权重归一化|未标注loss和训练组合|verified|high/mid阶段行为、low candidate、masked mass归一化、空mask和零身份梯度测试|低置信路径不执行熵最小化|
|MUSE-009|Spec 2.2、3|候选累计质量至少0.75且最多3类；不可达时身份损失为0|`candidate_set_mask`；`candidate_set_cross_entropy`|verified|质量下限、3类上限、不可达和masked梯度测试；epoch 17/40/41边界测试|候选仅作用于`U_L`|
|MUSE-010|Spec 2.2|从`z_dom`回归SNR、CFO、相位噪声、K因子、时延和AGC等模拟扰动|`MUSETrainingHeads.nuisance_head`；paired simulator view|verified|shape/finite/empty-mask测试；M1 paired-view梯度与metadata配对测试|目标只来自模拟器metadata|
|MUSE-011|Spec 5|按`physical_sample_id`和epoch确定性SHA mask逐样本选择strong或单一satellite identity学生视图|`stable_sample_keys`；`select_satellite_student_mask`；identity logits/`z_id`逐行选择接线|verified|`test_m3_sha_mask_selects_exactly_one_identity_student_per_row`；`test_m1_m2_never_enable_satellite_identity_student`；稳定选择与metadata顺序测试|M3每行只消费一个identity分支；教师不读取satellite学生视图|
|MUSE-012|Spec 2.2、3|仅连续稳定至少3次的`U_H`以0.05–0.10贡献更新分类prototype；更新EMA受独立`proto_momentum`控制；S3C冻结|`MUSETemporalMemory`；`MUSEClassificationPrototypeBank`；epoch state配置|verified|`test_prototype_momentum_and_unlabeled_contribution_are_distinct_controls`；`test_proto_momentum_boundary_is_095_then_099_at_s3b_and_s3c`；`test_epoch_181_freezes_muse_statistics_prior_and_local_teacher_state`；state回环测试|`U_s`不能创建未由`L_s`初始化的新类；开放集prototype不接收`U_s`|
|MUSE-013|Spec 1、5；`项目.md` 4.2|`U_s`不得生成proxy unknown或更新开放集geometry|无TX真值数据视图；`_assert_muse_open_geometry_role`|verified|`test_muse_training_step_has_no_u_s_truth_or_in_loop_label_diagnostics`；`test_muse_rejects_any_u_s_open_geometry_update`|违规以固定`MUSE_PROTOCOL_U_S_OPEN_GEOMETRY_FORBIDDEN`失败|
|MUSE-014|Spec 6|实现并真实运行M0/M1/M2/M3单seed同协议矩阵，M0与M1同step预算|`launch_phase1_adv3b02_muse_ssdg_20260819.sh`；训练预算门控|implemented|launcher dry-run与fake行为测试；M0/M1 5-step等预算回归；能力单调测试|launcher实现已验证，但真实M0–M3矩阵未运行；取得四臂真实run artifact后才能升级为`verified`|
|MUSE-015|Spec 7；AGENTS.md|每个完成训练的候选必须由launcher唯一一次canonical评测clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|训练入口返回`DELEGATED_TO_MUSE_LAUNCHER`；一次strict联合真实评测调用和四份语义拆分|implemented|`test_muse_can_delegate_final_target_eval_without_changing_legacy`；launcher fake控制流验证一次联合调用、无内部target artifact、四场景完整性和逐场景失败状态|真实M0–M3尚未训练，因此没有真实四场景metrics|
|MUSE-016|Spec 7、8|保存配置、训练日志、严格checkpoint身份、逐场景日志和不可覆盖run root|strict evaluator audit；launcher artifact闭合；预登记报告|implemented|strict成功/失败测试；`test_launcher_rejects_non_strict_or_fallback_reconstruction_metadata`；existing-root拒绝、训练失败保留、四组非空artifact后才写`ARTIFACTS_COMPLETE`|正式模式禁止fallback、missing/unexpected/shape mismatch；真实输出根尚未生成|
|MUSE-017|Spec 2.1、5|训练期局部、自监督和nuisance头不得进入deployment bundle|`MUSETrainingHeads.deployment_state_dict`；checkpoint分层|verified|`test_training_state_contains_trainable_heads_but_no_deployment_state`；checkpoint round-trip；真实checkpoint smoke|MUSE训练态只进入训练checkpoint字段|
|MUSE-018|Spec 8|首轮记录H/M/L覆盖、有效未标注权重、三头JS分歧、prototype更新量和`z_id->receiver`泄漏诊断；伪标签precision只能由训练外诊断器计算|epoch telemetry、训练外诊断入口和最终checkpoint/report字段|implemented|`test_muse_telemetry_exposes_fixed_satellite_and_reliability_fields`；集成日志字段回归|训练内precision遥测当前为`N/A`，符合禁止读取`U_s`真值；但尚无独立训练外precision诊断入口，也无真实M0–M3 telemetry/泄漏探针结果，子要求未闭合|

## 反向生产文件审计

审计范围由`git diff --name-status 0e1019beb8f9c3217b4ae84f1a56a4be6dd5ba9e..4c66489ea058f5fe8401c29a237a58708bd7451f`确定。测试、报告和Task证据不作为生产逻辑；全部新增或修改生产文件如下。

|生产文件|Git变化|反向映射MUSE ID|可达性证据|结论|
|---|---|---|---|---|
|`code/cvsrffi/muse_ssdg.py`|新增，1,198行|MUSE-003至012、017|schedule、routing、loss、memory、satellite、training-head和checkpoint聚焦测试；训练入口实际导入|全部逻辑可映射，无规范外生产逻辑|
|`code/SSDG/train_ssdg.py`|修改，净增932行|MUSE-001至013、017、018|训练集成与协议负测；真实checkpoint M3 smoke；主`train()`调用链|全部新增MUSE路径可映射，无未接线模块|
|`code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`|新增，451行|MUSE-014至016|launcher dry-run与fake非dry-run控制流测试；`bash -n`由Task 7验证|全部逻辑可映射，无额外实验臂或发布gate|

反向审计结论：3/3个生产文件均至少映射到一个MUSE ID；未发现无法映射而需要删除或重新审批的生产逻辑。

## 单一N607 release归档清单

Task 8只完成发布准备，不创建归档、不连接N607。建议单一归档名为`adv3b02_muse_ssdg_code_4c66489ea058.tar.gz`，只包含下列3个Tasks 1–7生产文件；代码身份固定为`4c66489ea058f5fe8401c29a237a58708bd7451f`。

|归档成员|远端相对落点|用途|
|---|---|---|
|`code/cvsrffi/muse_ssdg.py`|同名路径|MUSE日程、路由、loss、memory和训练期头|
|`code/SSDG/train_ssdg.py`|同名路径|ADV3B02/MUSE训练接线和telemetry/checkpoint|
|`code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`|同名路径|M0–M3不可覆盖launcher与四场景评测闭环|

归档创建与N607落地仍为`pending`：后续runner须按最小流程只计算一次归档本地SHA和一次远端SHA，随后执行远端`python -m py_compile code/cvsrffi/muse_ssdg.py code/SSDG/train_ssdg.py`及`bash -n code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`。不得增加成员级SHA、seal、receipt或额外发布gate。

## 当前汇总与声明边界

- 总要求：18；`verified`：13；`implemented`：5；`pending`：0。
- 实现追踪闭合：18/18；反向生产文件映射：3/3。
- MUSE-002升级条件：保存实际loader receipt的精确路径，并读回四角色物理ID两两不交、source/target receiver集合不交和target样本计数0字段。
- MUSE-014升级条件：完成真实M0–M3单seed同协议矩阵并保存四臂run artifact；dry-run和fake runner不能替代该证据。
- MUSE-018升级条件：提供与训练状态、选择和停止逻辑隔离的训练外precision诊断入口，并在真实矩阵中保存H/M/L、有效权重、JS、prototype更新、泄漏探针和训练外precision结果。
- 最高风险剩余项：尚未运行真实M0–M3单seed训练，且MUSE-002实际loader receipt与MUSE-018独立precision诊断入口均未闭合；当前没有真实clean/三LEO场景结果、实际telemetry或性能晋级证据。
- final fix smoke出现1条来自`code/model.py`旧式`torch.cuda.amp.autocast`调用的`FutureWarning`；smoke退出码仍为0，prototype/prior/identity选择、前后向和S3C state回环均通过。该warning未升级为error。
- 当前状态是实现与发布准备完成，不是`ARTIFACTS_COMPLETE`、`ANALYZED`或性能完成。
