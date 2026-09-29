"""Copy this branch's explicitly owned D92 artifacts to the workspace front door."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WS=Path('E:/type10-7')
paths=['docs/D92_SUPPORT_UPGRADE_20260928.md','code/cvsrffi/stage2_d92_support_cv.py',
    'tools/predict_d92_support_cv.py','tools/build_d92_confirmation_data.py','tools/evaluate_d92_registration_proxy.py',
    'tools/run_d92_confirmation.py','tools/publish_d92_confirmation.py','tools/score_d92_confirmation.py',
    'tools/summarize_d92_confirmation.py','tools/read_d92_run.py','tools/sync_d92_run_record.py',
    'configs/d92_registration_proxy_20260928.json','configs/d92_confirmation_20260928.json',
    'configs/d92_confirmation_data_20260928.json','configs/d92_confirmation_data_record_20260928.json',
    'configs/d92_scv_frozen_20260928.json','configs/d92_confirmation_recovery_20260928.json',
    'tests/test_d92_confirmation_score.py','tests/test_run_d92_confirmation.py','tests/test_summarize_d92_confirmation.py',
    'docs/D92_SOURCE_AUX_PLAN_20260928.md','docs/D92_SOURCEFREE_AUX_20260928.md','docs/D92_UPGRADE_HANDOFF_20260928.md',
    'code/cvsrffi/stage2_d92_sourcefree_head.py','code/cvsrffi/d92_ground_summary.py',
    'tools/predict_d92_sourcefree_head.py','tools/prepare_d92_sourcefree_confirmation.py','tools/inspect_d92_ground_payload.py','tools/collect_d92_fit_logs.py',
    'configs/d92_sourcefree_confirmation_20260928.json','configs/d92_sourcefree_data_20260928.json',
    'configs/d92_sourcefree_data_record_20260928.json','configs/d92_sourcefree_frozen_20260928.json',
    'tests/test_d92_sourcefree_head.py','tests/test_d92_ground_summary.py','tests/test_d92_confirmation_allocation.py',
    'tests/test_predict_d92_sourcefree_head.py','tools/audit_d92_confirmation.py','tools/plot_d92_confirmation.py',
    'tests/test_audit_d92_confirmation.py','tools/collect_d92_optimizer_audit.py','tools/finalize_d92_sourcefree_record.py',
    'code/cvsrffi/stage2_d92_summary_joint.py','tools/predict_d92_summary_joint.py',
    'tests/test_d92_summary_joint.py','tests/test_predict_d92_summary_joint.py',
    'configs/d92_summary_joint_frozen_20260928.json',
    'docs/D92_SUMMARY_JOINT_DESIGN_20260928.md','docs/D92_SUMMARY_JOINT_DELIVERY_20260928.md',
    'tools/prepare_d92_summary_joint_benchmark.py','tools/summarize_d92_repeated_benchmark.py',
    'tests/test_summarize_d92_repeated_benchmark.py',
    'configs/d92_sgjoint_repeat_rx3_20260928.json','configs/d92_sgjoint_repeat_rx1_20260928.json',
    'configs/d92_sgjoint_repeat_rx3_data_20260928.json','configs/d92_sgjoint_repeat_rx1_data_20260928.json']
paths += ['tools/collect_d92_summary_joint_audit.py','tests/test_collect_d92_summary_joint_audit.py',
    'tools/finalize_d92_sgjoint_record.py','code/cvsrffi/stage2_d92_mv_kme.py','tests/test_d92_mv_kme.py',
    'configs/d92_mv_kme_frozen_20260928.json','docs/D92_MV_KME_DESIGN_20260928.md',
    'tools/export_d92_mv_kme_features.py','tools/predict_d92_mv_kme.py',
    'tests/test_export_d92_mv_kme_features.py','tests/test_predict_d92_mv_kme.py',
    'tools/prepare_d92_mv_kme_benchmark.py','tools/preflight_d92_mvkme.py','configs/d92_mvkme_repeat_rx3_20260928.json',
    'configs/d92_mvkme_repeat_rx1_20260928.json','configs/d92_mvkme_repeat_rx3_data_20260928.json',
    'configs/d92_mvkme_repeat_rx1_data_20260928.json','tools/prepare_d92_mvkme_recovery.py',
    'configs/d92_mvkme_repeat_rx3_recovery_20260928.json','configs/d92_mvkme_repeat_rx3_recovery_data_20260928.json',
    'tools/collect_d92_mvkme_audit.py','tests/test_collect_d92_mvkme_audit.py','tools/finalize_d92_mvkme_record.py']
paths += ['docs/D92_BNNA_DESIGN_20260929.md','code/cvsrffi/stage2_d92_bnna.py',
    'configs/d92_bnna_frozen_20260929.json','tests/test_d92_bnna.py',
    'tools/export_d92_bnna_features.py','tools/predict_d92_bnna.py',
    'tests/test_export_d92_bnna_features.py','tests/test_predict_d92_bnna.py',
    'tools/prepare_d92_bnna_benchmark.py','configs/d92_bnna_repeat_rx3_20260929.json',
    'configs/d92_bnna_repeat_rx1_20260929.json','configs/d92_bnna_repeat_rx3_data_20260929.json',
    'configs/d92_bnna_repeat_rx1_data_20260929.json',
    'tools/collect_d92_bnna_audit.py','tests/test_collect_d92_bnna_audit.py','docs/D92_BNNA_P0_REVIEW_20260929.md',
    'tools/finalize_d92_bnna_record.py','tests/test_finalize_d92_bnna_record.py','docs/D92_NEXT_SUPPORT_DESIGN_20260929.md']
paths += ['code/cvsrffi/stage2_d92_orbit_shared.py','configs/d92_orbit_shared_frozen_20260929.json',
    'tools/predict_d92_orbit_shared.py','tools/d92_orbit_feature_cache.py',
    'tools/collect_d92_orbit_shared_audit.py','tools/prepare_d92_orbit_shared_benchmark.py',
    'tests/test_d92_orbit_shared.py','tests/test_d92_orbit_feature_cache.py',
    'tests/test_predict_d92_orbit_shared.py','tests/test_collect_d92_orbit_shared_audit.py',
    'configs/d92_osc_repeat_rx3_20260929.json','configs/d92_osc_repeat_rx1_20260929.json',
    'configs/d92_osc_repeat_rx3_data_20260929.json','configs/d92_osc_repeat_rx1_data_20260929.json',
    'docs/D92_OSC_P0_REVIEW_20260929.md']
paths += ['tools/prepare_d92_osc_recovery.py',
    'configs/d92_osc_repeat_rx3_recovery_20260929.json','configs/d92_osc_repeat_rx1_recovery_20260929.json',
    'configs/d92_osc_repeat_rx3_recovery_data_20260929.json','configs/d92_osc_repeat_rx1_recovery_data_20260929.json']
paths += ['tools/finalize_d92_osc_record.py','tests/test_finalize_d92_osc_record.py']
paths += ['docs/D92_POST_OSC_DESIGN_20260929.md']
paths += ['docs/D92_OSC_SUPPORT_DIAGNOSTIC_20260929.md','docs/D92_OSC_SUPPORT_DIAGNOSTIC_20260929.json']
paths += ['docs/D92_SUPPORT_INFORMATION_DESIGN_20260929.md','docs/D92_MVRIDGE_P0_REVIEW_20260929.md',
    'code/cvsrffi/stage2_d92_multiview_ridge.py','configs/d92_multiview_ridge_frozen_20260929.json',
    'tools/predict_d92_multiview_ridge.py','tools/prepare_d92_multiview_ridge_benchmark.py',
    'tools/collect_d92_multiview_ridge_audit.py','tests/test_d92_multiview_ridge.py',
    'tests/test_predict_d92_multiview_ridge.py','tests/test_collect_d92_multiview_ridge_audit.py',
    'configs/d92_mvridge_repeat_rx3_20260929.json','configs/d92_mvridge_repeat_rx1_20260929.json',
    'configs/d92_mvridge_repeat_rx3_data_20260929.json','configs/d92_mvridge_repeat_rx1_data_20260929.json']
paths += ['tools/finalize_d92_mvridge_record.py','tests/test_finalize_d92_mvridge_record.py']
paths += ['docs/D92_FIXED_PHASE1_INFORMATION_AUDIT_20260929.md']
paths += ['docs/D92_FIXED_PHASE1_BRANCH_METADATA_20260929.json']
paths += ['docs/D92_BRANCH_SUPPORT_PROBE_DESIGN_20260929.md',
    'code/cvsrffi/d92_branch_support_probe.py','configs/d92_branch_support_probe_frozen_20260929.json',
    'tests/test_d92_branch_support_probe.py','tools/export_d92_branch_support_features.py',
    'tests/test_export_d92_branch_support_features.py','tools/evaluate_d92_branch_support_probe.py',
    'tests/test_evaluate_d92_branch_support_probe.py','tools/prepare_d92_branch_support_probe.py',
    'tools/run_d92_branch_support_probe.py','tools/publish_d92_branch_support_probe.py',
    'tools/preflight_d92_branch_support_probe.py','tests/test_run_d92_branch_support_probe.py',
    'configs/d92_branch_support_probe_20260929.json','configs/d92_branch_support_probe_rx3_20260929.json',
    'configs/d92_branch_support_probe_rx1_20260929.json','docs/D92_BRANCH_SUPPORT_PROBE_P0_20260929.md']
paths += ['tools/summarize_d92_branch_support_probe.py','tests/test_summarize_d92_branch_support_probe.py',
    'tools/analyze_d92_branch_support_probe.py']
paths += ['docs/D92_BRANCH_AUGMENT_DESIGN_20260929.md','docs/D92_BRANCH_RIDGE_P0_20260929.md','code/cvsrffi/d92_branch_ridge.py',
    'configs/d92_branch_ridge_frozen_20260929.json','tests/test_d92_branch_ridge.py',
    'tools/export_d92_branch_features.py','tools/evaluate_d92_branch_ridge.py',
    'tests/test_export_d92_branch_features.py','tests/test_evaluate_d92_branch_ridge.py',
    'tools/prepare_d92_branch_ridge_benchmark.py','tools/preflight_d92_branch_ridge.py',
    'configs/d92_branch_ridge_repeat_rx3_20260929.json','configs/d92_branch_ridge_repeat_rx1_20260929.json',
    'configs/d92_branch_ridge_repeat_rx3_data_20260929.json','configs/d92_branch_ridge_repeat_rx1_data_20260929.json']
paths += ['tools/collect_d92_branch_ridge_audit.py','tests/test_collect_d92_branch_ridge_audit.py']
paths += ['tools/finalize_d92_branch_ridge_record.py','tests/test_finalize_d92_branch_ridge_record.py']
paths += ['docs/D92_INDEPENDENT_DATA_AVAILABILITY_20260929.md','docs/D92_INDEPENDENT_DATA_METADATA_20260929.json']
paths += ['code/cvsrffi/d92_branch_interaction.py','configs/d92_branch_interaction_frozen_20260929.json',
    'configs/d92_branch_interaction_support_20260929.json','configs/d92_branch_interaction_support_rx3_20260929.json',
    'configs/d92_branch_interaction_support_rx1_20260929.json','docs/D92_BRANCH_INTERACTION_DESIGN_20260929.md',
    'docs/D92_BRANCH_INTERACTION_P0_20260929.md','docs/D92_BRANCH_RIDGE_K_NEW_EXPLANATION_20260929.md',
    'tools/evaluate_d92_branch_interaction_probe.py','tools/summarize_d92_branch_interaction_probe.py',
    'tools/run_d92_branch_interaction_probe.py','tools/publish_d92_branch_interaction_probe.py',
    'tools/preflight_d92_branch_interaction_probe.py','tools/prepare_d92_branch_interaction_probe.py',
    'tests/test_d92_branch_interaction.py','tests/test_evaluate_d92_branch_interaction_probe.py',
    'tests/test_summarize_d92_branch_interaction_probe.py','tests/test_run_d92_branch_interaction_probe.py']
paths += ['tools/evaluate_d92_branch_interaction.py','tests/test_evaluate_d92_branch_interaction.py',
    'tools/prepare_d92_branch_interaction_benchmark.py','tools/preflight_d92_branch_interaction.py',
    'tests/test_d92_branch_interaction_benchmark.py',
    'configs/d92_branch_interaction_repeat_rx3_20260929.json','configs/d92_branch_interaction_repeat_rx1_20260929.json',
    'configs/d92_branch_interaction_repeat_rx3_data_20260929.json','configs/d92_branch_interaction_repeat_rx1_data_20260929.json',
    'tools/summarize_d92_branch_interaction_benchmark.py','tests/test_summarize_d92_branch_interaction_benchmark.py',
    'tools/collect_d92_branch_interaction_audit.py','tests/test_collect_d92_branch_interaction_audit.py']
paths += ['code/cvsrffi/d92_branch_metric.py','configs/d92_branch_metric_frozen_20260929.json',
    'configs/d92_branch_metric_support_20260929.json','configs/d92_branch_metric_support_rx3_20260929.json',
    'configs/d92_branch_metric_support_rx1_20260929.json','docs/D92_NEXT_SUPPORT_METHOD_20260929.md',
    'docs/D92_EXISTING_GROUND_FEASIBILITY_20260929.md','docs/D92_BRANCH_METRIC_P0_20260929.md',
    'tools/evaluate_d92_branch_metric_probe.py','tools/summarize_d92_branch_metric_probe.py',
    'tools/run_d92_branch_metric_probe.py','tools/publish_d92_branch_metric_probe.py',
    'tools/preflight_d92_branch_metric_probe.py','tools/prepare_d92_branch_metric_probe.py',
    'tools/analyze_d92_branch_metric_probe.py',
    'tests/test_d92_branch_metric.py','tests/test_evaluate_d92_branch_metric_probe.py',
    'tests/test_summarize_d92_branch_metric_probe.py','tests/test_run_d92_branch_metric_probe.py']
paths += ['code/cvsrffi/d92_branch_orbit_ce.py','configs/d92_branch_orbit_ce_frozen_20260929.json',
    'configs/d92_branch_orbit_ce_support_20260929.json','configs/d92_branch_orbit_ce_support_rx3_20260929.json',
    'configs/d92_branch_orbit_ce_support_rx1_20260929.json','docs/D92_NEXT_AFTER_METRIC_20260929.md',
    'docs/D92_BRANCH_ORBIT_CE_P0_20260929.md','tools/export_d92_branch_orbit_support_features.py',
    'tools/evaluate_d92_branch_orbit_ce_probe.py','tools/summarize_d92_branch_orbit_ce_probe.py',
    'tools/run_d92_branch_orbit_ce_probe.py','tools/publish_d92_branch_orbit_ce_probe.py',
    'tools/preflight_d92_branch_orbit_ce_probe.py','tools/prepare_d92_branch_orbit_ce_probe.py',
    'tools/analyze_d92_branch_orbit_ce_probe.py','tests/test_d92_branch_orbit_ce.py',
    'tests/test_export_d92_branch_orbit_support_features.py','tests/test_evaluate_d92_branch_orbit_ce_probe.py',
    'tests/test_summarize_d92_branch_orbit_ce_probe.py','tests/test_run_d92_branch_orbit_ce_probe.py']
paths += ['docs/D92_BRANCH_ORBIT_COST_20260929.md']
paths += ['code/cvsrffi/d92_branch_local_ridge.py',
    'configs/d92_branch_local_ridge_frozen_20260929.json',
    'configs/d92_branch_local_ridge_support_20260929.json',
    'configs/d92_branch_local_ridge_support_rx3_20260929.json',
    'configs/d92_branch_local_ridge_support_rx1_20260929.json',
    'docs/D92_NEXT_AFTER_ORBIT_20260929.md', 'docs/D92_LOCAL_TEST_ENV_RECOVERY_20260929.md',
    'docs/D92_BRANCH_LOCAL_RIDGE_P0_20260929.md',
    'tools/evaluate_d92_branch_local_ridge_probe.py', 'tools/summarize_d92_branch_local_ridge_probe.py',
    'tools/run_d92_branch_local_ridge_probe.py', 'tools/publish_d92_branch_local_ridge_probe.py',
    'tools/analyze_d92_branch_local_ridge_probe.py', 'tools/preflight_d92_branch_local_ridge_probe.py',
    'tools/prepare_d92_branch_local_ridge_probe.py',
    'tests/test_d92_branch_local_ridge.py', 'tests/test_evaluate_d92_branch_local_ridge_probe.py',
    'tests/test_summarize_d92_branch_local_ridge_probe.py', 'tests/test_run_d92_branch_local_ridge_probe.py']
paths += ['docs/D92_INDEPENDENT_DATA_METADATA_ADDENDUM_20260929.md',
    'docs/D92_BRANCH_LOCAL_RIDGE_RESULT_20260929.md',
    'docs/D92_BRANCH_LOCAL_RIDGE_SUPPORT_RESULT_20260929.md',
    'docs/D92_BRANCH_LOCAL_RIDGE_BENCHMARK_P0_20260929.md',
    'tools/evaluate_d92_branch_local_ridge.py','tests/test_evaluate_d92_branch_local_ridge.py',
    'tools/prepare_d92_branch_local_ridge_benchmark.py','tools/preflight_d92_branch_local_ridge.py',
    'tests/test_d92_branch_local_ridge_benchmark.py',
    'tools/summarize_d92_branch_local_ridge_benchmark.py','tests/test_summarize_d92_branch_local_ridge_benchmark.py',
    'tools/collect_d92_branch_local_ridge_audit.py','tests/test_collect_d92_branch_local_ridge_audit.py',
    'configs/d92_branch_local_ridge_repeat_rx3_20260929.json','configs/d92_branch_local_ridge_repeat_rx1_20260929.json',
    'configs/d92_branch_local_ridge_repeat_rx3_data_20260929.json','configs/d92_branch_local_ridge_repeat_rx1_data_20260929.json']
paths += ['docs/D92_NEXT_AFTER_LOCAL_RIDGE_20260929.md',
    'docs/D92_LOCAL_MARGIN_MATH_REVIEW_20260929.md',
    'code/cvsrffi/d92_branch_local_margin.py',
    'configs/d92_branch_local_margin_frozen_20260929.json',
    'configs/d92_branch_local_margin_support_20260929.json',
    'configs/d92_branch_local_margin_support_rx3_20260929.json',
    'configs/d92_branch_local_margin_support_rx1_20260929.json',
    'tools/evaluate_d92_branch_local_margin_probe.py',
    'tools/summarize_d92_branch_local_margin_probe.py',
    'tools/run_d92_branch_local_margin_probe.py',
    'tools/publish_d92_branch_local_margin_probe.py',
    'tools/preflight_d92_branch_local_margin_probe.py',
    'tools/prepare_d92_branch_local_margin_probe.py',
    'tools/analyze_d92_branch_local_margin_probe.py',
    'tests/test_d92_branch_local_margin.py',
    'tests/test_evaluate_d92_branch_local_margin_probe.py',
    'tests/test_summarize_d92_branch_local_margin_probe.py',
    'tests/test_run_d92_branch_local_margin_probe.py']
paths += ['tools/evaluate_d92_branch_local_margin.py',
    'tools/prepare_d92_branch_local_margin_benchmark.py',
    'tools/preflight_d92_branch_local_margin.py',
    'tools/summarize_d92_branch_local_margin_benchmark.py',
    'tools/collect_d92_branch_local_margin_audit.py',
    'tests/test_evaluate_d92_branch_local_margin.py',
    'tests/test_d92_branch_local_margin_benchmark.py',
    'tests/test_summarize_d92_branch_local_margin_benchmark.py',
    'tests/test_collect_d92_branch_local_margin_audit.py',
    'docs/D92_LOCAL_MARGIN_BENCHMARK_REVIEW_20260929.md']
paths += ['docs/D92_ADAPTATION_REGISTRATION_TARGET_20260930.md']
paths += ['tools/d92_adaptation_registration_pairing.py',
    'tools/collect_d92_adaptation_registration_identities.py',
    'tools/summarize_d92_adaptation_registration.py',
    'tests/test_d92_adaptation_registration_pairing.py',
    'tests/test_collect_d92_adaptation_registration_identities.py',
    'tests/test_summarize_d92_adaptation_registration.py',
    'docs/D92_ABC_ANALYSIS_REVIEW_20260930.md']
paths += ['docs/D92_BRANCH_LOCAL_RIDGE_ABC_RESULT_20260930.md']
for name in paths:
    src=ROOT/name;dst=WS/name;dst.parent.mkdir(parents=True,exist_ok=True)
    # Each name is unique to this task; original D92 and shared wrappers are not mirrored.
    shutil.copyfile(src,dst)
    if src.read_bytes()!=dst.read_bytes():raise ValueError('Mirror mismatch: '+name)
    text=dst.read_text(encoding='utf-8')
    if '\ufffd' in text:raise ValueError('Encoding corruption: '+name)
print('VERIFIED explicit owned mirrors:',len(paths))
