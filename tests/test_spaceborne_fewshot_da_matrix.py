import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


class SpaceborneFewShotDaMatrixTest(unittest.TestCase):
    def test_smoke_matrix_has_separate_sfe_and_ftrc_candidates(self):
        from spaceborne_fewshot_da_matrix import make_candidates

        candidates = make_candidates(plan="SMOKE")
        ids = [c.cid for c in candidates]

        self.assertIn("SFE_ZID_PROTO_K5_SYNTH", ids)
        self.assertIn("FTRC_SAT_RXTX_K2_LABELED_BASE", ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(c.target_visibility for c in candidates))
        self.assertTrue(all(c.label_set_relation for c in candidates))
        ftrc = next(c for c in candidates if c.protocol == "CVS-FTRC")
        self.assertEqual(ftrc.target_visibility, "target_receiver_satellite_support_labeled")

    def test_launcher_and_report_expose_target_visibility_and_metrics(self):
        from spaceborne_fewshot_da_matrix import make_candidates, render_launcher, render_report

        candidates = make_candidates(plan="SMOKE")
        launcher = render_launcher("spaceborne_fewshot_da_smoke_test", candidates)
        report = render_report("spaceborne_fewshot_da_smoke_test", candidates)

        self.assertIn("eval_spaceborne_fewshot.py", launcher)
        self.assertIn("train_target_adapt.py", launcher)
        self.assertIn("--target_label_mode labeled", launcher)
        self.assertIn("--target_samples_per_rx_tx 2", launcher)
        self.assertIn("--target_channel_view satellite", launcher)
        self.assertIn("--target_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", launcher)
        self.assertNotIn("provided_satellite", launcher)
        self.assertIn("H_sg", report)
        self.assertIn("target_visibility", report)
        self.assertIn("label_set_relation", report)
        self.assertIn("full_accuracy", report)
        self.assertIn("coverage", report)

    def test_matrix_json_is_machine_readable(self):
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload

        payload = matrix_payload("spaceborne_fewshot_da_smoke_test", make_candidates(plan="SMOKE"))
        decoded = json.loads(json.dumps(payload))

        self.assertEqual(decoded["run_id"], "spaceborne_fewshot_da_smoke_test")
        self.assertEqual(len(decoded["candidates"]), 2)
        self.assertEqual(decoded["candidates"][0]["protocol"], "CVS-SFE")

    def test_core_plan_is_supervised_satellite_only_without_semisupervised_losses(self):
        from spaceborne_fewshot_da_matrix import make_candidates, render_launcher, render_report

        candidates = make_candidates(plan="CORE")
        launcher = render_launcher("spaceborne_fewshot_da_supervised_core_test", candidates)
        report = render_report("spaceborne_fewshot_da_supervised_core_test", candidates)

        ftrc_ks = sorted(c.k for c in candidates if c.protocol == "CVS-FTRC")
        sfe_ks = sorted(c.k for c in candidates if c.protocol == "CVS-SFE")

        self.assertEqual(ftrc_ks, [1, 2, 5, 10, 20])
        self.assertEqual(sfe_ks, [1, 2, 5, 10, 20])
        self.assertIn("--target_channel_view satellite", launcher)
        self.assertIn("--target_label_mode labeled", launcher)
        self.assertIn("--entropy_weight 0", launcher)
        self.assertIn("--consistency_weight 0", launcher)
        self.assertIn("--pseudo_weight 0", launcher)
        self.assertIn("--epochs 20", launcher)
        self.assertIn("--adapt_steps_per_epoch 20", launcher)
        self.assertNotIn("--target_label_mode unlabeled", launcher)
        self.assertNotIn("--sat_eval_max_batches 1", launcher)
        self.assertNotIn("--eval_max_batches 1", launcher)
        self.assertNotIn("provided_satellite", launcher)
        self.assertIn("no semi-supervised", report.lower())

    def test_launcher_waits_for_background_candidate_jobs(self):
        from spaceborne_fewshot_da_matrix import make_candidates, render_launcher

        launcher = render_launcher("spaceborne_fewshot_da_supervised_core_test", make_candidates(plan="CORE"))

        self.assertIn("PIDS=()", launcher)
        self.assertIn("NAMES=()", launcher)
        self.assertIn("[SPACEBORNE-FSDA-LAUNCHED]", launcher)
        self.assertIn("wait \"${PIDS[${idx}]}\"", launcher)
        self.assertIn("[SPACEBORNE-FSDA-COMPLETE]", launcher)

    def test_wisig_newclass_plan_exports_features_and_passes_tx_id_audit(self):
        from spaceborne_fewshot_da_matrix import make_candidates, render_launcher, render_report

        candidates = make_candidates(plan="WISIG_NEWCLASS")
        launcher = render_launcher("spaceborne_fewshot_wisig_newclass_test", candidates)
        report = render_report("spaceborne_fewshot_wisig_newclass_test", candidates)

        self.assertEqual([c.cid for c in candidates], ["SFE_WISIG_NEW_TX_K5_STRICT"])
        self.assertIn("export_spaceborne_features.py", launcher)
        self.assertIn("eval_spaceborne_fewshot.py", launcher)
        self.assertIn("--source_tx_ids", launcher)
        self.assertIn("SOURCE_TX_IDS", launcher)
        self.assertIn("--new_tx_ids", launcher)
        self.assertIn("NEW_TX_IDS", launcher)
        self.assertIn("NEW_WISIG_PKL", launcher)
        self.assertIn("tx_overlap_audit", report)
        self.assertIn("Y_T_has_explicit_nonoverlap_tx", report)

    def test_wisig_newclass_card8_plan_adds_one_real_sfe_candidate_per_gpu(self):
        from spaceborne_fewshot_da_matrix import make_candidates, render_launcher

        candidates = make_candidates(plan="WISIG_NEWCLASS_CARD8")
        launcher = render_launcher("spaceborne_fewshot_wisig_card8_test", candidates)

        self.assertEqual(len(candidates), 8)
        self.assertEqual(sorted(c.gpu for c in candidates), list(range(8)))
        self.assertTrue(all(c.command_kind == "feature_sfe_wisig_nonoverlap" for c in candidates))
        self.assertIn("--gate_mode cosine", launcher)
        self.assertIn("--unknown_threshold 0.6", launcher)
        self.assertIn("--unknown_threshold 0.8", launcher)
        self.assertIn("--seed 1372", launcher)

    def test_wisig_enhanced_card8_plan_covers_gates_adapters_and_rollback(self):
        from spaceborne_fewshot_da_matrix import make_candidates, render_launcher, matrix_payload, render_report

        candidates = make_candidates(plan="WISIG_ENHANCED_CARD8")
        launcher = render_launcher("spaceborne_fewshot_wisig_enhanced_test", candidates)
        report = render_report("spaceborne_fewshot_wisig_enhanced_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_wisig_enhanced_test", candidates)

        self.assertEqual(len(candidates), 8)
        self.assertEqual(sorted(c.gpu for c in candidates), list(range(8)))
        self.assertIn("--gate_mode mahalanobis", launcher)
        self.assertIn("--gate_mode openmax", launcher)
        self.assertIn("--gate_mode combined", launcher)
        self.assertIn("--target_adapter_type feature_residual", launcher)
        self.assertIn("--target_adapter_type logit_lora", launcher)
        self.assertIn("--rollback_enabled true", launcher)
        self.assertIn("rollback", report.lower())
        self.assertIn("gate_mode", json.dumps(payload))
        self.assertIn("target_adapter_type", json.dumps(payload))

    def test_oa_mse_plan_exports_stage2_protocol_cards_and_validator_fields(self):
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher, render_report

        candidates = make_candidates(plan="OA_MSE_CARD3")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_test", candidates)
        report = render_report("spaceborne_fewshot_oa_mse_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_test", candidates)

        self.assertEqual([c.oa_mse_stage for c in candidates], ["mse_lite", "mse_subspace", "oa_mse_head"])
        self.assertEqual([c.command_kind for c in candidates], ["feature_oa_mse_wisig_nonoverlap"] * 3)
        self.assertEqual([c.k for c in candidates], [0, 2, 5])
        self.assertEqual(candidates[1].target_old_support_per_tx, candidates[1].k)
        self.assertEqual(candidates[2].target_old_support_per_tx, candidates[2].k)
        self.assertEqual(candidates[2].target_new_support_per_tx, candidates[2].k)
        self.assertNotIn("new_class_accuracy", candidates[0].metrics)
        self.assertNotIn("new_class_accuracy", candidates[1].metrics)
        self.assertIn("new_class_accuracy", candidates[2].metrics)
        self.assertIn("--gate_mode oa_mse", launcher)
        self.assertIn("--protocol source_open_set", launcher)
        self.assertIn("--protocol ftrc", launcher)
        self.assertIn("--protocol sfe", launcher)
        self.assertIn("--target_old_tx_ids", launcher)
        self.assertIn("--target_old_support_per_tx", launcher)
        self.assertIn("--source_rxs", launcher)
        self.assertIn("--target_old_rxs", launcher)
        self.assertIn("--new_rxs", launcher)
        self.assertIn("--oa_mse_adapter_steps", launcher)
        self.assertIn("--old_acc_target 0.9", launcher)
        self.assertIn("--seen_new_acc_target 0.75", launcher)
        stage2b_block = launcher.split("id=OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD", 1)[1].split("id=OA_MSE_STAGE2C_HEAD_SEEN_NEW", 1)[0]
        self.assertIn("--target_old_support_per_tx", stage2b_block)
        self.assertIn("--target_old_support_per_tx \\\"2\\\"", stage2b_block)
        self.assertIn("--shots 0", stage2b_block)
        self.assertNotIn("--shots 2", stage2b_block)
        stage2c_block = launcher.split("id=OA_MSE_STAGE2C_HEAD_SEEN_NEW", 1)[1]
        self.assertIn("--shots 5", stage2c_block)
        self.assertIn("OA-MSE", report)
        decoded = json.loads(json.dumps(payload))["candidates"]
        self.assertTrue(all(item["route_family"] == "OA_MSE_HEAD" for item in decoded))
        self.assertTrue(all(item["onboard_low_compute_training"] is True for item in decoded))
        self.assertTrue(all(item["weibull_evt_required"] is True for item in decoded))
        self.assertTrue(all(item["target_adapter_required"] is True for item in decoded))
        self.assertTrue(all(item["pseudo_unknown_energy_required"] is True for item in decoded))
        self.assertTrue(all(item["seen_new_evidence_gate_required"] is True for item in decoded))
        self.assertTrue(all(item["seen_new_anchor_gate_required"] is True for item in decoded))
        self.assertTrue(all(item["siamese_verifier_required"] is True for item in decoded))
        self.assertTrue(all(item["accepted_only_online_update_required"] is True for item in decoded))
        bundle_text = " ".join(item["oa_mse_onboard_adaptation_bundle"] for item in decoded)
        for token in (
            "weibull_evt",
            "target_adapter",
            "pseudo_unknown_energy",
            "seen_new_evidence_gate",
            "seen_new_anchor_gate",
            "siamese_verifier",
            "accepted_only_online_update",
            "stage2_receiver_domain",
        ):
            self.assertIn(token, bundle_text)
        self.assertTrue(all(float(item["old_acc_target"]) >= 0.90 for item in decoded))
        self.assertTrue(all(float(item["seen_new_acc_target"]) >= 0.75 for item in decoded))
        self.assertTrue(all(item["unknown_query_eval_only"] is True for item in decoded))
        self.assertTrue(all(item["target_new_query_not_threshold_fit"] is True for item in decoded))
        self.assertTrue(all(item["seen_new_evidence_gate_unknown_query_calibration"] is False for item in decoded))
        self.assertTrue(all(float(item["unknown_FAR_target"]) <= 0.05 for item in decoded))
        stage2c = next(item for item in decoded if item["oa_mse_stage"] == "oa_mse_head")
        for token in (
            "old_acc",
            "seen_new_acc",
            "H_old_new",
            "unknown_FAR",
            "unknown_to_seen_new",
        ):
            self.assertIn(token, stage2c["stage2c_success_metric_bundle"])
        for token in (
            "candidate_label",
            "seen_new_minus_old_score",
            "seen_new_anchor_similarity",
            "seen_new_anchor_delta",
        ):
            self.assertIn(token, stage2c["score_table_required_columns"])
        self.assertIn("pseudo_unknown_only", stage2c["seen_new_evidence_gate_calibration_scope"])
        self.assertTrue(all("defer" in item["model_output_semantics"] for item in decoded))
        self.assertTrue(all(item["clean_view_role"] == "control_only" for item in decoded))
        self.assertTrue(all(item["deployment_success_claim_allowed"] is False for item in decoded))
        self.assertTrue(all(item["evidence_level"] == "receiver_x_transmitter_proxy_stress" for item in decoded))
        self.assertTrue(all(item["support_query_split_verified"] is True for item in decoded))

    def test_oa_mse_launcher_defaults_use_project_phase2_single_rsat_labels(self):
        from spaceborne_fewshot_da_matrix import make_candidates, render_launcher

        launcher = render_launcher("spaceborne_fewshot_oa_mse_test", make_candidates(plan="OA_MSE_CARD3"))

        self.assertIn('SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"', launcher)
        self.assertIn('TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"', launcher)
        self.assertIn('NEW_TX_IDS="${NEW_TX_IDS:-1-16,1-18}"', launcher)
        self.assertIn('OA_MSE_UNKNOWN_TX_IDS="${OA_MSE_UNKNOWN_TX_IDS:-10-1,10-10}"', launcher)
        self.assertIn('CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"', launcher)
        self.assertIn('TARGET_RECEIVER_IDS="${TARGET_RECEIVER_IDS:-20-1}"', launcher)
        self.assertNotIn('TARGET_RECEIVER_IDS="${TARGET_RECEIVER_IDS:-7}"', launcher)

    def test_oa_mse_proxy32_expands_to_four_rows_per_gpu_with_proxy_alpha_selection(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_PROXY32")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_proxy32_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_proxy32_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(32, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 4 for gpu in range(8)})
        self.assertEqual({c.oa_mse_adapter_selection_policy for c in candidates}, {"proxy_line_search"})
        self.assertEqual({c.oa_mse_adapter_alpha_eval_sweep for c in candidates}, {True})
        self.assertIn("--oa_mse_adapter_selection_policy proxy_line_search", launcher)
        self.assertIn("--oa_mse_adapter_alpha_eval_sweep", launcher)
        self.assertTrue(all(item["parameters"]["oa_mse_adapter_selection_policy"] == "proxy_line_search" for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_adapter_alpha_eval_sweep"] is True for item in rows))
        self.assertTrue(all(item["lane"] == "phase2_spaceborne_fsl" for item in rows))

    def test_oa_mse_boundary32_uses_target_boundary_guard_and_stronger_soft_proto(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_BOUNDARY32")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_boundary32_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_boundary32_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(32, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 4 for gpu in range(8)})
        self.assertEqual({c.oa_mse_adapter_selection_policy for c in candidates}, {"target_boundary_guard"})
        self.assertTrue(all(c.oa_mse_soft_proto_weight >= 0.12 for c in candidates))
        self.assertIn("--oa_mse_adapter_selection_policy target_boundary_guard", launcher)
        self.assertIn("--oa_mse_adapter_alpha_eval_sweep", launcher)
        self.assertIn("target_boundary_guard_selector", json.dumps(rows))
        self.assertTrue(all(item["parameters"]["oa_mse_adapter_selection_policy"] == "target_boundary_guard" for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_soft_proto_weight"] >= 0.12 for item in rows))

    def test_oa_mse_uncertain32_emits_old_surrogate_uncertain_band_controls(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_UNCERTAIN32")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_uncertain32_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_uncertain32_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(32, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 4 for gpu in range(8)})
        self.assertEqual({c.oa_mse_adapter_selection_policy for c in candidates}, {"target_boundary_guard"})
        self.assertTrue(all(c.old_surrogate_reject_relax > 0.0 for c in candidates))
        self.assertTrue(all(c.oa_mse_siamese_accept_threshold >= 0.62 for c in candidates))
        self.assertIn("--old_surrogate_reject_relax", launcher)
        self.assertIn("--oa_mse_siamese_quantile", launcher)
        self.assertIn("--oa_mse_siamese_accept_threshold", launcher)
        self.assertIn("uncertain_band", json.dumps(rows))
        self.assertTrue(all(item["parameters"]["old_surrogate_reject_relax"] > 0.0 for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_siamese_accept_threshold"] >= 0.62 for item in rows))

    def test_oa_mse_veto32_emits_unknown_risk_veto_controls(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_VETO32")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_veto32_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_veto32_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(32, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 4 for gpu in range(8)})
        self.assertEqual({c.command_kind for c in candidates}, {"feature_oa_mse_wisig_nonoverlap"})
        self.assertEqual({c.oa_mse_adapter_selection_policy for c in candidates}, {"target_boundary_guard"})
        self.assertTrue(all(c.oa_mse_siamese_unknown_veto for c in candidates))
        self.assertTrue(all(c.old_surrogate_reject_relax > 0.0 for c in candidates))
        self.assertIn("--oa_mse_siamese_unknown_veto", launcher)
        self.assertIn("--oa_mse_siamese_min_old_support_evidence_delta", launcher)
        self.assertIn("--oa_mse_siamese_min_old_surrogate_reject_delta", launcher)
        self.assertIn("--oa_mse_siamese_min_energy_delta", launcher)
        self.assertIn("unknown_veto", json.dumps(rows))
        self.assertTrue(all(item["lane"] == "phase2_spaceborne_fsl" for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_siamese_unknown_veto"] for item in rows))
        self.assertTrue(
            all(item["parameters"]["oa_mse_siamese_min_old_support_evidence_delta"] is not None for item in rows)
        )

    def test_oa_mse_classcond32_emits_coupled_anchor_margin_veto_controls(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_CLASSCOND32")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_classcond32_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_classcond32_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(32, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 4 for gpu in range(8)})
        self.assertEqual({c.command_kind for c in candidates}, {"feature_oa_mse_wisig_nonoverlap"})
        self.assertEqual({c.oa_mse_siamese_unknown_veto_mode for c in candidates}, {"coupled"})
        self.assertTrue(all(c.oa_mse_siamese_unknown_veto for c in candidates))
        self.assertTrue(all(c.oa_mse_siamese_min_old_support_anchor_margin is not None for c in candidates))
        self.assertIn("--oa_mse_siamese_unknown_veto_mode coupled", launcher)
        self.assertIn("--oa_mse_siamese_min_old_support_anchor_margin", launcher)
        self.assertIn("--oa_mse_siamese_min_veto_failures", launcher)
        self.assertIn("OA_MSE_CLASSCOND32", json.dumps(rows))
        self.assertTrue(all(item["parameters"]["oa_mse_siamese_unknown_veto_mode"] == "coupled" for item in rows))
        self.assertTrue(
            all(item["parameters"]["oa_mse_siamese_min_old_support_anchor_margin"] is not None for item in rows)
        )

    def test_oa_mse_calguard32_emits_post_accept_old_unknown_guard_controls(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_CALGUARD32")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_calguard32_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_calguard32_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(32, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 4 for gpu in range(8)})
        self.assertEqual({c.command_kind for c in candidates}, {"feature_oa_mse_wisig_nonoverlap"})
        self.assertTrue(all(c.oa_mse_old_unknown_acceptance_guard for c in candidates))
        self.assertIn("--oa_mse_old_unknown_acceptance_guard", launcher)
        self.assertIn("--oa_mse_old_unknown_guard_min_best_old_score", launcher)
        self.assertIn("OA_MSE_CALGUARD32", json.dumps(rows))
        self.assertTrue(all(item["parameters"]["oa_mse_old_unknown_acceptance_guard"] for item in rows))
        self.assertTrue(
            all(item["parameters"]["oa_mse_old_unknown_guard_min_best_old_score"] is not None for item in rows)
        )

    def test_oa_mse_balance64_expands_to_eight_rows_per_gpu_and_four_active_default(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_BALANCE64")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_balance64_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_balance64_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(64, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 8 for gpu in range(8)})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {4})
        self.assertEqual(payload["lane_quota_mode"], "phase2_only")
        self.assertEqual(payload["phase1_rows_expected"], 0)
        self.assertEqual(payload["phase2_rows_expected"], 64)
        self.assertEqual(payload["stage2_sample_protocol"]["source_receiver_labels"], "1-1,1-19,14-7,18-2,19-2,2-1,2-19")
        self.assertIn('STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-4}"', launcher)
        self.assertIn("OA_MSE_BALANCE64", json.dumps(rows))
        self.assertEqual(
            {item["slot"] for item in rows},
            {f"GPU{gpu}/{slot}" for gpu in range(8) for slot in "ABCDEFGH"},
        )
        self.assertTrue(
            all(item["registry_key"] == f"spaceborne_fewshot_oa_mse_balance64_test:{item['candidate_id']}" for item in rows)
        )
        self.assertEqual(64, len({item["command_hash"] for item in rows}))
        self.assertTrue(any(c.target_old_support_per_tx == 20 for c in candidates))
        self.assertTrue(any(c.target_new_support_per_tx == 20 for c in candidates))
        self.assertTrue(any(c.oa_mse_old_unknown_acceptance_guard for c in candidates))
        self.assertTrue(any(not c.oa_mse_old_unknown_acceptance_guard for c in candidates))
        self.assertTrue(
            all(item["parameters"]["stage2_max_active_per_gpu"] == 4 for item in rows)
        )
        self.assertTrue(
            all(item["parameters"]["oa_mse_adapter_selection_policy"] == "target_boundary_guard" for item in rows)
        )

    def test_oa_mse_softmix64_enables_soft_prototype_boundary_loss(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_SOFTMIX64")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_softmix64_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_softmix64_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(64, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 8 for gpu in range(8)})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {4})
        self.assertIn("--oa_mse_soft_proto_boundary_weight", launcher)
        self.assertIn("--soft_proto_boundary_margin", launcher)
        self.assertTrue(all(c.oa_mse_soft_proto_boundary_weight > 0 for c in candidates))
        self.assertTrue(all(c.oa_mse_soft_proto_weight >= 0.18 for c in candidates))
        self.assertTrue(
            all(item["parameters"]["oa_mse_soft_proto_boundary_weight"] > 0 for item in rows)
        )
        self.assertIn("OA_MSE_SOFTMIX64", json.dumps(rows))

    def test_oa_mse_void64_enables_void_background_gate_and_full_gpu_packing(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_VOID64")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_void64_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_void64_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(64, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 8 for gpu in range(8)})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {4})
        self.assertTrue(all(c.oa_mse_void_background_weight > 0 for c in candidates))
        self.assertTrue(all(c.oa_mse_void_gate for c in candidates))
        self.assertIn("--oa_mse_void_background_weight", launcher)
        self.assertIn("--oa_mse_void_gate", launcher)
        self.assertIn("--oa_mse_void_gate_min_score", launcher)
        self.assertIn("--oa_mse_void_gate_min_margin", launcher)
        self.assertTrue(all(item["parameters"]["oa_mse_void_background_weight"] > 0 for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_void_gate"] for item in rows))
        self.assertTrue(
            all("pseudo_unknown_void_background_gate" in item["fusion_inputs"] for item in rows)
        )
        self.assertIn("OA_MSE_VOID64", json.dumps(rows))

    def test_oa_mse_softvoid128_expands_queue_and_interleaves_gpu_launches(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_SOFTVOID128")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_softvoid128_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_softvoid128_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(128, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 16 for gpu in range(8)})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {6})
        self.assertEqual(payload["expected_count"], 128)
        self.assertEqual(payload["lane_quota_mode"], "phase2_only")
        self.assertEqual(payload["phase1_rows_expected"], 0)
        self.assertEqual(payload["phase2_rows_expected"], 128)
        self.assertEqual(payload["phase2_gpu_utilization_policy"]["queued_rows_per_gpu"], 16)
        self.assertEqual(payload["phase2_gpu_utilization_policy"]["max_active_per_gpu"], 6)
        self.assertIn('STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-6}"', launcher)
        self.assertIn("OA_MSE_SOFTVOID128", json.dumps(rows))
        self.assertTrue(all(item["parameters"]["oa_mse_void_background_weight"] > 0 for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_soft_proto_boundary_weight"] > 0 for item in rows))
        first_launches = [
            line
            for line in launcher.splitlines()
            if line.startswith('echo "[SPACEBORNE-FSDA-CANDIDATE]')
        ][:8]
        self.assertEqual(
            [f"OA_MSE_SOFTVOID128_GPU{gpu}_A_MSE_SUBSPACE_KOLD2_KNEW0" for gpu in range(8)],
            [line.split("id=", 1)[1].split(" ", 1)[0] for line in first_launches],
        )

    def test_oa_mse_anchorguard128_uses_four_per_gpu_and_old_unknown_guard(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_ANCHORGUARD128")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_anchorguard128_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_anchorguard128_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(128, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 16 for gpu in range(8)})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {4})
        self.assertEqual(payload["phase2_gpu_utilization_policy"]["max_active_per_gpu"], 4)
        self.assertIn('STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-4}"', launcher)
        self.assertIn("OA_MSE_ANCHORGUARD128", json.dumps(rows))
        self.assertTrue(all(c.oa_mse_old_unknown_acceptance_guard for c in candidates))
        self.assertTrue(all(c.oa_mse_void_gate for c in candidates))
        self.assertTrue(all(c.oa_mse_void_background_weight > 0 for c in candidates))
        self.assertTrue(
            all("soft_prototype_mixture" in item["fusion_inputs"] for item in rows)
        )
        first_launches = [
            line
            for line in launcher.splitlines()
            if line.startswith('echo "[SPACEBORNE-FSDA-CANDIDATE]')
        ][:8]
        self.assertEqual(
            [f"OA_MSE_ANCHORGUARD128_GPU{gpu}_A_MSE_SUBSPACE_KOLD2_KNEW0" for gpu in range(8)],
            [line.split("id=", 1)[1].split(" ", 1)[0] for line in first_launches],
        )

    def test_oa_mse_mixhead128_uses_multiproto_score_head(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_MIXHEAD128")
        launcher = render_launcher("spaceborne_fewshot_oa_mse_mixhead128_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_oa_mse_mixhead128_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(128, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 16 for gpu in range(8)})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {4})
        self.assertEqual(payload["phase2_gpu_utilization_policy"]["max_active_per_gpu"], 4)
        self.assertIn('STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-4}"', launcher)
        self.assertIn("--oa_mse_multiproto_score", launcher)
        self.assertIn("--multiproto_score_weight", launcher)
        self.assertIn("OA_MSE_MIXHEAD128", json.dumps(rows))
        self.assertTrue(all(c.oa_mse_multiproto_score for c in candidates))
        self.assertTrue(all(item["parameters"]["oa_mse_multiproto_score"] for item in rows))
        self.assertTrue(all("soft_multi_prototype_score_head" in item["fusion_inputs"] for item in rows))
        first_launches = [
            line
            for line in launcher.splitlines()
            if line.startswith('echo "[SPACEBORNE-FSDA-CANDIDATE]')
        ][:8]
        self.assertEqual(
            [f"OA_MSE_MIXHEAD128_GPU{gpu}_A_MSE_SUBSPACE_KOLD2_KNEW0" for gpu in range(8)],
            [line.split("id=", 1)[1].split(" ", 1)[0] for line in first_launches],
        )

    def test_oa_mse_struct48_splits_conservative_and_aggressive_modules(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_STRUCT48")
        launcher = render_launcher("spaceborne_fewshot_struct48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_struct48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {4})
        self.assertEqual(payload["phase1_rows_expected"], 0)
        self.assertEqual(payload["phase2_rows_expected"], 48)
        self.assertEqual(payload["phase2_gpu_utilization_policy"]["queued_rows_per_gpu"], 6)
        self.assertTrue(all(c.oa_mse_anchor_density_gate for c in candidates))
        self.assertEqual(Counter(c.oa_mse_adapter_kind for c in candidates), {"low_rank": 24, "residual_mlp": 24})
        self.assertIn("--oa_mse_anchor_density_gate", launcher)
        self.assertIn("--oa_mse_adapter_kind residual_mlp", launcher)
        self.assertIn("anchor_density_one_class_gate", json.dumps(rows))
        self.assertTrue(all(item["parameters"]["oa_mse_anchor_density_gate"] for item in rows))
        self.assertEqual(Counter(item["category"] for item in rows), {"conservative": 24, "aggressive": 24})
        self.assertTrue(any(item["parameters"]["oa_mse_adapter_kind"] == "residual_mlp" for item in rows))
        self.assertTrue(any(item["parameters"]["anchor_density_gate_action"] == "reject" for item in rows))

    def test_oa_mse_simplified48_uses_simplified_leo_residual_channel(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_SIMPLIFIED48")
        launcher = render_launcher("spaceborne_fewshot_simplified48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_simplified48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual(payload["star_ground_channel_policy"]["default_impl"], "simplified_leo_residual")
        self.assertEqual(
            payload["star_ground_channel_policy"]["target_channel_scenarios"],
            "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        )
        self.assertIn("--star_ground_channel_impl simplified_leo_residual", launcher)
        self.assertIn("--target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak", launcher)
        self.assertIn("--target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak", launcher)
        self.assertNotIn("--target_new_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", launcher)
        self.assertTrue(all(c.star_ground_channel_impl == "simplified_leo_residual" for c in candidates))
        self.assertTrue(all(c.target_channel_scenarios == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak" for c in candidates))
        self.assertTrue(all(item["star_ground_channel_impl"] == "simplified_leo_residual" for item in rows))
        self.assertTrue(all(item["parameters"]["target_channel_scenarios"] == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak" for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_adapter_kind"] == "residual_mlp" for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_void_gate"] for item in rows))
        self.assertTrue(any(item["parameters"]["anchor_density_gate_action"] == "reject" for item in rows))

    def test_oa_mse_reghead48_enables_seen_new_registration_override_and_simplified_channel(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_REGHEAD48")
        launcher = render_launcher("spaceborne_fewshot_reghead48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_reghead48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {2})
        self.assertEqual(payload["phase1_rows_expected"], 0)
        self.assertEqual(payload["phase2_rows_expected"], 48)
        self.assertEqual(payload["phase2_gpu_utilization_policy"]["max_active_per_gpu"], 2)
        self.assertEqual(payload["star_ground_channel_policy"]["default_impl"], "simplified_leo_residual")
        self.assertTrue(all(c.star_ground_channel_impl == "simplified_leo_residual" for c in candidates))
        self.assertTrue(all(c.target_channel_scenarios == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak" for c in candidates))
        self.assertTrue(all(c.oa_mse_support_retention_guard for c in candidates))
        self.assertTrue(all(c.oa_mse_two_branch_background_guard for c in candidates))
        self.assertTrue(all(c.oa_mse_seen_new_registration_override for c in candidates))
        self.assertIn('STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"', launcher)
        self.assertIn("--oa_mse_seen_new_registration_override", launcher)
        self.assertIn("--seen_new_override_min_evidence_delta", launcher)
        self.assertIn("--star_ground_channel_impl simplified_leo_residual", launcher)
        self.assertIn("seen_new_registration_override", json.dumps(rows))
        self.assertTrue(all(item["parameters"]["oa_mse_seen_new_registration_override"] for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_support_retention_guard"] for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_two_branch_background_guard"] for item in rows))
        self.assertTrue(all("seen_new_registration_override" in item["score_table_required_columns"] for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_adapter_kind"] == "residual_mlp" for item in rows))
        self.assertTrue(any(item["parameters"]["multiproto_topk"] == 3 for item in rows))

    def test_oa_mse_geom48_adds_support_center_loss_and_multi_receiver_pool(self):
        from collections import Counter

        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_GEOM48")
        launcher = render_launcher("spaceborne_fewshot_geom48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_geom48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {2})
        self.assertEqual(payload["phase2_rows_expected"], 48)
        self.assertEqual(payload["star_ground_channel_policy"]["default_impl"], "simplified_leo_residual")
        self.assertEqual(
            set(payload["stage2_sample_protocol"]["target_receiver_pool_labels"]),
            {"20-1", "3-19", "7-14", "7-7", "8-8"},
        )
        self.assertEqual(
            set(item["target_receiver_labels"] for item in rows),
            {"20-1", "3-19", "7-14", "7-7", "8-8"},
        )
        self.assertTrue(all(item["parameters"]["oa_mse_support_center_ce_weight"] > 0 for item in rows))
        self.assertEqual({c.query_per_tx for c in candidates}, {30})
        self.assertEqual({c.target_old_query_per_tx for c in candidates}, {30})
        self.assertTrue(all(item["parameters"]["query_per_tx"] == 30 for item in rows))
        self.assertTrue(all(item["parameters"]["target_old_query_per_tx"] == 30 for item in rows))
        self.assertTrue(all(item["parameters"]["manytx_receiver_specific_query_cap"] == 30 for item in rows))
        self.assertTrue(all(item["support_center_geometry_registration"] for item in rows))
        self.assertTrue(all(item["star_ground_channel_impl"] == "simplified_leo_residual" for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_adapter_kind"] == "residual_mlp" for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_adapter_kind"] == "low_rank" for item in rows))
        self.assertIn("--oa_mse_support_center_ce_weight", launcher)
        self.assertIn("--support_center_temperature", launcher)
        self.assertIn("--support_center_margin", launcher)
        self.assertIn("--star_ground_channel_impl simplified_leo_residual", launcher)
        self.assertIn('--query_per_tx \\"30\\"', launcher)
        self.assertIn('--target_old_query_per_tx \\"30\\"', launcher)
        self.assertNotIn("--target_new_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", launcher)

    def test_oa_mse_triage48_splits_old_unknown_seen_new_failure_modes(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_TRIAGE48")
        launcher = render_launcher("spaceborne_fewshot_triage48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_triage48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(
            Counter(c.optimization_category for c in candidates),
            {"old_retention": 16, "unknown_boundary": 16, "seen_new_rescue": 16},
        )
        self.assertEqual({c.query_per_tx for c in candidates}, {30})
        self.assertEqual({c.target_old_query_per_tx for c in candidates}, {30})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {2})
        self.assertEqual(payload["star_ground_channel_policy"]["default_impl"], "simplified_leo_residual")
        self.assertEqual(
            set(item["target_receiver_labels"] for item in rows),
            {"20-1", "3-19", "7-14", "7-7", "8-8"},
        )
        self.assertTrue(all(item["parameters"]["manytx_receiver_specific_query_cap"] == 30 for item in rows))
        self.assertTrue(all("split_objective_triage" in item["fusion_inputs"] for item in rows))
        self.assertTrue(any(item["parameters"]["anchor_density_gate_action"] == "reject" for item in rows))
        self.assertTrue(any(item["parameters"]["anchor_density_gate_action"] == "uncertain" for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_void_gate"] for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_void_background_weight"] == 0.0 for item in rows))
        seen_new_rescue = [item for item in rows if item["parameters"]["optimization_category"] == "seen_new_rescue"]
        self.assertTrue(seen_new_rescue)
        self.assertTrue(all(item["parameters"]["target_new_support_per_tx"] >= 10 for item in seen_new_rescue))
        self.assertTrue(all(item["target_old_k"] == item["target_new_k"] for item in seen_new_rescue))
        self.assertTrue(all(item["parameters"]["oa_mse_seen_new_registration_override"] for item in seen_new_rescue))
        self.assertTrue(any(item["parameters"]["seen_new_override_max_background_score"] >= 0.90 for item in seen_new_rescue))
        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])
        self.assertIn("--star_ground_channel_impl simplified_leo_residual", launcher)
        self.assertIn("--oa_mse_void_gate", launcher)
        self.assertIn("--oa_mse_seen_new_registration_override", launcher)
        self.assertIn('--query_per_tx \\"30\\"', launcher)

    def test_oa_mse_looo48_adds_source_leave_one_old_out_boundary(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_LOOO48")
        launcher = render_launcher("spaceborne_fewshot_looo48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_looo48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual({c.query_per_tx for c in candidates}, {30})
        self.assertEqual({c.target_old_query_per_tx for c in candidates}, {30})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {2})
        self.assertTrue(all(c.oa_mse_source_looo_unknown_weight > 0 for c in candidates))
        self.assertTrue(any(c.oa_mse_source_looo_unknown_weight >= 0.28 for c in candidates))
        self.assertTrue(all("source_leave_one_old_out" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_source_looo_unknown_weight"] > 0 for item in rows))
        self.assertTrue(all(item["parameters"]["source_looo_max_samples_per_class"] >= 18 for item in rows))
        self.assertEqual(payload["star_ground_channel_policy"]["default_impl"], "simplified_leo_residual")
        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])
        self.assertIn("--oa_mse_source_looo_unknown_weight", launcher)
        self.assertIn("--source_looo_unknown_margin", launcher)
        self.assertIn("--source_looo_interclass_margin", launcher)

    def test_oa_mse_supportcv48_adds_query_free_support_cv_selector(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_SUPPORTCV48")
        launcher = render_launcher("spaceborne_fewshot_supportcv48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_supportcv48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual(
            Counter(c.oa_mse_adapter_selection_policy for c in candidates),
            {"support_cv_constrained": 24, "support_cv_risk_balanced": 24},
        )
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {2})
        self.assertEqual({c.star_ground_channel_impl for c in candidates}, {"simplified_leo_residual"})
        self.assertTrue(all(c.oa_mse_support_center_ce_weight > 0 for c in candidates))
        self.assertTrue(all(c.oa_mse_known_coverage_weight > 0 for c in candidates))
        self.assertTrue(all(not c.oa_mse_support_reconstruction_arbitration for c in candidates))
        self.assertTrue(all("support_leave_one_out_adapter_selector" in item["fusion_inputs"] for item in rows))
        self.assertTrue(
            all(
                item["parameters"]["oa_mse_adapter_selection_policy"]
                in {"support_cv_constrained", "support_cv_risk_balanced"}
                for item in rows
            )
        )
        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])
        self.assertIn("--oa_mse_adapter_selection_policy support_cv_constrained", launcher)
        self.assertIn("--oa_mse_adapter_selection_policy support_cv_risk_balanced", launcher)
        self.assertIn("--star_ground_channel_impl simplified_leo_residual", launcher)

    def test_oa_mse_bgcap48_adds_support_calibrated_background_cap(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_BGCAP48")
        launcher = render_launcher("spaceborne_fewshot_bgcap48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_bgcap48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertTrue(all(c.identity_consensus_support_background_cap for c in candidates))
        self.assertTrue(all(c.oa_mse_negative_anchor_weight > 0 for c in candidates))
        self.assertTrue(all(c.oa_mse_void_background_weight > 0 for c in candidates))
        self.assertTrue(all("support_calibrated_background_cap" in item["fusion_inputs"] for item in rows))
        self.assertTrue(
            all("identity_consensus_support_background_cap" in item["score_table_required_columns"] for item in rows)
        )
        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])
        self.assertIn("--identity_consensus_support_background_cap", launcher)
        self.assertIn("--identity_consensus_support_background_cap_quantile", launcher)
        self.assertIn("--oa_mse_negative_anchor_weight", launcher)

    def test_oa_mse_kret48_adds_support_neighborhood_known_retention(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_KRET48")
        launcher = render_launcher("spaceborne_fewshot_kret48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_kret48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {2})
        self.assertEqual({c.star_ground_channel_impl for c in candidates}, {"simplified_leo_residual"})
        self.assertTrue(all(c.pre_reject_support_neighborhood_retention for c in candidates))
        self.assertTrue(all(not c.identity_consensus_support_background_cap for c in candidates))
        self.assertTrue(all("support_neighborhood_known_retention" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all(item["parameters"]["pre_reject_support_neighborhood_retention"] for item in rows))
        self.assertTrue(
            all("pre_reject_arbitration_support_retention" in item["score_table_required_columns"] for item in rows)
        )
        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])
        self.assertIn("--pre_reject_support_neighborhood_retention", launcher)
        self.assertIn("--pre_reject_support_retention_max_background_score", launcher)
        self.assertIn("--star_ground_channel_impl simplified_leo_residual", launcher)

    def test_oa_mse_riskret48_constrains_retention_with_source_risk(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_RISKRET48")
        launcher = render_launcher("spaceborne_fewshot_riskret48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_riskret48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertTrue(all(c.pre_reject_support_neighborhood_retention for c in candidates))
        self.assertTrue(all(c.pre_reject_support_retention_require_source_looo_pass for c in candidates))
        self.assertEqual({c.pre_reject_support_retention_source_looo_max_failures for c in candidates}, {0, 1})
        self.assertTrue(all(c.oa_mse_source_looo_risk_arbitration for c in candidates))
        self.assertTrue(all("source_risk_constrained_support_retention" in item["fusion_inputs"] for item in rows))
        self.assertTrue(
            all(
                item["parameters"]["pre_reject_support_retention_require_source_looo_pass"]
                for item in rows
            )
        )
        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])
        self.assertIn("--pre_reject_support_retention_require_source_looo_pass", launcher)
        self.assertIn("--pre_reject_support_retention_source_looo_max_failures", launcher)

    def test_oa_mse_constrain48_adds_known_coverage_constraints(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_CONSTRAIN48")
        launcher = render_launcher("spaceborne_fewshot_constrain48_test", candidates)
        payload = matrix_payload("spaceborne_fewshot_constrain48_test", candidates)
        rows = payload["candidates"]

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual({c.query_per_tx for c in candidates}, {30})
        self.assertEqual({c.target_old_query_per_tx for c in candidates}, {30})
        self.assertEqual({c.stage2_max_active_per_gpu for c in candidates}, {2})
        self.assertEqual({c.oa_mse_adapter_selection_policy for c in candidates}, {"constrained_retention_risk"})
        self.assertTrue(all(c.oa_mse_known_coverage_weight > 0 for c in candidates))
        self.assertTrue(all(c.oa_mse_source_looo_unknown_weight > 0 for c in candidates))
        self.assertTrue(all(c.old_acc_target >= 0.95 for c in candidates))
        self.assertTrue(all(c.seen_new_acc_target >= 0.80 for c in candidates))
        self.assertEqual(payload["star_ground_channel_policy"]["default_impl"], "simplified_leo_residual")
        self.assertEqual(
            set(item["target_receiver_labels"] for item in rows),
            {"20-1", "3-19", "7-14", "7-7", "8-8"},
        )
        self.assertTrue(all("known_coverage_margin_loss" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_known_coverage_weight"] > 0 for item in rows))
        self.assertTrue(all(item["parameters"]["oa_mse_adapter_selection_policy"] == "constrained_retention_risk" for item in rows))
        self.assertTrue(all(item["parameters"]["anchor_density_gate_action"] == "uncertain" for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_void_gate"] for item in rows))
        self.assertTrue(any(item["parameters"]["oa_mse_void_background_weight"] == 0.0 for item in rows))
        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])
        self.assertIn("--oa_mse_known_coverage_weight", launcher)
        self.assertIn("--oa_mse_adapter_selection_policy constrained_retention_risk", launcher)
        self.assertIn("--star_ground_channel_impl simplified_leo_residual", launcher)

    def test_oa_mse_payload_is_launchable_after_manytx_label_availability_repair(self):
        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload

        payload = matrix_payload("spaceborne_fewshot_oa_mse_test", make_candidates(plan="OA_MSE_CARD3"))
        sample_protocol = {
            "old_tx_ids": [0, 1, 2, 3, 4, 5],
            "cen51_train_receiver_ids": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
            "source_receiver_ids": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
            "source_receiver_labels": ["1-1", "1-19", "14-7", "18-2", "19-2", "2-1", "2-19"],
            "recommended_k_shot_anchors": [1, 2, 5, 10, 15, 20, 50],
            "few_shot_upper_bound": 20,
        }
        result = validate(payload["candidates"], expected_count=3, sample_protocol=sample_protocol)
        phase2 = result["launchability_summary"]["by_lane"]["phase2_spaceborne_fsl"]

        self.assertEqual("PASS", result["verdict"], result["issues"])
        self.assertEqual("LANE_HAS_LAUNCHABLE_ROWS", phase2["runner_readiness"])
        self.assertEqual(3, phase2["launchable"])
        self.assertEqual(0, phase2["local_patch_required"])

    def test_h06_oldrelax48_requires_terminal_old_primary_consensus(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_H06_OLDRELAX48")
        payload = matrix_payload("spaceborne_fewshot_h06_oldrelax48_test", candidates)
        rows = payload["candidates"]
        launcher = render_launcher("spaceborne_fewshot_h06_oldrelax48_test", candidates)

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"conservative": 24, "aggressive": 24})
        self.assertEqual(Counter(c.target_old_support_per_tx for c in candidates), {5: 40, 10: 8})
        self.assertEqual({c.target_new_support_per_tx for c in candidates}, {0})
        self.assertEqual({c.new_tx_ids for c in candidates}, {"__NONE__"})
        self.assertTrue(all("CEN51_R04_H06_LOW_PROB_HYBRID_R010/latest_model.pth" in c.ground_model_default_ckpt for c in candidates))
        self.assertTrue(all(c.oa_mse_old_primary_gate for c in candidates))
        self.assertTrue(all(c.old_primary_require_soft_mixture for c in candidates))
        self.assertTrue(all(c.old_primary_require_support_knn for c in candidates))
        self.assertTrue(all(c.oa_mse_class_envelope_gate for c in candidates))
        self.assertTrue(all(c.old_primary_require_class_envelope for c in candidates))
        self.assertTrue(all(c.retention_rescue_candidate_only for c in candidates))
        self.assertTrue(all(c.old_primary_promote_rescue_candidates for c in candidates))
        self.assertEqual(Counter(c.old_primary_unknown_veto_min_sources for c in candidates), {2: 24, 1: 24})
        self.assertTrue(all(not c.oa_mse_support_conformal_arbitration for c in candidates))
        self.assertTrue(all(not c.oa_mse_support_reconstruction_arbitration for c in candidates))
        self.assertTrue(all("class_envelope_required_old_primary_consensus" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all("retention_rescue_candidate_only" in item["fusion_inputs"] for item in rows))
        self.assertEqual({item["target_new_tx_labels"] for item in rows}, {""})
        self.assertEqual({item["target_new_leo_query"] for item in rows}, {"not_applicable_old_unknown_only"})
        self.assertIn("OA_MSE_H06_OLDRELAX48", launcher)
        self.assertIn("--old_primary_unknown_veto_min_sources", launcher)
        self.assertIn("--retention_rescue_candidate_only", launcher)
        self.assertIn("--old_primary_promote_rescue_candidates", launcher)

        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])

    def test_h06_oldqual48_constructs_support_quality_repair_without_seen_new(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_H06_OLDQUAL48")
        payload = matrix_payload("spaceborne_fewshot_h06_oldqual48_test", candidates)
        rows = payload["candidates"]
        launcher = render_launcher("spaceborne_fewshot_h06_oldqual48_test", candidates)

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"support_quality": 24, "prototype_geometry": 24})
        self.assertEqual({c.target_new_support_per_tx for c in candidates}, {0})
        self.assertEqual({c.new_tx_ids for c in candidates}, {"__NONE__"})
        self.assertTrue(all(c.oa_mse_support_center_ce_weight > 0 for c in candidates))
        self.assertTrue(all(c.oa_mse_multiproto_score for c in candidates))
        self.assertTrue(all(c.oa_mse_mixture_consistency_gate for c in candidates))
        self.assertTrue(all(c.oa_mse_support_conformal_arbitration for c in candidates))
        self.assertTrue(all(c.oa_mse_support_reconstruction_arbitration for c in candidates))
        self.assertTrue(all(not c.oa_mse_old_primary_gate for c in candidates))
        self.assertTrue(all(not c.old_primary_promote_rescue_candidates for c in candidates))
        self.assertTrue(all("support_quality_prototype_construction" in item["fusion_inputs"] for item in rows))
        self.assertEqual({item["target_new_tx_labels"] for item in rows}, {""})
        self.assertEqual({item["target_new_leo_query"] for item in rows}, {"not_applicable_old_unknown_only"})
        self.assertIn("OA_MSE_H06_OLDQUAL48", launcher)
        self.assertIn("--oa_mse_support_center_ce_weight", launcher)
        self.assertIn("--oa_mse_support_conformal_arbitration", launcher)
        self.assertIn("--oa_mse_support_reconstruction_arbitration", launcher)

        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])

    def test_h06_oldrisk48_targets_query_free_background_risk_without_seen_new(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_H06_OLDRISK48")
        payload = matrix_payload("spaceborne_fewshot_h06_oldrisk48_test", candidates)
        rows = payload["candidates"]
        launcher = render_launcher("spaceborne_fewshot_h06_oldrisk48_test", candidates)

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"query_free_background_risk": 24, "unknown_separability": 24})
        self.assertEqual(
            Counter(c.oa_mse_adapter_selection_policy for c in candidates),
            {"constrained_retention_risk": 24, "identity_preserving_risk": 24},
        )
        self.assertEqual({c.target_new_support_per_tx for c in candidates}, {0})
        self.assertEqual({c.new_tx_ids for c in candidates}, {"__NONE__"})
        self.assertTrue(all(c.oa_mse_two_branch_background_guard for c in candidates))
        self.assertTrue(all(c.oa_mse_pre_reject_defer_arbitration for c in candidates))
        self.assertTrue(all(c.oa_mse_source_looo_risk_arbitration for c in candidates))
        self.assertTrue(all(c.oa_mse_old_unknown_acceptance_guard for c in candidates))
        self.assertTrue(all(not c.oa_mse_old_primary_gate for c in candidates))
        self.assertTrue(all("query_free_background_risk" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all("unknown_score_joint_veto" in item["fusion_inputs"] for item in rows))
        self.assertEqual({item["target_new_tx_labels"] for item in rows}, {""})
        self.assertEqual({item["target_new_leo_query"] for item in rows}, {"not_applicable_old_unknown_only"})
        self.assertIn("OA_MSE_H06_OLDRISK48", launcher)
        self.assertIn("--oa_mse_two_branch_background_guard", launcher)
        self.assertIn("--oa_mse_old_unknown_acceptance_guard", launcher)
        self.assertIn("--oa_mse_source_looo_risk_arbitration", launcher)
        self.assertIn("--oa_mse_adapter_selection_policy constrained_retention_risk", launcher)
        self.assertIn("--oa_mse_adapter_selection_policy identity_preserving_risk", launcher)

        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])

    def test_h06_oldfuse48_combines_oldqual_oldrisk_and_calibrates_rollback(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_H06_OLDFUSE48")
        payload = matrix_payload("spaceborne_fewshot_h06_oldfuse48_test", candidates)
        rows = payload["candidates"]
        launcher = render_launcher("spaceborne_fewshot_h06_oldfuse48_test", candidates)

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(Counter(c.optimization_category for c in candidates), {"oldqual_oldrisk_fusion": 24, "rollback_calibration": 24})
        self.assertEqual(
            Counter(c.oa_mse_adapter_selection_policy for c in candidates),
            {"constrained_retention_risk": 24, "identity_preserving_risk": 24},
        )
        self.assertEqual({c.target_new_support_per_tx for c in candidates}, {0})
        self.assertEqual({c.new_tx_ids for c in candidates}, {"__NONE__"})
        self.assertTrue(all(c.oa_mse_support_center_ce_weight > 0 for c in candidates))
        self.assertTrue(all(c.oa_mse_multiproto_score for c in candidates))
        self.assertTrue(all(c.oa_mse_mixture_consistency_gate for c in candidates))
        self.assertTrue(all(c.oa_mse_two_branch_background_guard for c in candidates))
        self.assertTrue(all(c.oa_mse_pre_reject_defer_arbitration for c in candidates))
        self.assertTrue(all(c.oa_mse_source_looo_risk_arbitration for c in candidates))
        self.assertTrue(all(c.oa_mse_old_unknown_acceptance_guard for c in candidates))
        self.assertTrue(all(not c.oa_mse_old_primary_gate for c in candidates))
        self.assertTrue(all("oldqual_oldrisk_fusion" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all("rollback_calibration" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all("H06_OLDFUSE48" in item["candidate_id"] for item in rows))
        self.assertTrue(all("h06_oldqual_oldrisk_fusion_rollback_calibration" in item["update_module"] for item in rows))
        self.assertEqual({item["target_new_tx_labels"] for item in rows}, {""})
        self.assertEqual({item["target_new_leo_query"] for item in rows}, {"not_applicable_old_unknown_only"})
        self.assertIn("OA_MSE_H06_OLDFUSE48", launcher)
        self.assertIn("--oa_mse_two_branch_background_guard", launcher)
        self.assertIn("--oa_mse_old_unknown_acceptance_guard", launcher)
        self.assertIn("--oa_mse_support_center_ce_weight", launcher)
        self.assertIn("--pre_reject_support_neighborhood_retention", launcher)

        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])

    def test_h06_rollsafe48_repairs_rollback_safe_old_retention_without_seen_new(self):
        from collections import Counter

        from optimizer_validate_matrix import validate
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload, render_launcher

        candidates = make_candidates(plan="OA_MSE_H06_ROLLSAFE48")
        payload = matrix_payload("spaceborne_fewshot_h06_rollsafe48_test", candidates)
        rows = payload["candidates"]
        launcher = render_launcher("spaceborne_fewshot_h06_rollsafe48_test", candidates)

        self.assertEqual(48, len(candidates))
        self.assertEqual(Counter(c.gpu for c in candidates), {gpu: 6 for gpu in range(8)})
        self.assertEqual(
            Counter(c.optimization_category for c in candidates),
            {"rollback_safe_retention": 24, "deployment_gate_rescue": 24},
        )
        self.assertEqual(
            Counter(c.oa_mse_adapter_selection_policy for c in candidates),
            {"constrained_retention_risk": 24, "identity_preserving_risk": 24},
        )
        self.assertEqual({c.target_new_support_per_tx for c in candidates}, {0})
        self.assertEqual({c.new_tx_ids for c in candidates}, {"__NONE__"})
        self.assertTrue(all(c.oa_mse_support_center_ce_weight > 0 for c in candidates))
        self.assertTrue(all(c.oa_mse_multiproto_score for c in candidates))
        self.assertTrue(all(c.oa_mse_mixture_consistency_gate for c in candidates))
        self.assertTrue(all(c.oa_mse_two_branch_background_guard for c in candidates))
        self.assertTrue(all(c.oa_mse_pre_reject_defer_arbitration for c in candidates))
        self.assertTrue(all(c.pre_reject_support_neighborhood_retention for c in candidates))
        self.assertTrue(all(c.oa_mse_retention_rescue_gate for c in candidates))
        self.assertTrue(all(c.retention_rescue_candidate_only for c in candidates))
        self.assertTrue(all(c.oa_mse_source_looo_risk_arbitration for c in candidates))
        self.assertTrue(all(c.oa_mse_old_unknown_acceptance_guard for c in candidates))
        self.assertTrue(all(c.oa_mse_support_retention_guard for c in candidates))
        self.assertTrue(all(not c.oa_mse_old_primary_gate for c in candidates))
        self.assertTrue(all("rollback_safe_retention" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all("defer_first_deployment_gate" in item["fusion_inputs"] for item in rows))
        self.assertTrue(all("H06_ROLLSAFE48" in item["candidate_id"] for item in rows))
        self.assertTrue(all("h06_rollback_safe_retention" in item["update_module"] for item in rows))
        self.assertEqual({item["target_new_tx_labels"] for item in rows}, {""})
        self.assertEqual({item["target_new_leo_query"] for item in rows}, {"not_applicable_old_unknown_only"})
        self.assertIn("OA_MSE_H06_ROLLSAFE48", launcher)
        self.assertIn("--oa_mse_two_branch_background_guard", launcher)
        self.assertIn("--oa_mse_retention_rescue_gate", launcher)
        self.assertIn("--retention_rescue_candidate_only", launcher)
        self.assertIn("--pre_reject_support_neighborhood_retention", launcher)
        self.assertIn("--oa_mse_old_unknown_acceptance_guard", launcher)

        result = validate(rows, expected_count=48, matrix_root=payload, launcher_text=launcher)
        self.assertEqual("PASS", result["verdict"], result["issues"])

    def test_oa_mse_payload_carries_resolved_manytx_tx_and_rx_labels(self):
        from spaceborne_fewshot_da_matrix import make_candidates, matrix_payload

        payload = matrix_payload("spaceborne_fewshot_oa_mse_test", make_candidates(plan="OA_MSE_CARD3"))
        rows = payload["candidates"]

        self.assertEqual(3, len(rows))
        for item in rows:
            self.assertIn("target_receiver_labels", item)
            self.assertIn("source_receiver_labels", item)
            self.assertIn("cen51_train_receiver_labels", item)
            self.assertIn("manytx_target_rx_index", item)
            self.assertIn("target_new_tx_labels", item)
            self.assertIn("unknown_tx_labels", item)
            self.assertIn("target_unknown_tx_labels", item)
            self.assertEqual("1-1,1-19,14-7,18-2,19-2,2-1,2-19", item["source_receiver_labels"])
            self.assertEqual("1-1,1-19,14-7,18-2,19-2,2-1,2-19", item["cen51_train_receiver_labels"])
            self.assertEqual("20-1", item["target_receiver_labels"])
            self.assertEqual("10", item["manytx_target_rx_index"])
            self.assertEqual("rx7", item["target_receiver_ids"])
            self.assertEqual("1-16,1-18", item["target_new_tx_labels"])
            self.assertEqual("10-1,10-10", item["unknown_tx_labels"])
            self.assertEqual("10-1,10-10", item["target_unknown_tx_labels"])
            self.assertNotIn("UNRESOLVED", item["target_new_tx_labels"])
            self.assertNotIn("UNRESOLVED", item["unknown_tx_labels"])


if __name__ == "__main__":
    unittest.main()
