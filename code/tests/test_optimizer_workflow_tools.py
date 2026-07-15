import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import optimizer_validate_matrix as ovm
from optimizer_workflow_lib import load_standard_summary
from optimizer_state_current_view import current_view
from optimizer_validate_matrix import load_stage2_sample_protocol, validate


def _write_minimal_preflight_controls(project_root: Path) -> None:
    (project_root / "AGENTS.md").write_text("AGENTS safety rules\n", encoding="utf-8")
    (project_root / "项目.md").write_text("CVS project protocol\n", encoding="utf-8")
    tools_dir = project_root / "tools"
    prompt_dir = (
        project_root
        / "automation_reports"
        / "CV-SincNet"
        / "automation_prompt_backups"
        / "20260615_001820_stage2_closed_loop_v4"
    )
    state_dir = project_root / "automation_reports" / "CV-SincNet"
    tools_dir.mkdir(parents=True)
    prompt_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "optimizer_control_manifest.md").write_text(
        "\n".join(
            [
                "# CV-SincNet Stage2 Automation Control Manifest",
                "Load required control files in this order:",
                "1. `AGENTS.md`",
                "2. `项目.md`",
                "3. `tools/optimizer_control_manifest.md`",
                "4. `automation_reports/CV-SincNet/automation_prompt_backups/20260615_001820_stage2_closed_loop_v4/stage2_prompt.md`",
                "5. `tools/optimizer_workflow_contract.md`",
                "6. `automation_reports/CV-SincNet/stage2_optimizer_state.json`",
            ]
        ),
        encoding="utf-8",
    )
    (prompt_dir / "stage2_prompt.md").write_text("active optimizer prompt\n", encoding="utf-8")
    (tools_dir / "optimizer_workflow_contract.md").write_text("workflow contract\n", encoding="utf-8")
    (state_dir / "stage2_optimizer_state.json").write_text(
        json.dumps(
            {
                "last_updated_local": "2026-06-24T18:00:00+08:00",
                "lane_monitor_policy": {"basis": "process/CWD/cmdline/GPU-only"},
                "lane_capacity_policy": {"phase2_max_active_per_gpu": 2},
                "idle_lane_execution_policy": {"status": "REPAIR_UNTIL_RUNNER_EXECUTES"},
                "stage2_sample_protocol": {
                    "old_tx_ids": [0, 1, 2, 3, 4, 5],
                    "cen51_train_receiver_ids": [0, 1, 2, 3, 4, 5, 6],
                },
                "latest_two_lane_monitor_result": {"phase1_monitor_state": 1, "phase2_monitor_state": 1},
                "objective_changelog": [{"audit": "only"}],
            }
        ),
        encoding="utf-8",
    )


def _write_preflight_matrix(project_root: Path, duplicate_command_hash: bool = False) -> tuple[Path, Path]:
    run_id = "stage2_spaceborne_preflight_test"
    items = [
        _stage2_item(0, "A", 0),
        _stage2_item(1, "B", 33),
    ]
    for idx, item in enumerate(items):
        item["candidate_id"] = f"{run_id}_C{idx}"
        item["estimated_run_path"] = f"/home/project/runs/{run_id}/{item['candidate_id']}"
        item["estimated_log_path"] = f"/home/project/logs/{run_id}/{item['candidate_id']}.out"
        item["registry_key"] = f"{run_id}:{item['candidate_id']}"
        item["command_hash"] = "duplicate" if duplicate_command_hash else f"hash-{idx}"
        item["exact_command"] = f"RUN_ID={run_id} python code/eval_spaceborne_fewshot.py --candidate {idx}"
    matrix_path = project_root / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_matrix_v1",
                "expected_count": 2,
                "n607_run_id": run_id,
                "candidates": items,
            }
        ),
        encoding="utf-8",
    )
    launcher_path = project_root / "launcher.sh"
    launcher_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'ROOT="${ROOT:-/home/project}"',
                f'RUN_ID="${{RUN_ID:-{run_id}}}"',
                'RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"',
                'LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"',
                'PHASE2_LOCAL_PATCH_REQUIRED="${PHASE2_LOCAL_PATCH_REQUIRED:-0}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix_path, launcher_path


def test_collect_results_normalizes_legacy_batch_best(tmp_path):
    source = tmp_path / "centralized_cen54_postrun_summary.json"
    source.write_text(
        json.dumps(
            {
                "batch": "CEN54",
                "lane": "centralized",
                "finished_count": 1,
                "hard_error_count": 0,
                "evidence_hash": "abc",
                "batch_bests": {
                    "clean": {
                        "candidate": "CEN54_R01",
                        "run_name": "CEN54_R01_demo",
                        "strict_udu": 84.0,
                        "sat": {"clear_leo": 48.0, "storm_mp": 41.0},
                        "receiver_floor": 73.0,
                        "risk_adjusted": 71.0,
                    }
                },
                "items": [
                    {
                        "candidate_id": "CEN54_R01",
                        "run_name": "CEN54_R01_demo",
                        "finished": True,
                        "modes": {
                            "PRIMARY": {
                                "strict_udu": 84.0,
                                "sat": {"clear_leo": 48.0, "storm_mp": 41.0},
                                "receiver_floor": 73.0,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-16",
    )
    summary = load_standard_summary(source)
    assert summary["schema"] == "optimizer_batch_summary_v1"
    assert summary["batch"] == "CEN54"
    assert summary["items"][0]["views"]["PRIMARY"]["sat_floor"] == 41.0
    assert summary["batch_bests"]["clean"]["metrics"]["receiver_floor"] == 73.0


def test_validate_matrix_accepts_4_plus_4_and_fed_constraints():
    items = []
    for idx in range(1, 5):
        items.append(
            {
                "candidate_id": f"FED21_R0{idx}",
                "lane": "federated_vmb",
                "parent_run": "FED20_R03",
                "lineage": "FED20_R03",
                "route_signature": "fed_vmb",
                "retirement_status": "not_retired",
                "invalidity_status": "not_invalidated",
                "principle_rejection_ref": "none",
                "experimental_rejection_ref": "none",
                "retirement_evidence_count": 0,
                "retirement_evidence_refs": [],
                "replacement_reason": "test",
                "hypothesis": "hold clean",
                "control": "control",
                "key_changes": ["fedprox"],
                "parameters": {
                    "--wisig_train_ratio": "0.1",
                    "--epochs": "200",
                    "--fl_rounds": "200",
                    "--fl_client_key": "receiver",
                },
                "gpu": idx - 1,
                "estimated_run_path": "/runs/x",
                "estimated_log_path": "/logs/x.out",
                "cross_domain_target_metric": "strict",
                "satellite_channel_target_metric": "sat_floor",
                "allowed_tradeoff": "bounded",
                "must_not_regress_floor": "floors",
                "comparability_status": "COMPARABLE",
                "expected_failure_signals": "none",
                "fallback_or_alternative": "blocked",
                "exact_command": "--wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200 --fl_client_key receiver",
                "launchability_status": "READY",
            }
        )
    for idx in range(5, 9):
        item = dict(items[0])
        item["candidate_id"] = f"FED21_A0{idx}"
        item["gpu"] = idx - 1
        items.append(item)
    result = validate(items, 8)
    assert result["verdict"] == "PASS"
    assert result["categories"]["conservative"] == 4
    assert result["categories"]["aggressive"] == 4
    assert result["categories"]["unknown"] == 0


def _stage2_item(gpu_idx, slot_letter, idx):
    category = "conservative" if idx < 32 else "aggressive"
    candidate_id = f"S2N99_GPU{gpu_idx}_{slot_letter}_DIAG_{idx:02d}"
    return {
        "candidate_id": candidate_id,
        "category": category,
        "slot": f"GPU{gpu_idx}/{slot_letter}",
        "lane": "phase2_spaceborne_fsl",
        "stage2_mode": "Stage2-A_zero_label_deploy",
        "parent_run": "stage2_parent",
        "lineage": "stage2_parent",
        "route_signature": "stage2_diag",
        "retirement_status": "not_retired",
        "invalidity_status": "not_invalidated",
        "principle_rejection_ref": "none",
        "experimental_rejection_ref": "none",
        "retirement_evidence_count": 0,
        "retirement_evidence_refs": [],
        "replacement_reason": "test",
        "hypothesis": "test",
        "control": "control",
        "key_changes": "none",
        "parameters": {},
        "gpu": f"GPU{gpu_idx}",
        "estimated_run_path": "/runs/x",
        "estimated_log_path": "/logs/x.out",
        "cross_domain_target_metric": "satellite",
        "satellite_channel_target_metric": "satellite",
        "allowed_tradeoff": "bounded",
        "must_not_regress_floor": "floors",
        "comparability_status": "COMPARABLE",
        "expected_failure_signals": "none",
        "fallback_or_alternative": "blocked",
        "exact_command": "bash launch.sh",
        "launchability_status": "READY",
        "source_tx_ids": "0,1,2,3,4,5",
        "new_tx_ids": "6,7",
        "unknown_tx_ids": "8,9",
        "target_old_tx_ids": "0,1,2,3,4,5",
        "target_new_tx_ids": "6,7",
        "cen51_train_rxs": "rx0,rx1,rx2,rx3,rx4,rx5,rx6",
        "target_receiver_ids": "rx7",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "satellite_seed": 713101,
        "clean_view_role": "not_accessible_in_phase2",
        "dataset_role": "terrestrial_proxy",
        "evidence_level": "receiver_x_transmitter_proxy_stress",
        "deployment_success_claim_allowed": False,
        "support_query_split_verified": True,
        "receiver_disjoint_verified": True,
        "tx_split_disjoint_verified": True,
        "old_support_query_split": "Stage2-A support=empty; Stage2-B support=target_old_fewshot; query=target_old_rx7",
        "new_support_query_split": "Stage2-A/B support=empty; Stage2-C support=target_new_fewshot only when enrolled; query=target_new_rx7",
        "target_old_leo_query": "ManySig TX 0-5 received by target receiver rx7",
        "target_new_leo_query": "non-ManySig TX 6,7 received by target receiver rx7",
        "unknown_leo_query": "unknown TX 8,9 received by target receiver rx7",
        "cen51_base_checkpoint_or_config": "best_strict_udu_model.pth",
        "threshold_selection_label_scope": "source_old_and_allowed_support_only; unknown_query_eval_only",
        "unknown_query_eval_only": True,
        "target_new_query_not_threshold_fit": True,
        "unknown_FAR_target": 0.05,
        "onboard_low_compute_training": True,
        "compute_budget_profile": "feature_level_low_rank_adapter_rank2_max40steps_no_backbone_update",
        "adapter_trainable_params_cap": 4096,
        "max_adapt_steps": 40,
        "old_acc_target": 0.90,
        "seen_new_acc_target": 0.75,
        "weibull_evt_required": True,
        "target_adapter_required": True,
        "pseudo_unknown_energy_required": True,
        "seen_new_evidence_gate_required": True,
        "seen_new_anchor_gate_required": True,
        "siamese_verifier_required": True,
        "accepted_only_online_update_required": True,
        "oa_mse_onboard_adaptation_bundle": "weibull_evt+target_adapter+pseudo_unknown_energy+seen_new_evidence_gate+seen_new_anchor_gate+siamese_verifier+accepted_only_online_update+stage2_receiver_domain",
        "registry_key": f"test_run:{candidate_id}",
        "command_hash": f"hash-{idx}",
    }


def _phase1_training_item(gpu_idx, idx):
    item = _stage2_item(gpu_idx, "A", idx)
    item.update(
        {
            "candidate_id": f"S2N99_GPU{gpu_idx}_A_SAFE_SSDG_TRAIN_{idx:02d}",
            "category": "conservative" if idx < 32 else "aggressive",
            "lane": "phase1_ground_dg",
            "clean_view_role": "control_only",
            "phase_axis": "Phase1-GroundDG",
            "stage2_mode": "NOT_APPLICABLE",
            "protocol": "Safe-SSDG-CVS-R01",
            "route_family": "SAFE_SSDG_CVS_R01",
            "route_signature": "safe_ssdg_cvs_r01_training",
            "ground_dg_claim_scope": "full_200e_training_candidate_not_completion_claim",
            "source_ssl_split": "0.1L/0.7U/0.2Val",
            "no_target_receiver_in_training": True,
            "cen51_parent_run_or_control": "CEN51_R04_BEST",
            "phase1_non_regression_target": "matched_CEN51_R04",
            "optimization_target": "exceed_matched_CEN51_R04",
            "target_lift_over_cen51": "sat_mean_5_delta_pp>0; sat_floor_5_delta_pp>0; overall_delta_pp>=0; strict_udu_delta_pp>=0; receiver_floor_delta_pp>=0",
            "satellite_channel_primary_metric": True,
            "satellite_channel_lift_target": "deployment-primary star-ground lift: sat_mean_5_delta_pp>0; sat_floor_5_delta_pp>0",
            "phase1_star_ground_aug_policy": "default_on",
            "phase1_star_ground_aug_default_enabled": True,
            "phase1_star_ground_aug_route_family": "CVS-SAT-PAIC",
            "phase1_star_ground_aug_mode": "concat_sat_ce_only_paic_curriculum",
            "use_concat_sat_channel_aug": True,
            "concat_sat_ce_only": True,
            "use_sat_consistency": True,
            "sat_view_schedule": "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
            "sat_cons_start_epoch": 60,
            "lambda_sat_cls": 1.0,
            "lambda_sat_cons": 0.03,
            "star_ground_aug_exploration_axis": "CVS-SAT-PAIC curriculum CE-only satellite branch plus late weak z_id consistency; compare against explicit CEN51 refresh controls only.",
            "CEN51_COMPARABLE": True,
            "pseudo_precision_audit_target": 0.95,
            "pseudo_coverage_is_risk_metric": True,
            "forbid_meta_learning_dg_mainline": True,
            "must_not_regress_floor": "overall>=88.57; strict_udu>=84.87; receiver_floor>=79.53; sat_mean_5>=46.564; sat_floor_5>=41.52",
            "parameters": {
                "epochs": 200,
                "phase1_candidate": "Safe-SSDG-CVS-R01",
                "entrypoint": "python ${ROOT}/code/SSDG/train_ssdg.py",
                "split_mode": "tx_rx_day_1_7_2",
                "labeled_ratio": 0.1,
                "unlabeled_ratio": 0.7,
                "source_val_ratio": 0.2,
                "phase1_star_ground_aug_default_enabled": True,
                "phase1_star_ground_aug_route_family": "CVS-SAT-PAIC",
                "use_concat_sat_channel_aug": True,
                "concat_sat_ce_only": True,
                "use_sat_consistency": True,
                "sat_view_schedule": "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
            },
            "exact_command": (
                "cd /home/szu2070436088/2510044040/CV-SincNet && "
                f"PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet "
                f"CUDA_VISIBLE_DEVICES={gpu_idx} python -u /home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py "
                "--wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl "
                "--split_mode tx_rx_day_1_7_2 --labeled_ratio 0.1 --unlabeled_ratio 0.7 --source_val_ratio 0.2 "
                f"--output_dir /home/szu2070436088/2510044040/CV-SincNet/runs/test_phase1/{gpu_idx}_{idx} "
                "--epochs 200 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only "
                "--sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak "
                "--sat_view_schedule '1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak' "
                "--sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --device cuda:0"
            ),
            "launchability_status": "DEFERRED_RETRY_CAPACITY_WITH_EXACT_FULL_200E_TRAINING_COMMAND",
        }
    )
    return item


def _phase1_protocol_item(gpu_idx, idx):
    item = _stage2_item(gpu_idx, "A", idx)
    item.update(
        {
            "candidate_id": f"S2N99_GPU{gpu_idx}_A_MSSL_PROTOCOL_{idx:02d}",
            "lane": "phase1_ground_dg",
            "phase_axis": "Phase1-GroundDG",
            "protocol": "Meta-SSL-CVS-R04",
            "route_signature": "meta_ssl_cvs_r04_protocol_smoke",
            "ground_dg_claim_scope": "post_200e_protocol_regression_only_not_new_completion_claim",
            "parameters": {
                "full_training_status": "post_200e_single_seed_partial; protocol_regression_only_in_this_lightweight_queue",
                "epochs": 200,
            },
            "exact_command": "bash code/scripts/launch_stage2_optimizer_next64.sh # row runs --meta_ssl_protocol_check_only",
            "launchability_status": "LAUNCHABLE_META_SSL_POST_200E_PROTOCOL_REGRESSION",
        }
    )
    return item


def _current_run_bound_64_items(n607_run_id):
    items = []
    idx = 0
    root = "/home/szu2070436088/2510044040/CV-SincNet"
    launcher = f"{root}/code/scripts/launch_stage2_optimizer_20260622_999999_next64zz.sh"
    for gpu_idx in range(8):
        phase1 = _phase1_training_item(gpu_idx, idx)
        items.append(phase1)
        idx += 1
        for slot_letter in "BCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1

    for item in items:
        candidate_id = item["candidate_id"]
        item["estimated_run_path"] = f"{root}/runs/{n607_run_id}/{candidate_id}"
        item["estimated_log_path"] = f"{root}/logs/{n607_run_id}/{candidate_id}.out"
        item["registry_key"] = f"{n607_run_id}:{candidate_id}"
        if item["lane"] == "phase2_spaceborne_fsl":
            item["exact_command"] = (
                f"cd {root} && RUN_ID={n607_run_id} PHASE2_LOCAL_PATCH_REQUIRED=0 "
                f"bash {launcher} # {candidate_id}"
            )
    return items


def test_validate_matrix_requires_canonical_64_stage2_gpu_slots():
    items = []
    idx = 0
    for gpu_idx in range(8):
        phase1 = _phase1_training_item(gpu_idx, idx)
        phase1["launchability_status"] = "phase1_launchable_training_candidate_full_200e_pending_remote_gates"
        items.append(phase1)
        idx += 1
        for slot_letter in "BCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1

    result = validate(items, 64)

    assert result["verdict"] == "PASS"


def test_validate_matrix_rejects_phase1_safe_ssdg_local_verify_defer():
    items = []
    idx = 0
    for gpu_idx in range(8):
        phase1 = _phase1_training_item(gpu_idx, idx)
        phase1["candidate_id"] = f"S2N99_GPU{gpu_idx}_A_SAFE_SSDG_LOCAL_VERIFY_{idx:02d}"
        phase1["runtime_class"] = "phase1_training_deferred_local_verify"
        phase1["comparability_status"] = "PENDING_LOCAL_SAFE_SSDG_LAUNCHER_VERIFY_NOT_SERVER_LAUNCHABLE_THIS_TURN"
        phase1["launchability_status"] = "DEFERRED_RETRY_LOCAL_VERIFY_PHASE1_SAFE_SSDG_CVS_R01_LAUNCHER_SCHEMA_REQUIRED"
        phase1["defer_reason"] = (
            "Current run preserves Phase1 row-scoped DEFERRED_RETRY_LOCAL_VERIFY "
            "until Safe-SSDG PAIC launcher schema is locally verified."
        )
        phase1["exact_command"] = (
            "cd /home/szu2070436088/2510044040/CV-SincNet && "
            "# DEFERRED local verification: "
            "/home/szu2070436088/2510044040/CV-SincNet/code/train.py "
            "--epochs 200 --use_safe_ssdg_cvs --wisig_train_ratio 0.1 "
            "--ssl_labeled_ratio 0.1 --ssl_unlabeled_ratio 0.7 --ssl_val_ratio 0.2 "
            "--use_concat_sat_channel_aug --concat_sat_ce_only "
            "--sat_view_schedule '1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak' "
            "--use_sat_consistency"
        )
        items.append(phase1)
        idx += 1
        for slot_letter in "BCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "phase1_safe_ssdg_default_must_be_executable" in issue_names


def test_validate_matrix_rejects_duplicate_or_mismatched_64_stage2_gpu_slots():
    items = []
    idx = 0
    for gpu_idx in range(8):
        phase1 = _phase1_training_item(gpu_idx, idx)
        phase1["launchability_status"] = "phase1_launchable_training_candidate_full_200e_pending_remote_gates"
        items.append(phase1)
        idx += 1
        for slot_letter in "BCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1
    items[7]["slot"] = "GPU0/B"

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "candidate_id_slot_mismatch" in issue_names
    assert "canonical_gpu_slot_coverage_failed" in issue_names


def test_validate_matrix_rejects_stage2_only_matrix_without_phase1_quota():
    items = []
    idx = 0
    for gpu_idx in range(8):
        for slot_letter in "ABCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "lane_quota_mismatch" in issue_names
    assert "per_gpu_lane_quota_mismatch" in issue_names


def test_validate_matrix_rejects_phase1_protocol_only_rows_for_server_landed_training_gate():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_protocol_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "phase1_server_landed_training_candidate_required" in issue_names


def test_validate_matrix_rejects_legacy_stage2_source_new_split():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_protocol_item(gpu_idx, idx))
        items[-1]["launchability_status"] = "DEFERRED_RETRY_CAPACITY_WITH_EXACT_FULL_200E_TRAINING_COMMAND"
        items[-1]["exact_command"] = "python code/train.py --epochs 200 --use_meta_ssl_cvs"
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            item["source_tx_ids"] = "0,1,2,3"
            item["new_tx_ids"] = "4,5"
            item["unknown_tx_ids"] = "6,7"
            item.pop("target_old_tx_ids", None)
            item.pop("target_new_tx_ids", None)
            item.pop("target_receiver_ids", None)
            item.pop("cen51_train_rxs", None)
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "missing_target_old_tx_ids" in issue_names
    assert "source_tx_ids_inconsistent_with_manysig_old_tx_ids" in issue_names
    assert "target_new_tx_ids_overlap_manysig_old_tx_ids" in issue_names
    assert "missing_target_receiver_ids" in issue_names


def test_validate_matrix_allows_multi_receiver_target_domain_for_launchable_stage2():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["target_receiver_ids"] = "rx7,rx8"
            items.append(item)
            idx += 1

    result = validate(items, 64)
    phase2 = result["launchability_summary"]["by_lane"]["phase2_spaceborne_fsl"]

    assert result["verdict"] == "PASS"
    assert phase2["runner_readiness"] == "LANE_HAS_LAUNCHABLE_ROWS"
    assert phase2["launchable"] == 56


def test_validate_matrix_accepts_wisig_receiver_labels_without_numeric_false_overlap():
    items = []
    idx = 0
    sample_protocol = {
        "status": "ACTIVE",
        "old_tx_ids": [0, 1, 2, 3, 4, 5],
        "source_receiver_labels": "1-1,1-19,14-7,18-2,19-2,2-1,2-19",
    }
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            item.pop("cen51_train_rxs", None)
            item.pop("target_receiver_ids", None)
            item["source_receiver_labels"] = "1-1,1-19,14-7,18-2,19-2,2-1,2-19"
            item["target_receiver_labels"] = "3-19"
            item["target_old_leo_query"] = "ManySig old TX received by target receiver label 3-19"
            item["target_new_leo_query"] = "ManyTx non-old TX received by target receiver label 3-19"
            item["unknown_leo_query"] = "ManyTx held-out non-old TX received by target receiver label 3-19"
            items.append(item)
            idx += 1

    result = validate(items, 64, sample_protocol=sample_protocol)
    issue_names = {issue["issue"] for issue in result["issues"]}
    phase2 = result["launchability_summary"]["by_lane"]["phase2_spaceborne_fsl"]

    assert result["verdict"] == "PASS"
    assert "target_receiver_ids_overlap_cen51_train_receivers" not in issue_names
    assert phase2["runner_readiness"] == "LANE_HAS_LAUNCHABLE_ROWS"


def test_validate_matrix_rejects_target_receiver_domain_overlap_with_cen51_train_receivers():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["target_receiver_ids"] = "rx6,rx8"
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "target_receiver_ids_overlap_cen51_train_receivers" in issue_names


def test_validate_matrix_allows_multi_receiver_phase2_batch_when_each_row_is_single_rsat():
    items = []
    idx = 0
    target_receivers = ["rx7", "rx8", "rx9", "rx10", "rx11"]
    for gpu_idx in range(8):
        phase1 = _phase1_training_item(gpu_idx, idx)
        phase1["launchability_status"] = (
            "phase1_launchable_training_candidate_full_200e_pending_remote_gates"
        )
        items.append(phase1)
        idx += 1
        for slot_offset, slot_letter in enumerate("BCDEFGH"):
            item = _stage2_item(gpu_idx, slot_letter, idx)
            r_sat = target_receivers[(gpu_idx * 7 + slot_offset) % len(target_receivers)]
            item["target_receiver_ids"] = r_sat
            item["old_support_query_split"] = (
                "Stage2-A support=empty; Stage2-B support=target_old_fewshot; "
                f"query=target_old_{r_sat}"
            )
            item["new_support_query_split"] = (
                "Stage2-A/B support=empty; Stage2-C support=target_new_fewshot "
                f"only when enrolled; query=target_new_{r_sat}"
            )
            item["target_old_leo_query"] = (
                f"ManySig TX 0-5 received by target receiver {r_sat}"
            )
            item["target_new_leo_query"] = (
                f"non-ManySig TX 6,7 received by target receiver {r_sat}"
            )
            item["unknown_leo_query"] = (
                f"unknown TX 8,9 received by target receiver {r_sat}"
            )
            items.append(item)
            idx += 1

    result = validate(items, 64)
    phase2 = result["launchability_summary"]["by_lane"]["phase2_spaceborne_fsl"]

    assert result["verdict"] == "PASS"
    assert phase2["runner_readiness"] == "LANE_HAS_LAUNCHABLE_ROWS"
    assert phase2["launchable"] == 56
    assert phase2["local_patch_required"] == 0


def test_validate_matrix_allows_intermediate_positive_k_values():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["stage2_mode"] = "Stage2-B_old_label_calibration"
                item["target_old_support_per_tx"] = 4
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "PASS"
    assert "k_shot_must_be_positive_integer" not in issue_names
    assert "k_gt_fewshot_bound_must_be_labeled_higher_medium_or_saturation" not in issue_names


def test_validate_matrix_rejects_non_positive_k_values():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["stage2_mode"] = "Stage2-B_old_label_calibration"
                item["target_old_support_per_tx"] = 0
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "k_shot_must_be_positive_integer" in issue_names


def test_validate_matrix_requires_high_k_interpretation():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["stage2_mode"] = "Stage2-B_old_label_calibration"
                item["target_old_support_per_tx"] = 50
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "k_gt_fewshot_bound_must_be_labeled_higher_medium_or_saturation" in issue_names


def test_validate_matrix_rejects_duplicate_command_hash_in_64_row_matrix():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1
    items[1]["command_hash"] = items[2]["command_hash"]

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "duplicate_command_hash" in issue_names


def test_load_stage2_sample_protocol_reads_nested_state_values():
    protocol = load_stage2_sample_protocol(
        {
            "stage2_sample_protocol": {
                "status": "ACTIVE",
                "receiver_split_policy": {
                    "cen51_train_receiver_ids": [2, 3, 4],
                },
                "tx_split_policy": {
                    "old_tx_ids": [10, 11, 12, 13, 14, 15],
                },
            }
        }
    )

    assert protocol["cen51_train_receiver_ids"] == [2, 3, 4]
    assert protocol["old_tx_ids"] == [10, 11, 12, 13, 14, 15]


def test_validate_matrix_rejects_stage2_b_new_support_and_missing_unknown_query():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["stage2_mode"] = "Stage2-B_old_label_calibration"
                item["target_new_leo_support"] = "target_new few-shot support should not be used in Stage2-B"
                item["unknown_tx_ids"] = "7,8"
                item.pop("unknown_leo_query", None)
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "stage2_b_must_not_use_target_new_support" in issue_names
    assert "target_new_tx_ids_overlap_unknown_tx_ids" in issue_names
    assert "missing_target_receiver_query_split_fields" in issue_names


def test_validate_matrix_rejects_stage2_a_target_new_support_and_unknown_threshold_scope():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["stage2_mode"] = "Stage2-A_zero_label_deploy"
                item["target_new_leo_support"] = "target_new support is leakage for Stage2-A"
                item["threshold_selection_label_scope"] = "target_unknown_query"
                item["unknown_query_eval_only"] = False
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "stage2_a_must_not_use_target_new_support" in issue_names
    assert "unknown_query_must_not_calibrate_thresholds" in issue_names


def test_validate_matrix_requires_oa_mse_route_fields_and_decision_semantics():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["route_family"] = "OA_MSE_HEAD"
                item["route_signature"] = "oa_mse_head_missing_fields"
                item["gate_mode"] = "oa_mse"
                item.pop("threshold_selection_label_scope", None)
                item["model_output_semantics"] = "old_or_new_binary"
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "oa_mse_missing_required_fields" in issue_names
    assert "oa_mse_output_semantics_must_distinguish_defer_uncertain_reject" in issue_names


def test_validate_matrix_requires_opgac_route_base_and_metric_fields():
    item = _stage2_item(0, "B", 1)
    item.update(
        {
            "route_family": "OPGAC_NET",
            "route_signature": "opgac_old80_missing_fields",
            "stage2_mode": "Stage2-B_old_label_calibration",
            "k_shot": 10,
            "target_old_leo_support": "K=10 target-old support on rx7",
            "target_new_leo_support": False,
            "new_support_query_split": "support=empty; query=target_new_rx7; threshold_fit=forbidden",
            "model_output_semantics": "old_or_new_binary",
        }
    )

    result = validate([item], 1)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "opgac_missing_required_fields" in issue_names
    assert "opgac_stage2_base_model_must_be_jref_c9_multicomp_m2_e220" in issue_names
    assert "opgac_output_semantics_must_distinguish_old_reject_ambiguous" in issue_names


def test_validate_matrix_allows_opgac_jref_c9_old80_first_metric_bundle():
    item = _stage2_item(0, "B", 1)
    item.update(
        {
            "route_family": "OPGAC_NET",
            "route_signature": "opgac_jref_c9_old80_first",
            "stage2_mode": "Stage2-B_old_label_calibration",
            "stage2_base_model_id": "JREF_C9_MULTICOMP_M2_E220",
            "stage2_base_model_role": "receiver_floor_diagnostic_not_deployment_success",
            "opgac_stage": "old_calibration",
            "opgac_memory_policy": "support_only",
            "opgac_local_code_hook": "code/cvsrffi/opgac_net.py",
            "opgac_eval_tool": "tools/evaluate_opgac_stage2.py",
            "opgac_query_update_forbidden": True,
            "opgac_overlap_policy": "provisional_or_ambiguous",
            "opgac_rollback_policy": "rollback_to_ground_old_memory",
            "opgac_metric_bundle": (
                "old_acc,old80_gap,unknown_far,old_unknown_hmean,coverage,old_frr,"
                "auroc,fpr95,rollback_rate,defer_rate,confusion_counts,same_row_rank"
            ),
            "opgac_primary_selection_metric": "old_unknown_hmean_under_unknown_far_le_0p05_after_old80",
            "opgac_same_row_ranking_required": True,
            "opgac_score_table_required_columns": (
                "candidate_label,best_old_score,best_seen_new_score,best_reject_score,"
                "top2_margin,threshold_delta,opgac_old_score,opgac_new_score"
            ),
            "stage2_priority_phase": "OLD80_FIRST",
            "old_acc_target": 0.80,
            "deployment_success_claim_allowed": False,
            "k_shot": 10,
            "target_old_leo_support": "K=10 target-old support on rx7",
            "target_new_leo_support": False,
            "new_support_query_split": "support=empty; query=target_new_rx7; threshold_fit=forbidden",
            "model_output_semantics": "old_label,reject,ambiguous,defer",
        }
    )

    result = validate([item], 1)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert "opgac_missing_required_fields" not in issue_names
    assert "opgac_stage2_base_model_must_be_jref_c9_multicomp_m2_e220" not in issue_names
    assert "opgac_metric_bundle_incomplete" not in issue_names
    assert "phase2_old_acc_target_must_be_at_least_0p90" not in issue_names


def test_validate_matrix_rejects_old_primary_route_rescue_without_terminal_gate():
    item = _stage2_item(0, "A", 0)
    item.update(
        {
            "candidate_id": "OA_MSE_H06_OLDRELAX48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0",
            "route_family": "OA_MSE_HEAD",
            "gate_mode": "oa_mse",
            "model_output_semantics": "old_label,reject,uncertain,defer",
            "stage2_mode": "Stage2-B_old_label_calibration",
            "target_new_k": 0,
            "target_new_support_per_tx": 0,
            "target_new_tx_ids": "",
            "target_new_tx_labels": "",
            "new_tx_ids": "",
            "target_new_leo_query": "not_applicable_old_unknown_only",
            "new_support_query_split": "not_applicable_old_unknown_only",
            "seen_new_evidence_gate_calibration_scope": "not_applicable_old_unknown_only",
            "oa_mse_retention_rescue_gate": True,
            "oa_mse_old_primary_gate": False,
        }
    )

    result = validate([item], 1)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "old_primary_route_retention_rescue_requires_old_primary_gate" in issue_names


def test_validate_matrix_rejects_old_primary_soft_class_envelope_ablation():
    item = _stage2_item(0, "A", 0)
    item.update(
        {
            "candidate_id": "OA_MSE_H06_OLDRELAX48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0",
            "route_family": "OA_MSE_HEAD",
            "gate_mode": "oa_mse",
            "model_output_semantics": "old_label,reject,uncertain,defer",
            "stage2_mode": "Stage2-B_old_label_calibration",
            "target_new_k": 0,
            "target_new_support_per_tx": 0,
            "target_new_tx_ids": "",
            "target_new_tx_labels": "",
            "new_tx_ids": "",
            "target_new_leo_query": "not_applicable_old_unknown_only",
            "new_support_query_split": "not_applicable_old_unknown_only",
            "seen_new_evidence_gate_calibration_scope": "not_applicable_old_unknown_only",
            "oa_mse_old_primary_gate": True,
            "oa_mse_class_envelope_gate": True,
            "class_envelope_gate_action": "uncertain",
            "old_primary_require_soft_mixture": True,
            "old_primary_require_support_knn": True,
            "old_primary_require_class_envelope": False,
            "fusion_inputs": "class_envelope_soft_ablation,old_primary_terminal_gate",
        }
    )

    result = validate([item], 1)
    issues = [issue for issue in result["issues"] if issue.get("field") == "old_primary_require_class_envelope"]

    assert result["verdict"] == "FAIL"
    assert any(issue["issue"] == "old_primary_gate_missing_required_subgate" for issue in issues)


def test_validate_matrix_rejects_old_primary_rescue_without_candidate_only_promotion():
    item = _stage2_item(0, "A", 0)
    item.update(
        {
            "candidate_id": "OA_MSE_H06_OLDRELAX48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0",
            "route_family": "OA_MSE_HEAD",
            "gate_mode": "oa_mse",
            "model_output_semantics": "old_label,reject,uncertain,defer",
            "stage2_mode": "Stage2-B_old_label_calibration",
            "target_new_k": 0,
            "target_new_support_per_tx": 0,
            "target_new_tx_ids": "",
            "target_new_tx_labels": "",
            "new_tx_ids": "",
            "target_new_leo_query": "not_applicable_old_unknown_only",
            "new_support_query_split": "not_applicable_old_unknown_only",
            "seen_new_evidence_gate_calibration_scope": "not_applicable_old_unknown_only",
            "oa_mse_old_primary_gate": True,
            "oa_mse_class_envelope_gate": True,
            "old_primary_require_soft_mixture": True,
            "old_primary_require_support_knn": True,
            "old_primary_require_class_envelope": True,
            "oa_mse_retention_rescue_gate": True,
            "old_primary_unknown_veto_action": "reject",
            "retention_rescue_candidate_only": False,
            "old_primary_promote_rescue_candidates": False,
            "score_table_required_columns": (
                "candidate_label,candidate_group,outcome_code,old_primary_consistency_pass,"
                "old_primary_unknown_veto,old_primary_blocked_accept,"
                "old_primary_rescue_promoted,old_primary_rescue_blocked"
            ),
        }
    )

    result = validate([item], 1)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "old_primary_route_retention_rescue_must_be_candidate_only" in issue_names
    assert "old_primary_route_rescue_candidates_must_be_terminally_promoted" in issue_names


def test_validate_matrix_requires_launchable_phase2_threshold_guard_fields():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item.pop("unknown_query_eval_only", None)
                item.pop("target_new_query_not_threshold_fit", None)
                item.pop("unknown_FAR_target", None)
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "missing_stage2_threshold_guard_fields" in issue_names


def test_validate_matrix_requires_onboard_low_compute_training_targets_and_modules():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                for field in (
                    "onboard_low_compute_training",
                    "compute_budget_profile",
                    "old_acc_target",
                    "seen_new_acc_target",
                    "weibull_evt_required",
                    "target_adapter_required",
                    "pseudo_unknown_energy_required",
                    "seen_new_evidence_gate_required",
                    "seen_new_anchor_gate_required",
                    "siamese_verifier_required",
                    "accepted_only_online_update_required",
                    "oa_mse_onboard_adaptation_bundle",
                ):
                    item.pop(field, None)
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "missing_onboard_low_compute_training_fields" in issue_names


def test_stage2_b_false_target_new_support_does_not_count_as_support():
    item = _stage2_item(0, "B", 0)
    item["stage2_mode"] = "Stage2-B_old_label_calibration"
    item["target_new_leo_support"] = False
    item["new_support_query_split"] = "support=empty; query=target_new_rx7; threshold_fit=forbidden"

    issues = ovm.stage2_sample_protocol_issues(item, ovm.DEFAULT_STAGE2_SAMPLE_PROTOCOL)
    issue_names = {issue["issue"] for issue in issues}

    assert "stage2_b_must_not_use_target_new_support" not in issue_names


def test_stage2_sample_protocol_rejects_clean_access_and_non_leo_weak_views():
    item = _stage2_item(0, "B", 0)
    item["clean_sample_access"] = True
    item["clean_view_role"] = "control_only"
    item["target_channel_view"] = "clean+satellite/LEO"
    item["target_channel_scenarios"] = "leo_clear_weak,clean,storm_mp"
    item["exact_command"] = "python run.py --target_channel_view clean --target_train_scenarios clear_leo,storm_mp"

    issues = ovm.stage2_sample_protocol_issues(item, ovm.DEFAULT_STAGE2_SAMPLE_PROTOCOL)
    issue_names = {issue["issue"] for issue in issues}

    assert "stage2_clean_sample_access_must_be_false" in issue_names
    assert "stage2_clean_view_role_must_be_not_accessible" in issue_names
    assert "stage2_target_channel_view_must_be_leo_weak_only" in issue_names
    assert "stage2_target_channel_scenarios_must_be_leo_weak_only" in issue_names
    assert "stage2_clean_token_forbidden_in_sample_view_fields" in issue_names
    assert "stage2_exact_command_accesses_non_leo_weak_or_clean_view" in issue_names


def test_stage2_sample_protocol_requires_no_clean_policy_and_overlay_provenance():
    item = _stage2_item(0, "B", 0)
    item.pop("phase2_sample_view_policy")
    item.pop("clean_sample_access")
    item.pop("satellite_seed")

    issues = ovm.stage2_sample_protocol_issues(item, ovm.DEFAULT_STAGE2_SAMPLE_PROTOCOL)
    issue_names = {issue["issue"] for issue in issues}

    assert "stage2_sample_view_policy_must_be_leo_weak_only_no_clean_access" in issue_names
    assert "stage2_clean_sample_access_guard_missing" in issue_names
    assert "stage2_leo_weak_overlay_provenance_missing" in issue_names


def test_validate_matrix_rejects_incomplete_onboard_adaptation_bundle():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["oa_mse_onboard_adaptation_bundle"] = "weibull_evt+target_adapter"
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "oa_mse_onboard_adaptation_bundle_incomplete" in issue_names


def test_validate_matrix_rejects_low_phase2_old_or_seen_new_targets():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["old_acc_target"] = 0.85
                item["seen_new_acc_target"] = 0.70
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "phase2_old_acc_target_must_be_at_least_0p90" in issue_names
    assert "phase2_seen_new_acc_target_must_be_at_least_0p75" in issue_names


def test_validate_matrix_rejects_target_new_query_threshold_fitting():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["threshold_selection_label_scope"] = "source_old_and_target_new_query"
                item["target_new_query_not_threshold_fit"] = False
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "target_new_query_must_not_fit_thresholds" in issue_names


def test_validate_matrix_reports_lane_launchability_without_treating_schema_pass_as_runner_pass():
    items = []
    idx = 0
    for gpu_idx in range(8):
        phase1 = _phase1_training_item(gpu_idx, idx)
        phase1["launchability_status"] = "phase1_launchable_training_candidate_full_200e_pending_remote_gates"
        items.append(phase1)
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            item["route_signature_family"] = "stage2_duplicate_family_v1"
            item["launchability_status"] = "LOCAL_PATCH_REQUIRED_ROUTE_DUPLICATION_REPAIR"
            item["defer_reason"] = "DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_DUPLICATION_REPAIR_REQUIRED_NO_STAGE2_LAUNCH"
            items.append(item)
            idx += 1

    result = validate(items, 64)
    phase1 = result["launchability_summary"]["by_lane"]["phase1_ground_dg"]
    phase2 = result["launchability_summary"]["by_lane"]["phase2_spaceborne_fsl"]

    assert result["verdict"] == "PASS"
    assert phase1["runner_readiness"] == "LANE_HAS_LAUNCHABLE_ROWS"
    assert phase1["launchable"] == 8
    assert phase2["runner_readiness"] == "LANE_LOCAL_PATCH_REQUIRED_NO_LAUNCHABLE_ROWS"
    assert phase2["launchable"] == 0
    assert phase2["local_patch_required"] == 56
    assert phase2["route_duplication_repair"] == 56


def test_validate_matrix_detects_mse_lite_and_mse_subspace_as_oa_mse_rows():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["route_signature"] = "mse_lite_source_open_gate"
                item["stage2_mode"] = "Stage2-A_MSE-Subspace_probe"
                item["model_output_semantics"] = "old_or_new_binary"
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "oa_mse_missing_required_fields" in issue_names
    assert "oa_mse_output_semantics_must_distinguish_defer_uncertain_reject" in issue_names


def test_validate_matrix_rejects_phase1_without_default_paic_star_ground_aug():
    items = []
    idx = 0
    stripped_keys = (
        "phase1_star_ground_aug_policy",
        "phase1_star_ground_aug_default_enabled",
        "phase1_star_ground_aug_route_family",
        "phase1_star_ground_aug_mode",
        "use_concat_sat_channel_aug",
        "concat_sat_ce_only",
        "use_sat_consistency",
        "sat_view_schedule",
        "star_ground_aug_exploration_axis",
    )
    for gpu_idx in range(8):
        phase1 = _phase1_training_item(gpu_idx, idx)
        for key in stripped_keys:
            phase1.pop(key, None)
        if isinstance(phase1.get("parameters"), dict):
            for key in stripped_keys:
                phase1["parameters"].pop(key, None)
        phase1["exact_command"] = "python code/train.py --epochs 200 --use_safe_ssdg_cvs"
        items.append(phase1)
        idx += 1
        for slot_letter in "BCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "phase1_star_ground_aug_default_required" in issue_names
    assert "phase1_star_ground_aug_requires_paic_route_family" in issue_names
    assert "phase1_star_ground_aug_requires_concat_ce_only_mode" in issue_names
    assert "phase1_star_ground_aug_requires_schedule" in issue_names


def test_phase1_control_surfaces_require_default_paic_star_ground_aug():
    prompt = (
        PROJECT_ROOT
        / "automation_reports"
        / "CV-SincNet"
        / "automation_prompt_backups"
        / "20260615_001820_stage2_closed_loop_v4"
        / "stage2_prompt.md"
    ).read_text(encoding="utf-8")
    contract = (PROJECT_ROOT / "tools" / "optimizer_workflow_contract.md").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "tools" / "optimizer_control_manifest.md").read_text(encoding="utf-8")
    state = json.loads((PROJECT_ROOT / "automation_reports" / "CV-SincNet" / "stage2_optimizer_state.json").read_text(encoding="utf-8"))

    for surface in (prompt, contract, manifest):
        assert "phase1_star_ground_aug_default_enabled" in surface
        assert "CVS-SAT-PAIC" in surface
        assert "concat_sat_ce_only" in surface

    default_policy = state["phase1_ground_dg_direction"]["phase1_star_ground_aug_default"]
    assert default_policy["status"] == "PAIC_STAR_GROUND_AUG_DEFAULT_ON_FOR_PHASE1"
    assert default_policy["route_family"] == "CVS-SAT-PAIC"
    assert default_policy["default_training_flags"]["concat_sat_ce_only"] is True
    assert default_policy["default_training_flags"]["use_concat_sat_channel_aug"] is True


def test_phase1_control_surfaces_require_executable_safe_ssdg_default():
    prompt = (
        PROJECT_ROOT
        / "automation_reports"
        / "CV-SincNet"
        / "automation_prompt_backups"
        / "20260615_001820_stage2_closed_loop_v4"
        / "stage2_prompt.md"
    ).read_text(encoding="utf-8")
    contract = (PROJECT_ROOT / "tools" / "optimizer_workflow_contract.md").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "tools" / "optimizer_control_manifest.md").read_text(encoding="utf-8")
    state = json.loads((PROJECT_ROOT / "automation_reports" / "CV-SincNet" / "stage2_optimizer_state.json").read_text(encoding="utf-8"))

    for surface in (prompt, contract, manifest):
        assert "Phase1 Safe-SSDG rows default executable" in surface
        assert "run_phase1_safe_ssdg_candidate" in surface
        assert "code/SSDG/train_ssdg.py" in surface

    policy = state["phase1_ground_dg_direction"]["phase1_safe_ssdg_execution_policy"]
    assert policy["status"] == "PHASE1_SAFE_SSDG_EXECUTABLE_BY_DEFAULT"
    assert policy["row_launcher_entrypoint"] == "run_phase1_safe_ssdg_candidate"
    assert policy["direct_training_entrypoint"] == "python ${ROOT}/code/SSDG/train_ssdg.py"
    assert policy["local_schema_defer_allowed_by_default"] is False


def test_optimizer_control_surfaces_require_evidence_first_current_run_matrix():
    prompt = (
        PROJECT_ROOT
        / "automation_reports"
        / "CV-SincNet"
        / "automation_prompt_backups"
        / "20260615_001820_stage2_closed_loop_v4"
        / "stage2_prompt.md"
    ).read_text(encoding="utf-8")
    contract = (PROJECT_ROOT / "tools" / "optimizer_workflow_contract.md").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "tools" / "optimizer_control_manifest.md").read_text(encoding="utf-8")
    state = json.loads((PROJECT_ROOT / "automation_reports" / "CV-SincNet" / "stage2_optimizer_state.json").read_text(encoding="utf-8"))

    for surface in (prompt, contract, manifest):
        assert "evidence-first current-run matrix" in surface
        assert "next-run matrix handoff is audit-only" in surface
        assert "same automation run" in surface

    policy = state["optimizer_matrix_generation_policy"]
    assert policy["status"] == "EVIDENCE_FIRST_CURRENT_RUN_MATRIX_REQUIRED"
    assert policy["next_run_matrix_handoff_is_audit_only"] is True
    assert policy["runner_required_in_same_automation_run"] is True


def test_optimizer_control_surfaces_require_idle_lane_repair_until_launch():
    prompt = (
        PROJECT_ROOT
        / "automation_reports"
        / "CV-SincNet"
        / "automation_prompt_backups"
        / "20260615_001820_stage2_closed_loop_v4"
        / "stage2_prompt.md"
    ).read_text(encoding="utf-8")
    contract = (PROJECT_ROOT / "tools" / "optimizer_workflow_contract.md").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "tools" / "optimizer_control_manifest.md").read_text(encoding="utf-8")
    state = json.loads((PROJECT_ROOT / "automation_reports" / "CV-SincNet" / "stage2_optimizer_state.json").read_text(encoding="utf-8"))

    for surface in (prompt, contract, manifest):
        assert "idle lane must execute" in surface
        assert "repair-until-launch" in surface
        assert "missing current-run matrix is repair work, not a terminal outcome" in surface
        assert "NO_CURRENT_MATRIX_VALIDATION" in surface
        assert "DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION" in surface

    policy = state["idle_lane_execution_policy"]
    assert policy["schema"] == "idle_lane_execution_policy_v1"
    allowed_status = (
        policy["status"] == "REPAIR_UNTIL_RUNNER_EXECUTES"
        or policy["status"].startswith("RUNNER_EXECUTED_CURRENT_MATRIX_")
        or policy["status"] == "MONITOR_ONLY_PHASE1_ACTIVE_PHASE2_IDLE"
    )
    assert allowed_status
    assert policy["idle_lane_must_execute_experiment"] is True
    if policy["status"] == "REPAIR_UNTIL_RUNNER_EXECUTES":
        assert policy["repair_until_runner_executes"] is True
    elif policy["status"].startswith("RUNNER_EXECUTED_CURRENT_MATRIX_"):
        assert policy["repair_until_runner_executes"] is False
        assert policy["latest_execution_n607_run_id"].startswith("stage2_spaceborne_")
    else:
        assert policy["repair_until_runner_executes"] is False
        assert state["required_next_action"] in {
            "MONITOR_PHASE1_RETRY_TO_COMPLETION_AND_ANALYZE_FULL_TRAINING_LOGS",
            "MONITOR_PHASE1_FLOORREPAIR_TO_COMPLETION_AND_ANALYZE_FULL_TRAINING_LOGS",
        }
        assert state["latest_two_lane_monitor_result"]["phase1_monitor_state"] == 0
        assert state["latest_two_lane_monitor_result"]["phase2_monitor_state"] == 1
        assert "active_same_lane_or_unsafe_ambiguous_process" in policy["terminal_blockers"]
    assert "NO_CURRENT_MATRIX_VALIDATION" in policy["disallowed_terminal_outcomes_for_idle_lane"]
    assert "NOT_RUN_NO_CURRENT_REPAIRED_MATRIX" in policy["disallowed_terminal_outcomes_for_idle_lane"]
    assert "DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION" in policy["disallowed_terminal_outcomes_for_idle_lane"]
    assert "missing_current_run_matrix" in policy["repairable_items_not_terminal"]
    assert "runner_identity_drift" in policy["repairable_items_not_terminal"]
    assert "required_control_file_unreadable" in policy["terminal_blockers"]
    assert "explicit_user_pause_or_stop" in policy["terminal_blockers"]
    assert policy["row_level_defer_only_with_at_least_one_launch"] is True


def test_optimizer_control_surfaces_require_training_log_observability():
    prompt = (
        PROJECT_ROOT
        / "automation_reports"
        / "CV-SincNet"
        / "automation_prompt_backups"
        / "20260615_001820_stage2_closed_loop_v4"
        / "stage2_prompt.md"
    ).read_text(encoding="utf-8")
    contract = (PROJECT_ROOT / "tools" / "optimizer_workflow_contract.md").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "tools" / "optimizer_control_manifest.md").read_text(encoding="utf-8")
    state = json.loads((PROJECT_ROOT / "automation_reports" / "CV-SincNet" / "stage2_optimizer_state.json").read_text(encoding="utf-8"))

    assert "Training Log Analysis Prompt" in prompt
    assert "MISSING_LOSS_TELEMETRY" in prompt
    assert "CONFIG_LOSS_ALIGNMENT_GAP" in prompt
    assert "Training Log Telemetry Contract" in contract
    assert "FULL_TRAINING_LOG_ANALYSIS_REQUIRED" in contract
    assert "LOSS_NORMAL_CLAIM_REQUIRES_CURVE" in contract
    assert "ADAPTER_LOSS_TRACE_REQUIRED" in contract
    assert "Training log observability" in manifest
    assert "training_log_observability_policy" in manifest

    policy = state["training_log_observability_policy"]
    assert policy["schema"] == "training_log_observability_policy_v1"
    assert policy["status"] == "FULL_TRAINING_LOG_ANALYSIS_REQUIRED_BEFORE_LOSS_OR_OPTIMIZATION_CLAIMS"
    assert "FULL_LOSS_TELEMETRY_REQUIRED" in policy["required_evidence_labels"]
    assert "per_epoch_metrics_csv_or_jsonl" in policy["training_rows_require"]
    assert "loss_trace" in policy["stage2_adapter_rows_require"]


def test_control_surfaces_reference_read_only_preflight_decision_bundle():
    prompt = (
        PROJECT_ROOT
        / "automation_reports"
        / "CV-SincNet"
        / "automation_prompt_backups"
        / "20260615_001820_stage2_closed_loop_v4"
        / "stage2_prompt.md"
    ).read_text(encoding="utf-8")
    contract = (PROJECT_ROOT / "tools" / "optimizer_workflow_contract.md").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "tools" / "optimizer_control_manifest.md").read_text(encoding="utf-8")

    for surface in (prompt, contract, manifest):
        assert "tools/optimizer_preflight_decision.py" in surface
        assert "PENDING_REMOTE_MONITOR" in surface
        assert "read-only local preflight" in surface
        assert "must not run SSH/SCP or launch" in surface


def test_validate_matrix_rejects_current_run_identity_drift():
    current_run = "stage2_spaceborne_current_20260622_190000"
    old_run = "stage2_spaceborne_old_20260622_152854"
    items = _current_run_bound_64_items(current_run)
    items[8]["estimated_run_path"] = items[8]["estimated_run_path"].replace(current_run, old_run)
    items[9]["registry_key"] = items[9]["registry_key"].replace(current_run, old_run)
    items[10]["exact_command"] = items[10]["exact_command"].replace(f"RUN_ID={current_run}", f"RUN_ID={old_run}")

    result = validate(
        items,
        64,
        matrix_root={
            "run_id": "stage2_optimizer_20260622_190000",
            "n607_run_id": current_run,
        },
    )
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "estimated_run_path_not_under_current_n607_run_id" in issue_names
    assert "registry_key_not_bound_to_current_n607_run_id" in issue_names


def test_validate_matrix_rejects_launcher_default_run_id_mismatch():
    current_run = "stage2_spaceborne_current_20260622_190000"
    old_run = "stage2_spaceborne_old_20260622_152854"
    items = _current_run_bound_64_items(current_run)
    launcher_text = f"""#!/usr/bin/env bash
ROOT="${{ROOT:-/home/szu2070436088/2510044040/CV-SincNet}}"
RUN_ID="${{RUN_ID:-{old_run}}}"
RUNS_ROOT="${{RUNS_ROOT:-${{ROOT}}/runs/${{RUN_ID}}}}"
LOG_ROOT="${{LOG_ROOT:-${{ROOT}}/logs/${{RUN_ID}}}}"
source "${{ROOT}}/tools/stage2_queue_runner_template.sh"
"""

    result = validate(
        items,
        64,
        matrix_root={
            "run_id": "stage2_optimizer_20260622_190000",
            "n607_run_id": current_run,
        },
        launcher_text=launcher_text,
        launcher_path="code/scripts/launch_stage2_optimizer_20260622_190000_next64zz.sh",
    )
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "launcher_default_run_id_mismatch" in issue_names


def test_repair_first_runner_identity_preflight_fixes_launcher_before_blocking():
    current_run = "stage2_spaceborne_current_20260622_190000"
    old_run = "stage2_spaceborne_old_20260622_152854"
    items = _current_run_bound_64_items(current_run)
    stale_launcher_text = f"""#!/usr/bin/env bash
ROOT="${{ROOT:-/home/szu2070436088/2510044040/CV-SincNet}}"
RUN_ID="${{RUN_ID:-{old_run}}}"
RUNS_ROOT="${{RUNS_ROOT:-/tmp/stale_runs}}"
LOG_ROOT="${{LOG_ROOT:-/tmp/stale_logs}}"
PHASE2_LOCAL_PATCH_REQUIRED="${{PHASE2_LOCAL_PATCH_REQUIRED:-1}}"
source "${{ROOT}}/tools/stage2_queue_runner_template.sh"
stage2_acquire_launcher_lock
"""

    red_result = validate(
        items,
        64,
        matrix_root={
            "run_id": "stage2_optimizer_20260622_190000",
            "n607_run_id": current_run,
        },
        launcher_text=stale_launcher_text,
        launcher_path="code/scripts/launch_stage2_optimizer_20260622_190000_next64zz.sh",
    )
    assert red_result["verdict"] == "FAIL"

    repaired_text, repairs = ovm.repair_launcher_identity_text(
        stale_launcher_text,
        expected_run_id=current_run,
        phase2_launchable_rows=56,
    )
    repaired_result = validate(
        items,
        64,
        matrix_root={
            "run_id": "stage2_optimizer_20260622_190000",
            "n607_run_id": current_run,
        },
        launcher_text=repaired_text,
        launcher_path="code/scripts/launch_stage2_optimizer_20260622_190000_next64zz.sh",
    )

    assert {repair["action"] for repair in repairs} == {
        "set_launcher_default_run_id",
        "bind_runs_root_to_run_id",
        "bind_log_root_to_run_id",
        "clear_phase2_local_patch_default_for_launchable_rows",
        "remove_direct_template_lock_call",
    }
    assert repaired_result["verdict"] == "PASS"


def test_optimizer_control_surfaces_require_runner_identity_preflight():
    prompt = (
        PROJECT_ROOT
        / "automation_reports"
        / "CV-SincNet"
        / "automation_prompt_backups"
        / "20260615_001820_stage2_closed_loop_v4"
        / "stage2_prompt.md"
    ).read_text(encoding="utf-8")
    contract = (PROJECT_ROOT / "tools" / "optimizer_workflow_contract.md").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "tools" / "optimizer_control_manifest.md").read_text(encoding="utf-8")
    state = json.loads((PROJECT_ROOT / "automation_reports" / "CV-SincNet" / "stage2_optimizer_state.json").read_text(encoding="utf-8"))

    for surface in (prompt, contract, manifest):
        assert "runner identity preflight" in surface
        assert "repair-first runner identity preflight" in surface
        assert "launcher default RUN_ID" in surface
        assert "n607_run_id" in surface
        assert "--repair-launcher-identity" in surface

    policy = state["runner_identity_preflight_policy"]
    assert policy["status"] == "REPAIR_FIRST_CURRENT_RUN_IDENTITY_PREFLIGHT_REQUIRED"
    assert policy["auto_repair_before_blocking"] is True
    assert policy["launcher_default_run_id_must_match_n607_run_id"] is True
    assert policy["matrix_paths_must_match_n607_run_id"] is True


def test_validate_matrix_rejects_unknown_tx_overlap_with_manysig_old_tx_ids():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            if gpu_idx == 0 and slot_letter == "B":
                item["unknown_tx_ids"] = "5,8"
            items.append(item)
            idx += 1

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "unknown_tx_ids_overlap_manysig_old_tx_ids" in issue_names


def test_validate_matrix_rejects_wisig_manytx_rows_without_resolved_tx_labels():
    items = []
    idx = 0
    sample_protocol = {
        "status": "ACTIVE",
        "old_tx_ids": [0, 1, 2, 3, 4, 5],
        "source_receiver_labels": "1-1,1-19,14-7,18-2,19-2,2-1,2-19",
        "confirmed_wisig_candidate_pool": {
            "old_tx_labels": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
            "target_receiver_labels": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        },
    }
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            item.pop("cen51_train_rxs", None)
            item.pop("target_receiver_ids", None)
            item["source_receiver_labels"] = "1-1,1-19,14-7,18-2,19-2,2-1,2-19"
            item["target_receiver_labels"] = "20-1"
            item["manytx_target_rx_index"] = "10"
            item["target_new_tx_ids"] = "6,7,8,9"
            item["unknown_tx_ids"] = "100,101,102,103"
            item["target_new_tx_labels"] = (
                "ManyTx non-Y_old tx_list ranks 6,7,8,9; resolve exact labels from ManyTx.pkl"
            )
            item["unknown_tx_labels"] = (
                "ManyTx held-out non-Y_old tx_list ranks 100,101,102,103"
            )
            items.append(item)
            idx += 1

    result = validate(items, 64, sample_protocol=sample_protocol)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "wisig_manytx_target_new_tx_labels_not_resolved" in issue_names
    assert "wisig_manytx_unknown_tx_labels_not_resolved" in issue_names
    assert "wisig_manytx_unknown_tx_ids_must_not_be_synthetic_numeric_ranks" in issue_names


def test_validate_matrix_rejects_wisig_manytx_label_overlap_with_manysig_old_labels():
    items = []
    idx = 0
    sample_protocol = {
        "status": "ACTIVE",
        "old_tx_ids": [0, 1, 2, 3, 4, 5],
        "source_receiver_labels": "1-1,1-19,14-7,18-2,19-2,2-1,2-19",
        "confirmed_wisig_candidate_pool": {
            "old_tx_labels": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
            "target_receiver_labels": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        },
    }
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            item.pop("cen51_train_rxs", None)
            item.pop("target_receiver_ids", None)
            item["source_receiver_labels"] = "1-1,1-19,14-7,18-2,19-2,2-1,2-19"
            item["target_receiver_labels"] = "20-1"
            item["manytx_target_rx_index"] = "10"
            item["target_new_tx_ids"] = "1-16,14-10"
            item["unknown_tx_ids"] = "10-1,20-15"
            item["target_new_tx_labels"] = "1-16,14-10"
            item["unknown_tx_labels"] = "10-1,20-15"
            items.append(item)
            idx += 1

    result = validate(items, 64, sample_protocol=sample_protocol)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "wisig_manytx_target_new_tx_labels_overlap_manysig_old_tx_labels" in issue_names
    assert "wisig_manytx_unknown_tx_labels_overlap_manysig_old_tx_labels" in issue_names


def test_validate_matrix_accepts_wisig_manytx_rows_with_resolved_nonold_tx_labels():
    items = []
    idx = 0
    sample_protocol = {
        "status": "ACTIVE",
        "old_tx_ids": [0, 1, 2, 3, 4, 5],
        "source_receiver_labels": "1-1,1-19,14-7,18-2,19-2,2-1,2-19",
        "confirmed_wisig_candidate_pool": {
            "old_tx_labels": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
            "target_receiver_labels": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        },
    }
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            item = _stage2_item(gpu_idx, slot_letter, idx)
            item.pop("cen51_train_rxs", None)
            item.pop("target_receiver_ids", None)
            item["source_receiver_labels"] = "1-1,1-19,14-7,18-2,19-2,2-1,2-19"
            item["target_receiver_labels"] = "20-1"
            item["manytx_target_rx_index"] = "10"
            item["target_new_tx_ids"] = "1-16,1-18"
            item["unknown_tx_ids"] = "10-1,10-10"
            item["target_new_tx_labels"] = "1-16,1-18"
            item["unknown_tx_labels"] = "10-1,10-10"
            items.append(item)
            idx += 1

    result = validate(items, 64, sample_protocol=sample_protocol)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "PASS"
    assert "wisig_manytx_target_new_tx_labels_not_resolved" not in issue_names
    assert "wisig_manytx_unknown_tx_labels_not_resolved" not in issue_names


def test_current_state_view_keeps_only_current_decision_fields():
    state = {
        "last_updated_local": "2026-06-21T21:31:12+08:00",
        "lane_monitor_policy": {"entry_policy": "lane scoped"},
        "lane_capacity_policy": {"phase1_max_active_per_gpu": 1},
        "idle_lane_execution_policy": {"status": "REPAIR_UNTIL_RUNNER_EXECUTES"},
        "training_log_observability_policy": {"status": "FULL_TRAINING_LOG_ANALYSIS_REQUIRED_BEFORE_LOSS_OR_OPTIMIZATION_CLAIMS"},
        "phase1_ground_dg_direction": {"default_route_family": "SAFE_SSDG_CVS_R01"},
        "latest_two_lane_monitor_result": {"phase1_monitor_state": 1, "phase2_monitor_state": 1},
        "latest_optimizer_runner_result": {"phase1_outcome": "LAUNCHED", "phase2_outcome": "DEFERRED"},
        "objective_changelog": [{"large": "historical"}],
        "active_focus": {"old": "audit only"},
        "phase2_spaceborne_fsl": {"route_retirement_policy": {"status": "ACTIVE"}},
    }

    view = current_view(state, Path("state.json"))

    assert view["schema"] == "stage2_optimizer_current_state_view_v1"
    assert "phase1_ground_dg_direction" in view["current"]
    assert "idle_lane_execution_policy" in view["current"]
    assert "training_log_observability_policy" in view["current"]
    assert "latest_two_lane_monitor_result" in view["current"]
    assert "latest_optimizer_runner_result" in view["current"]
    assert "objective_changelog" not in view["current"]
    assert "active_focus" not in view["current"]
    assert "phase2_spaceborne_fsl" not in view["current"]
    assert set(view["audit_only_keys_present"]) == {"objective_changelog", "active_focus", "phase2_spaceborne_fsl"}


def test_preflight_decision_reports_local_ready_as_pending_remote_monitor(tmp_path):
    from optimizer_preflight_decision import preflight_decision

    _write_minimal_preflight_controls(tmp_path)
    matrix_path, launcher_path = _write_preflight_matrix(tmp_path)

    decision = preflight_decision(
        project_root=tmp_path,
        matrix_path=matrix_path,
        launcher_path=launcher_path,
    )

    assert decision["schema"] == "optimizer_preflight_decision_v1"
    assert decision["overall_status"] == "PENDING_REMOTE_MONITOR"
    assert decision["control_readiness"]["status"] == "PASS"
    assert decision["matrix_readiness"]["status"] == "PASS"
    assert decision["matrix_readiness"]["runner_readiness"] == "MATRIX_HAS_LAUNCHABLE_ROWS"
    assert decision["launcher_readiness"]["status"] == "PASS"
    assert decision["duplicate_readiness"]["status"] == "PASS"
    assert decision["remote_readiness"]["status"] == "PENDING_REMOTE_MONITOR"
    assert decision["remote_readiness"]["remote_actions_performed"] is False
    assert decision["control_readiness"]["required_files"][0]["sha256"]


def test_preflight_decision_blocks_when_required_control_file_is_unreadable(tmp_path):
    from optimizer_preflight_decision import preflight_decision

    _write_minimal_preflight_controls(tmp_path)
    (tmp_path / "项目.md").unlink()

    decision = preflight_decision(project_root=tmp_path)

    assert decision["overall_status"] == "BLOCKED"
    assert decision["blocker_code"] == "USER_REQUIRED_SAFETY_STOP"
    assert decision["control_readiness"]["status"] == "BLOCKED"
    missing = [item for item in decision["control_readiness"]["required_files"] if item["status"] != "PASS"]
    assert missing[0]["relative_path"] == "项目.md"
    assert decision["remote_readiness"]["remote_actions_performed"] is False


def test_preflight_decision_checks_duplicate_command_hash_for_non64_matrix(tmp_path):
    from optimizer_preflight_decision import preflight_decision

    _write_minimal_preflight_controls(tmp_path)
    matrix_path, launcher_path = _write_preflight_matrix(tmp_path, duplicate_command_hash=True)

    decision = preflight_decision(
        project_root=tmp_path,
        matrix_path=matrix_path,
        launcher_path=launcher_path,
    )

    assert decision["overall_status"] == "BLOCKED"
    assert decision["duplicate_readiness"]["status"] == "BLOCKED"
    assert "duplicate_command_hash" in {issue["issue"] for issue in decision["duplicate_readiness"]["issues"]}


def test_validate_matrix_rejects_per_gpu_lane_quota_even_when_global_quota_matches():
    items = []
    idx = 0
    for gpu_idx in range(8):
        items.append(_phase1_training_item(gpu_idx, idx))
        idx += 1
        for slot_letter in "BCDEFGH":
            items.append(_stage2_item(gpu_idx, slot_letter, idx))
            idx += 1

    items[1].update(
        {
            "lane": "phase1_ground_dg",
            "phase_axis": "Phase1-GroundDG",
            "stage2_mode": "NOT_APPLICABLE",
            "protocol": "Safe-SSDG-CVS-R01",
            "route_family": "SAFE_SSDG_CVS_R01",
            "route_signature": "safe_ssdg_cvs_r01_training_extra_gpu0",
            "ground_dg_claim_scope": "full_200e_training_candidate_not_completion_claim",
            "source_ssl_split": "0.1L/0.7U/0.2Val",
            "no_target_receiver_in_training": True,
            "cen51_parent_run_or_control": "CEN51_R04_BEST",
            "phase1_non_regression_target": "matched_CEN51_R04",
            "optimization_target": "exceed_matched_CEN51_R04",
            "target_lift_over_cen51": "sat_mean_5_delta_pp>0; sat_floor_5_delta_pp>0; overall_delta_pp>=0; strict_udu_delta_pp>=0; receiver_floor_delta_pp>=0",
            "satellite_channel_primary_metric": True,
            "satellite_channel_lift_target": "deployment-primary star-ground lift: sat_mean_5_delta_pp>0; sat_floor_5_delta_pp>0",
            "CEN51_COMPARABLE": True,
            "pseudo_precision_audit_target": 0.95,
            "pseudo_coverage_is_risk_metric": True,
            "forbid_meta_learning_dg_mainline": True,
            "must_not_regress_floor": "overall>=88.57; strict_udu>=84.87; receiver_floor>=79.53; sat_mean_5>=46.564; sat_floor_5>=41.52",
            "exact_command": "python code/train.py --epochs 200 --use_safe_ssdg_cvs",
            "launchability_status": "DEFERRED_RETRY_CAPACITY_WITH_EXACT_FULL_200E_TRAINING_COMMAND",
        }
    )
    items[8] = _stage2_item(1, "A", 8)

    result = validate(items, 64)
    issue_names = {issue["issue"] for issue in result["issues"]}

    assert result["verdict"] == "FAIL"
    assert "lane_quota_mismatch" not in issue_names
    assert "per_gpu_lane_quota_mismatch" in issue_names


def test_pareto_cli_marks_diagnostic(tmp_path):
    summary = tmp_path / "optimizer_batch_summary_v1.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "optimizer_batch_summary_v1",
                "batch": "CEN54",
                "lane": "centralized",
                "items": [
                    {
                        "candidate_id": "CEN54_R01",
                        "run_name": "good",
                        "status": "finished",
                        "views": {"PRIMARY": {"strict_udu": 84.0, "sat_floor": 41.0, "receiver_floor": 73.0}},
                        "collapse_flags": [],
                    },
                    {
                        "candidate_id": "CEN54_A07",
                        "run_name": "sat_only",
                        "status": "finished",
                        "views": {"PRIMARY": {"strict_udu": 79.0, "sat_floor": 41.2, "receiver_floor": 59.0}},
                        "collapse_flags": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "ranking.json"
    subprocess.run(
        [sys.executable, str(TOOLS / "optimizer_rank_pareto.py"), str(summary), "--output", str(output)],
        check=True,
        cwd=PROJECT_ROOT,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pareto_front"] == ["CEN54_R01"]
    assert payload["diagnostic_only"][0]["candidate_id"] == "CEN54_A07"
