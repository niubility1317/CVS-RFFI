#!/usr/bin/env python3
"""Reconcile the mutable Stage2 state with the 2026-07-15 qKNNv42 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHECKPOINT = "/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
CHECKPOINT_SHA256 = "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
TIMESTAMP = "2026-07-15T19:05:00+08:00"
RUN_ID = "qknnv42_extreme_light_control_repair_20260715"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-state", type=Path, required=True)
    parser.add_argument("--git-mirror", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"state root must be an object: {path}")
    return value


def _reconcile(state: dict) -> dict:
    protocol = state["stage2_sample_protocol"]
    protocol["effective_from"] = TIMESTAMP
    protocol["status"] = "ACTIVE"
    protocol["current_policy_status"] = "QKNNV42_EXTREME_LIGHT_PROTOCOL_REPAIR_REQUIRED"
    protocol["base_model_policy"] = {
        "candidate_field_required": "adv3b02_checkpoint_and_sha256",
        "required_base": "ADV3B02_CORE90_SOFT_E200 strict checkpoint",
        "checkpoint_path": CHECKPOINT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "strict_load_required": True,
        "missing_unexpected_shape_mismatch_required": [0, 0, 0],
        "selection_rule": "The same sealed ADV3B02 checkpoint is used by candidate, identity-only, and strict-direct paired streams.",
    }
    protocol["launchable_stage2_required_fields"] = [
        "adv3b02_checkpoint_and_sha256",
        "target_receiver_label",
        "target_old_tx_labels",
        "target_new_tx_labels_real_nested_5_10_20",
        "k_shot in [1,5,10,20]",
        "development_k_shot=10",
        "target_channel_scenarios=leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "phase2_sample_view_policy=leo_weak_only_no_clean_access",
        "clean_sample_access=false",
        "clean_derived_signal_access=false",
        "clean_raw_path_runtime_reachability=false",
        "clean_cache_build_spec_runtime_reachability=false",
        "phase2_query_decision_policy=per_sample_all_registered_classes",
        "phase2_query_role_oracle_access=false",
        "phase2_query_true_batch_class_count_access=false",
        "phase2_query_class_quota_access=false",
        "phase2_query_batch_global_assignment=false",
        "query_label_fit=false",
        "dense_query_graph=false",
        "predict_then_seal_then_independent_score=true",
        "adapter_trainable_params<=50000",
        "adapt_epochs<=20",
        "persistent_state_bytes<=262144",
        "adaptive_tta_views=1_to_3_to_5_per_sample",
        "formal_launch_authority=true_only_after_all_runtime_and_data_evidence_pass",
    ]
    oracle = protocol["phase2_query_oracle_policy"]
    oracle.pop("phase2_query_class_count_access", None)
    oracle["phase2_query_true_batch_class_count_access"] = False
    oracle["registered_class_count_access"] = True
    oracle["query_label_fit"] = False
    oracle["dense_query_graph"] = False
    oracle["predictor_scorer_feedback"] = False
    oracle["status"] = "ACTIVE_HARD_GATE_12_FIELD_RUNTIME_CONTRACT"
    protocol["stage2_a"] = "Frozen strict ADV3B02 direct/identity control on sealed leo_*_weak query artifacts; no target labels or query fitting."
    protocol["stage2_b"] = "Use only sealed leo_*_weak target-old K-shot support for lightweight receiver adaptation; independently predict all registered old classes and seal before scoring."
    protocol["stage2_c"] = "Use sealed leo_*_weak target-old plus real nested target-new K-shot support for seen-new enrollment; independently classify every query across all registered classes without role or quota Oracle."
    protocol["k_shot_policy"] = {
        "status": "ACTIVE_K10_DEVELOPMENT_LOCKED_CONFIRMATION",
        "development_k_shot": 10,
        "confirmation_k_shots": [1, 5, 10, 20],
        "selection_feedback_forbidden_from": [1, 5, 20],
        "k1_requirement": "nonnegative_vs_identity_and_at_least_plus_2pp_vs_strict_direct_overall_and_per_receiver_with_paired_95ci_lower_gt_0",
        "k5_requirement": "matched_metric_drop_vs_k10_le_3pp",
        "forgetting_requirement": "no_worse_than_matched_identity_at_each_k",
    }
    directive = "2026-07-15 user selected strict ADV3B02 plus extreme-light qKNNv42, K10-only development, K1/5/20 locked confirmation, <=50k params, <=20 epochs, <=256KB state, adaptive 1-to-3-to-5 views, and explicit K1/forgetting gates."
    if directive not in protocol["user_directive_local"]:
        protocol["user_directive_local"] += " " + directive

    opgac = state["stage2_opgac_priority_policy"]
    opgac["status"] = "HISTORICAL_COMPARATOR_NO_CURRENT_LAUNCH_AUTHORITY"
    opgac["effective_from"] = TIMESTAMP
    opgac["current_phase2_boundary"] = "Superseded by project.md section 10.3.1 and the 2026-07-15 qKNNv42 strict ADV3B02 objective."
    opgac["route_family_default"] = "QKNNV42_EXTREME_LIGHT_STRICT_ADV3B02"
    opgac["stage2_base_model_id"] = "ADV3B02_CORE90_SOFT_E200"
    opgac["matrix_policy"]["include_opgac_rows_when_phase2_lane_idle_and_phase1_boundary_clears"] = False
    opgac["matrix_policy"]["oa_mse_role"] = "historical_comparator_only"

    capacity = state["lane_capacity_policy"]
    capacity["effective_from"] = TIMESTAMP
    capacity["combined_max_active_per_gpu"] = 2
    capacity["phase1_max_active_per_gpu"] = 1
    capacity["phase2_max_active_per_gpu"] = 2
    capacity["phase2_default_candidates_per_gpu"] = 1
    capacity["stage2_candidate_quota_per_matrix"] = 300
    capacity["user_directive"] = "Current AGENTS.md permits at most two concurrent training experiments per GPU in total; use one qKNN strict smoke per GPU by default and never infer capacity without a fresh live inventory."

    idle = state["idle_lane_execution_policy"]
    idle["status"] = "REPAIR_UNTIL_RUNNER_EXECUTES"
    idle["current_policy_status"] = "ACTIVE_STRICT_QKNN_REPAIR_OR_HARD_PROTOCOL_BLOCKER"
    idle["effective_from"] = TIMESTAMP
    idle["repair_until_runner_executes"] = True
    idle["required_next_action"] = "CLOSE_STRICT_RUNTIME_AND_TARGET_PACKAGE_BLOCKERS_THEN_FRESH_PREFLIGHT"

    state["qknnv42_extreme_light_stage2_policy"] = {
        "schema": "qknnv42_extreme_light_stage2_policy_v1",
        "status": "ACTIVE_LOCAL_PROTOCOL_REPAIR_REQUIRED",
        "effective_from": TIMESTAMP,
        "authority": ["AGENTS.md", "项目.md section 10.3.1", "live user objective"],
        "base_model_id": "ADV3B02_CORE90_SOFT_E200",
        "base_checkpoint": CHECKPOINT,
        "base_checkpoint_sha256": CHECKPOINT_SHA256,
        "target_receivers": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "old_tx_labels": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
        "new_class_counts": [5, 10, 20],
        "development_k": 10,
        "confirmation_k": [1, 5, 10, 20],
        "confirmation_seed_count_min": 5,
        "scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "performance_targets": {
            "k10_old_acc_min": 0.92,
            "k10_min_old_class_acc_min": 0.88,
            "k10_seen_new_acc_min_by_count": {"5": 0.92, "10": 0.90, "20": 0.86},
            "k5_max_drop_pp": 3.0,
            "k1_identity_adaptation_gain_min_pp": 0.0,
            "k1_strict_direct_gain_min_pp": 2.0,
            "k1_paired_95ci_lower_bound_gt_pp": 0.0,
            "forgetting_no_worse_than_identity_each_k": True,
        },
        "resource_caps": {
            "trainable_params_max": 50000,
            "adapt_epochs_max": 20,
            "persistent_state_bytes_max": 262144,
            "dense_query_graph": False,
            "adaptive_tta": "per_sample_1_to_3_to_5",
        },
        "formal_confirmation_shape": {
            "prediction_cells": 300,
            "joint_scenario_rows": 900,
            "formula": "5 receivers x 5 seeds x 4 K x 3 new-count cells; each prediction cell emits 3 scenarios",
        },
        "forbidden": [
            "clean_or_clean_derived_runtime_access",
            "query_role_oracle",
            "query_true_batch_class_count",
            "class_quota",
            "global_assignment",
            "query_label_fit",
            "dense_query_graph",
            "scorer_feedback",
        ],
        "current_candidate": {
            "id": "qknn_ground_effective8_v14",
            "trainable_params": 44048,
            "adapt_epochs": 12,
            "status": "RESOURCE_SHAPE_PASS_BUT_NOT_STRICT_RUNTIME_INTEGRATED",
            "formal_launch_authority": False,
        },
        "formal_blockers": [
            "adapter_head_tta_external_provenance_not_bound",
            "immutable_input_snapshot_or_different_uid_toctou_not_closed",
            "real_n607_landlock_equivalent_strict_smoke_not_passed",
            "sealed_real_target_packages_and_tx_coverage_not_built",
            "effective8_not_integrated_with_unique_strict_request_builder",
        ],
    }

    state["last_updated_local"] = TIMESTAMP
    state["current_run_id"] = RUN_ID
    state["current_launcher"] = "NOT_YET_LAUNCHABLE_STRICT_RUNTIME_REPAIR"
    state["current_matrix"] = "paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json"
    state["current_report"] = "automation_reports/CV-SincNet/qknnv42_extreme_light_optimization_20260715/report.md"
    state["current_validation"] = "LOCAL_DIAGNOSTIC_PIPELINE_117_TESTS_PASS_FORMAL_FALSE"
    state["current_state"] = "LOCAL_PROTOCOL_REPAIR_REQUIRED_FORMAL_LAUNCH_AUTHORITY_FALSE"
    state["required_next_action"] = "BIND_EFFECTIVE8_PROVENANCE_BUILD_SEALED_TARGET_PACKAGES_CLOSE_IMMUTABLE_SNAPSHOT_AND_RUN_LOCAL_STRICT_INTEGRATION_BEFORE_N607_PREFLIGHT"
    state["required_next_turn"] = state["required_next_action"]
    state["latest_qknnv42_control_repair_result"] = {
        "schema": "qknnv42_control_repair_result_v1",
        "timestamp": TIMESTAMP,
        "run_id": RUN_ID,
        "result": "CONTROL_STATE_RECONCILED_LOCAL_ONLY",
        "formal_launch_authority": False,
        "remote_action_performed": False,
        "historical_opgac_authority_revoked": True,
        "fresh_live_monitor_required_before_any_remote_action": True,
    }
    return state


def _write(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def main() -> int:
    args = _parse_args()
    state = _reconcile(_load(args.root_state))
    _write(args.root_state, state)
    _write(args.git_mirror, state)
    print(json.dumps({
        "status": state["current_state"],
        "run_id": state["current_run_id"],
        "formal_launch_authority": state["qknnv42_extreme_light_stage2_policy"]["current_candidate"]["formal_launch_authority"],
        "root_state": str(args.root_state),
        "git_mirror": str(args.git_mirror),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
