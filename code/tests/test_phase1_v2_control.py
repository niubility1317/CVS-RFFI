from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_v2_control import (  # noqa: E402
    TailSafetyConfig,
    TailSafetyStateMachine,
    assess_endpoint_contract,
    assess_feasibility_gate,
    assess_open_set_effective_budget,
    assess_source_episode_density_gate,
    assess_unlabeled_tri_state,
)
from SSDG.train_ssdg import build_arg_parser  # noqa: E402


def test_endpoint_contract_fails_closed_when_proxy_soft_gate_is_exported_as_final_boundary():
    decision = assess_endpoint_contract(
        {
            "phase": "Phase1_source_only",
            "endpoint_policy_id": "",
            "loss_gate_exported": True,
            "phase1_proxy_vaccept": 0.63,
            "final_accept_rate": 0.91,
            "unknown_FAR": 0.04,
            "stage2_success_claim": False,
            "deployment_success_claim": False,
        }
    )

    assert decision.fired
    assert "missing_endpoint_accept_v1" in decision.reason
    assert "loss_gate_exported" in decision.reason
    assert "phase1_claim_contains_real_unknown_metric" in decision.reason


def test_endpoint_contract_accepts_phase1_proxy_only_with_endpoint_v1_and_no_real_unknown_claim():
    decision = assess_endpoint_contract(
        {
            "phase": "Phase1_source_only",
            "endpoint_policy_id": "endpoint_accept_v1",
            "endpoint_accept_boundary_exported": True,
            "endpoint_threshold_source": "source_val_only",
            "endpoint_calibration_split": "source_val",
            "loss_gate_exported": False,
            "phase1_proxy_only": True,
            "real_unknown_eval_available": False,
            "phase1_proxy_vaccept": 0.33,
            "stage2_success_claim": False,
            "deployment_success_claim": False,
        }
    )

    assert not decision.fired
    assert decision.details["endpoint_contract_pass"] == 1.0


def test_tail_state_machine_blocks_late_tail_expansion_after_warning_and_rollback():
    machine = TailSafetyStateMachine(
        TailSafetyConfig(
            p95_target_deg=54.0,
            p99_target_deg=70.0,
            tail_cvar_target_deg=56.0,
            proxy_vaccept_target=0.35,
            warning_patience=1,
            rollback_patience=1,
            max_rollbacks=1,
        )
    )

    normal = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 50.0,
            "train/dm_accept_zid_p99_deg": 68.0,
            "train/dm_accept_zid_tail_cvar_deg": 53.0,
            "train/dm_accept_proxy_vaccept": 0.30,
        }
    )
    warning = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 63.0,
            "train/dm_accept_zid_p99_deg": 76.0,
            "train/dm_accept_zid_tail_cvar_deg": 66.0,
            "train/dm_accept_proxy_vaccept": 0.45,
        }
    )
    rollback = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 65.0,
            "train/dm_accept_zid_p99_deg": 78.0,
            "train/dm_accept_zid_tail_cvar_deg": 68.0,
            "train/dm_accept_proxy_vaccept": 0.48,
        }
    )
    stop = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 66.0,
            "train/dm_accept_zid_p99_deg": 79.0,
            "train/dm_accept_zid_tail_cvar_deg": 69.0,
            "train/dm_accept_proxy_vaccept": 0.50,
        }
    )

    assert normal.state == "NORMAL"
    assert warning.state == "WARNING"
    assert rollback.action == "ROLLBACK"
    assert stop.state == "STOP"
    assert stop.blocks_best
    assert stop.blocks_final


def test_tail_state_machine_blocks_best_and_final_on_best_p99_to_current_expansion():
    machine = TailSafetyStateMachine(
        TailSafetyConfig(
            p95_target_deg=90.0,
            p99_target_deg=90.0,
            tail_cvar_target_deg=90.0,
            proxy_vaccept_target=0.90,
            p99_expansion_block_best_delta=3.5,
            p99_expansion_block_final_delta=2.0,
        )
    )

    best = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 60.0,
            "train/dm_accept_zid_p99_deg": 80.0,
            "train/dm_accept_zid_tail_cvar_deg": 72.0,
            "train/dm_accept_proxy_vaccept": 0.12,
        }
    )
    final_block = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 62.0,
            "train/dm_accept_zid_p99_deg": 82.4,
            "train/dm_accept_zid_tail_cvar_deg": 74.0,
            "train/dm_accept_proxy_vaccept": 0.14,
        }
    )
    promotion_block = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 63.0,
            "train/dm_accept_zid_p99_deg": 84.0,
            "train/dm_accept_zid_tail_cvar_deg": 76.0,
            "train/dm_accept_proxy_vaccept": 0.15,
        }
    )

    assert not best.fired
    assert final_block.blocks_final
    assert not final_block.blocks_best
    assert "tail_expansion_blocks_final" in final_block.reason
    assert promotion_block.blocks_best
    assert promotion_block.blocks_final
    assert "tail_expansion_blocks_promotion" in promotion_block.reason
    assert promotion_block.details["tail_expansion_p99_delta"] == 4.0


def test_open_set_effective_budget_uses_weighted_loss_families_and_fails_closed_when_os_is_too_small():
    decision = assess_open_set_effective_budget(
        {
            "train/w_loss_tx_labeled": 1.0,
            "train/w_loss_sat_cls_labeled": 0.8,
            "train/w_loss_teacher_clean_kl": 0.4,
            "train/w_loss_direct_metric_accept": 0.02,
            "train/w_loss_source_episode": 0.004,
            "train/w_loss_proxy_unknown": 0.01,
        },
        min_budget=0.15,
    )

    assert decision.fired
    assert decision.details["B_os_eff"] < 0.15
    assert "B_os_eff_below_min" in decision.reason


def test_unlabeled_tri_state_marks_idle_direct_branch_non_promotable():
    decision = assess_unlabeled_tri_state(
        {
            "train/w_loss_u_direct_metric_accept": 0.0,
            "train/u_dm_accept_active": 0.0,
            "train/u_dm_accept_selected": 0.0,
            "train/pseudo_selected": 24,
        },
        required=True,
        min_selected=16,
    )

    assert decision.fired
    assert "US_DIRECT_LOSS_IDLE" in decision.reason
    assert decision.details["promotable"] == 0.0


def test_unlabeled_tri_state_requires_named_core_tail_outside_counts():
    decision = assess_unlabeled_tri_state(
        {
            "train/w_loss_u_direct_metric_accept": 0.03,
            "train/u_dm_accept_active": 1.0,
            "train/u_dm_accept_selected": 48.0,
            "train/u_tri_trusted_core_count": 20.0,
            "train/u_tri_ambiguous_tail_count": 16.0,
            "train/u_tri_outside_reject_count": 12.0,
        },
        required=True,
        min_selected=16,
    )

    assert not decision.fired
    assert decision.details["u_tri_trusted_core_count"] == 20.0
    assert decision.details["u_tri_ambiguous_tail_count"] == 16.0
    assert decision.details["u_tri_outside_reject_count"] == 12.0


def test_source_episode_density_gate_blocks_global_overflow_without_local_components():
    decision = assess_source_episode_density_gate(
        {
            "train/source_episode_overflow_rate": 0.972,
            "train/source_episode_receiver_local_component_count": 0.0,
            "train/source_episode_core_tail_outside_ready": 0.0,
            "train/source_episode_density_gate_active": 0.0,
        },
        overflow_warn=0.90,
    )

    assert decision.fired
    assert "SOURCE_EPISODE_OVERFLOW_HIGH" in decision.reason
    assert "RECEIVER_AWARE_LOCAL_COMPONENT_MISSING" in decision.reason
    assert "CORE_TAIL_OUTSIDE_NOT_READY" in decision.reason
    assert "SOURCE_EPISODE_DENSITY_GATE_INACTIVE" in decision.reason


def test_feasibility_gate_stops_full_target_when_relaxed_stage_is_unreachable():
    decision = assess_feasibility_gate(
        {
            "stage": "full",
            "relaxed_pass": False,
            "loss_response_slope": 0.0,
            "overflow_excess_cvar95_delta": 0.01,
        }
    )

    assert decision.fired
    assert "RELAXED_UNREACHABLE_STOP_FULL_TARGET" in decision.reason


def test_train_parser_exposes_phase1_v2_hard_gate_args():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--output_dir",
            "runs/tmp",
            "--phase1_v2_hard_gates",
            "true",
            "--endpoint_accept_policy_id",
            "endpoint_accept_v1",
            "--tail_safety_state_machine",
            "true",
            "--tail_safety_p99_expansion_block_final_delta",
            "2.0",
            "--tail_safety_p99_expansion_block_best_delta",
            "3.5",
            "--os_eff_min_budget",
            "0.20",
            "--u_tri_state_required",
            "true",
            "--source_episode_density_gate",
            "true",
            "--feasibility_gate",
            "true",
        ]
    )

    assert args.phase1_v2_hard_gates is True
    assert args.endpoint_accept_policy_id == "endpoint_accept_v1"
    assert args.tail_safety_state_machine is True
    assert args.tail_safety_p99_expansion_block_final_delta == 2.0
    assert args.tail_safety_p99_expansion_block_best_delta == 3.5
    assert args.os_eff_min_budget == 0.20
    assert args.u_tri_state_required is True
    assert args.source_episode_density_gate is True
    assert args.feasibility_gate is True
