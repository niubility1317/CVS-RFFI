from __future__ import annotations

from pathlib import Path
import sys

import torch
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import cvsrffi.phase1_v2_control as phase1_v2_control  # noqa: E402
from cvsrffi.phase1_v2_control import (  # noqa: E402
    TailSafetyConfig,
    TailSafetyStateMachine,
    assess_endpoint_contract,
    assess_feasibility_gate,
    assess_open_set_effective_budget,
    assess_phase1_v2_final_export_policy,
    assess_source_episode_density_gate,
    assess_unlabeled_tri_state,
)
from cvsrffi.phase2_prototypes import attach_endpoint_accept_v1_manifest  # noqa: E402
from SSDG.train_ssdg import (  # noqa: E402
    _backward_with_open_set_projection,
    _resolve_phase1_terminal_status,
    build_arg_parser,
    train,
)


def _verified_endpoint_package():
    package = {
        "feature_key": "z_id",
        "fused_tx_prototypes": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        "fused_tx_mask": torch.ones(2, 1, dtype=torch.bool),
        "fusion_accept_policy": "local_component",
        "global_fused_radius_is_accept_region": False,
        "metadata": {
            "source_checkpoint_sha256": "0" * 64,
            "run_id": "unit",
            "candidate_id": "control",
            "known_class_count": 2,
            "class_id_to_tx": ["tx0", "tx1"],
            "logit_class_order": [0, 1],
            "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
            "checkpoint_load_strict": True,
            "endpoint_runtime_entry_parity_digest": "1" * 64,
            "endpoint_runtime_entry_parity_sample_count": 8,
        },
        "fusion_components": [
            [{
                "component_id": 0, "mu": [1.0, 0.0], "source_domains": [0],
                "r_core_deg": 5.0, "r_accept_deg": 8.0, "r_tail_deg": 10.0, "r_vac_deg": 14.0,
                "density_p05": 0.5, "density_p10": 0.4, "nll_p95": 0.8, "nll_tail_p95": 1.0,
                "accept_enabled": True,
            }],
            [{
                "component_id": 0, "mu": [0.0, 1.0], "source_domains": [0],
                "r_core_deg": 5.0, "r_accept_deg": 8.0, "r_tail_deg": 10.0, "r_vac_deg": 14.0,
                "density_p05": 0.5, "density_p10": 0.4, "nll_p95": 0.8, "nll_tail_p95": 1.0,
                "accept_enabled": True,
            }],
        ],
        "endpoint_gate_thresholds": {
            "energy_max_by_class": {"0": 0.0, "1": 0.0},
            "energy_temperature": 1.0,
            "energy_formula_id": "negative_logsumexp_temperature_v1",
            "density_formula_id": "exp_neg_sq_normalized_angle_v1",
            "nll_formula_id": "half_sq_normalized_angle_v1",
            "logit_margin_core_min": 0.5,
            "logit_margin_tail_min": 1.0,
            "geo_margin_core_min_deg": 2.0,
            "geo_margin_tail_min_deg": 4.0,
            "allow_tail_auto_accept": False,
            "use_density_gate": True,
            "use_nll_gate": True,
            "use_energy_gate": True,
            "use_geo_margin_gate": True,
            "reject_nan": True,
            "max_radius_to_inter_ratio": 0.50,
        },
        "endpoint_calibration": {
            "threshold_source": "source_val_only",
            "calibration_split": "source_val",
            "num_samples": 32,
        },
    }
    return attach_endpoint_accept_v1_manifest(package)


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
    package = _verified_endpoint_package()
    decision = assess_endpoint_contract(
        {
            "phase": "Phase1_source_only",
            "endpoint_policy_id": "endpoint_accept_v1",
            "endpoint_accept_boundary_exported": True,
            "endpoint_artifact": package,
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


def test_endpoint_contract_rejects_non_source_val_threshold_or_calibration_split():
    decision = assess_endpoint_contract(
        {
            "phase": "Phase1_source_only",
            "endpoint_policy_id": "endpoint_accept_v1",
            "endpoint_accept_boundary_exported": True,
            "endpoint_threshold_source": "target_query",
            "endpoint_calibration_split": "unknown_query",
            "loss_gate_exported": False,
            "stage2_success_claim": False,
            "deployment_success_claim": False,
        }
    )

    assert decision.fired
    assert "invalid_endpoint_threshold_source" in decision.reason
    assert "invalid_endpoint_calibration_split" in decision.reason


def test_endpoint_contract_rejects_self_reported_manifest_without_package_verification():
    package = _verified_endpoint_package()
    decision = assess_endpoint_contract(
        {
            "phase": "Phase1_source_only",
            "endpoint_policy_id": "endpoint_accept_v1",
            "endpoint_accept_boundary_exported": True,
            "endpoint_threshold_source": "source_val_only",
            "endpoint_calibration_split": "source_val",
            "endpoint_artifact": package["endpoint_accept_v1"],
        }
    )

    assert decision.fired
    assert "endpoint_artifact_unverified_manifest_only" in decision.reason


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
            reference_window=1,
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
    machine.commit_reference(normal)
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
            reference_window=1,
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
    machine.commit_reference(best)
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


def test_tail_state_machine_blocks_final_on_tail_cvar_expansion():
    machine = TailSafetyStateMachine(
        TailSafetyConfig(
            p95_target_deg=90.0,
            p99_target_deg=90.0,
            tail_cvar_target_deg=90.0,
            proxy_vaccept_target=0.90,
            tail_cvar_expansion_block_final_delta=4.0,
            tail_cvar_expansion_block_best_delta=6.0,
            reference_window=1,
        )
    )

    best = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 60.0,
            "train/dm_accept_zid_p99_deg": 80.0,
            "train/dm_accept_zid_tail_cvar_deg": 70.0,
            "train/dm_accept_proxy_vaccept": 0.12,
        }
    )
    machine.commit_reference(best)
    final_block = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 61.0,
            "train/dm_accept_zid_p99_deg": 80.5,
            "train/dm_accept_zid_tail_cvar_deg": 74.5,
            "train/dm_accept_proxy_vaccept": 0.12,
        }
    )

    assert not best.fired
    assert final_block.blocks_final
    assert not final_block.blocks_best
    assert "tail_cvar_expansion_blocks_final" in final_block.reason
    assert final_block.details["tail_expansion_cvar_delta"] == 4.5


def test_tail_state_machine_blocks_promotion_until_robust_reference_window_is_ready():
    machine = TailSafetyStateMachine(TailSafetyConfig(reference_window=3))
    decision = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 50.0,
            "train/dm_accept_zid_p99_deg": 68.0,
            "train/dm_accept_zid_tail_cvar_deg": 53.0,
            "train/dm_accept_proxy_vaccept": 0.30,
        }
    )

    assert decision.state == "INSUFFICIENT"
    assert decision.fired
    assert decision.blocks_best
    assert decision.blocks_final


def test_tail_state_machine_round_trips_committed_reference_for_checkpoint_rollback():
    machine = TailSafetyStateMachine(TailSafetyConfig(reference_window=1))
    safe = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 50.0,
            "train/dm_accept_zid_p99_deg": 68.0,
            "train/dm_accept_zid_tail_cvar_deg": 53.0,
            "train/dm_accept_proxy_vaccept": 0.30,
        }
    )
    machine.commit_reference(safe)
    state = machine.state_dict()
    restored = TailSafetyStateMachine(TailSafetyConfig(reference_window=1))
    restored.load_state_dict(state)
    restored.acknowledge_rollback()

    assert restored.best_p99 == 68.0
    assert restored.best_tail_cvar == 53.0
    assert restored.reference_best == machine.reference_best
    assert restored.state == "NORMAL"


def test_tail_reference_cannot_staircase_to_worse_p99_under_small_step_deltas():
    machine = TailSafetyStateMachine(
        TailSafetyConfig(
            reference_window=1,
            p95_target_deg=90.0,
            p99_target_deg=90.0,
            tail_cvar_target_deg=90.0,
            proxy_vaccept_target=0.90,
        )
    )
    first = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 45.0,
            "train/dm_accept_zid_p99_deg": 50.0,
            "train/dm_accept_zid_tail_cvar_deg": 52.0,
            "train/dm_accept_proxy_vaccept": 0.30,
        }
    )
    machine.commit_reference(first)
    worse = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 40.0,
            "train/dm_accept_zid_p99_deg": 51.9,
            "train/dm_accept_zid_tail_cvar_deg": 53.0,
            "train/dm_accept_proxy_vaccept": 0.20,
        }
    )

    assert worse.details["tail_reference_improved"] == 0.0
    assert machine.best_p99 == 50.0


def test_tail_safety_blocks_checkpoint_when_only_p99_is_over_target():
    machine = TailSafetyStateMachine(
        TailSafetyConfig(
            p95_target_deg=80.0,
            p99_target_deg=70.0,
            tail_cvar_target_deg=90.0,
            proxy_vaccept_target=0.9,
            reference_window=1,
            warning_patience=3,
        )
    )
    decision = machine.update(
        {
            "train/dm_accept_zid_p95_deg": 50.0,
            "train/dm_accept_zid_p99_deg": 71.0,
            "train/dm_accept_zid_tail_cvar_deg": 60.0,
            "train/dm_accept_proxy_vaccept": 0.2,
        }
    )

    assert decision.fired
    assert decision.blocks_best
    assert decision.blocks_final
    assert "p99_over" in decision.reason


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


def test_open_set_effective_budget_prefers_effective_gradient_norms_over_large_zero_grad_loss():
    decision = assess_open_set_effective_budget(
        {
            "train/w_loss_direct_metric_accept": 100.0,
            "train/w_loss_tx_labeled": 1.0,
            "train/os_gradient_effective_open_norm": 0.0,
            "train/os_gradient_balanced_closed_norm": 2.0,
        },
        min_budget=0.15,
    )

    assert decision.fired
    assert decision.details["B_os_eff"] == 0.0
    assert decision.details["B_os_uses_gradient_norm"] == 1.0


def test_open_set_gradient_controller_scales_real_gradients_to_budget():
    model = torch.nn.Linear(2, 1, bias=False)
    x = torch.tensor([[1.0, 0.5]])
    output = model(x)
    closed = (output - 1.0).pow(2).mean()
    opened = 0.001 * (output + 1.0).pow(2).mean()
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    info = _backward_with_open_set_projection(
        model,
        scaler,
        closed,
        opened,
        project_conflicts=True,
        budget_controller=True,
        min_budget=0.20,
        max_os_scale=1000.0,
        min_closed_scale=0.10,
    )

    assert info["pre_budget"] < 0.20
    assert info["post_budget"] >= 0.20 - 1e-6
    assert info["os_scale"] > 1.0
    assert model.weight.grad is not None


def test_open_set_gradient_budget_ignores_closed_only_head_gradients():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Parameter(torch.tensor([1.0]))
            self.closed_head = torch.nn.Parameter(torch.tensor([1.0]))

    model = Toy()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    closed_loss = model.shared.square().sum() + 1000.0 * model.closed_head.square().sum()
    open_loss = model.shared.square().sum()

    info = _backward_with_open_set_projection(
        model,
        scaler,
        closed_loss,
        open_loss,
        project_conflicts=False,
        budget_controller=False,
    )

    assert info["shared_param_count"] == 1.0
    assert info["budget_scope_shared_zid_path"] == 1.0
    assert info["pre_budget"] == pytest.approx(0.5)
    assert info["total_closed_grad_norm"] > 100.0 * info["closed_grad_norm"]


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
            "train/u_tri_state_source": "geometry",
            "train/u_tri_query_count": 48.0,
            "train/u_tri_trusted_core_count": 20.0,
            "train/u_tri_ambiguous_tail_count": 16.0,
            "train/u_tri_outside_reject_count": 12.0,
            "train/u_tri_class_coverage": 6.0,
            "train/u_tri_domain_coverage": 4.0,
            "train/u_tri_pair_disagreement_rate": 0.10,
            "train/u_tri_pseudo_component_agreement_rate": 0.90,
        },
        required=True,
        min_selected=16,
    )

    assert not decision.fired
    assert decision.details["u_tri_trusted_core_count"] == 20.0
    assert decision.details["u_tri_ambiguous_tail_count"] == 16.0
    assert decision.details["u_tri_outside_reject_count"] == 12.0


def test_unlabeled_tri_state_required_rejects_fallback_counts():
    decision = assess_unlabeled_tri_state(
        {
            "train/w_loss_u_direct_metric_accept": 0.03,
            "train/u_dm_accept_active": 1.0,
            "train/u_dm_accept_selected": 48.0,
            "train/u_tri_state_source": "fallback",
            "train/u_tri_query_count": 48.0,
            "train/u_tri_trusted_core_count": 20.0,
            "train/u_tri_ambiguous_tail_count": 16.0,
            "train/u_tri_outside_reject_count": 12.0,
            "train/u_tri_class_coverage": 6.0,
            "train/u_tri_domain_coverage": 4.0,
            "train/u_tri_pair_disagreement_rate": 0.10,
            "train/u_tri_pseudo_component_agreement_rate": 0.90,
        },
        required=True,
        min_selected=16,
    )

    assert decision.fired
    assert "US_TRI_STATE_NOT_GEOMETRY" in decision.reason


def test_unlabeled_tri_state_required_checks_query_count_conservation():
    decision = assess_unlabeled_tri_state(
        {
            "train/w_loss_u_direct_metric_accept": 0.03,
            "train/u_dm_accept_active": 1.0,
            "train/u_dm_accept_selected": 48.0,
            "train/u_tri_state_source": "geometry",
            "train/u_tri_query_count": 50.0,
            "train/u_tri_trusted_core_count": 20.0,
            "train/u_tri_ambiguous_tail_count": 16.0,
            "train/u_tri_outside_reject_count": 12.0,
            "train/u_tri_class_coverage": 6.0,
            "train/u_tri_domain_coverage": 4.0,
            "train/u_tri_pair_disagreement_rate": 0.10,
            "train/u_tri_pseudo_component_agreement_rate": 0.90,
        },
        required=True,
        min_selected=16,
    )

    assert decision.fired
    assert "US_TRI_STATE_COUNT_MISMATCH" in decision.reason


@pytest.mark.parametrize(
    ("counts", "reason"),
    [
        ((0.0, 10.0, 90.0), "US_TRI_STATE_CORE_COLLAPSE"),
        ((100.0, 0.0, 0.0), "US_TRI_STATE_ALL_CORE_DEGENERATE"),
    ],
)
def test_unlabeled_tri_state_rejects_degenerate_routes(counts, reason):
    core, ambiguous, outside = counts
    decision = assess_unlabeled_tri_state(
        {
            "train/w_loss_u_direct_metric_accept": 0.03,
            "train/u_dm_accept_active": 1.0,
            "train/u_dm_accept_selected": 32.0,
            "train/u_tri_state_source": "geometry",
            "train/u_tri_query_count": 100.0,
            "train/u_tri_trusted_core_count": core,
            "train/u_tri_ambiguous_tail_count": ambiguous,
            "train/u_tri_outside_reject_count": outside,
            "train/u_tri_class_coverage": 6.0,
            "train/u_tri_domain_coverage": 4.0,
            "train/u_tri_pair_disagreement_rate": 0.10,
            "train/u_tri_pseudo_component_agreement_rate": 0.90,
        },
        required=True,
        min_selected=16,
    )

    assert decision.fired
    assert reason in decision.reason


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


def test_source_episode_density_gate_requires_quantile_and_density_evidence():
    decision = assess_source_episode_density_gate(
        {
            "train/source_episode_overflow_rate": 0.40,
            "train/source_episode_receiver_local_component_count": 4.0,
            "train/source_episode_core_tail_outside_ready": 1.0,
            "train/source_episode_density_gate_active": 1.0,
            "train/source_episode_zid_p95_deg": float("nan"),
            "train/source_episode_zid_p99_deg": float("nan"),
            "train/source_episode_zid_tail_cvar_deg": float("nan"),
        },
        overflow_warn=0.90,
        min_local_components=2,
    )

    assert decision.fired
    assert "SOURCE_EPISODE_QUANTILES_MISSING" in decision.reason


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


def test_phase1_v2_final_export_policy_blocks_non_tail_guard_failures():
    decision = assess_phase1_v2_final_export_policy(
        [
            "B_os_eff_below_min",
            "US_DIRECT_LOSS_IDLE",
            "SOURCE_EPISODE_OVERFLOW_HIGH",
        ],
        tail_blocks_final=False,
    )

    assert decision.fired
    assert "phase1_v2_guard_blocks_final_export" in decision.reason
    assert decision.details["final_export_allowed"] == 0.0


def test_final_export_skip_does_not_depend_on_tail_stop_flag_after_guard_block():
    assert hasattr(phase1_v2_control, "should_skip_phase1_v2_final_export")
    assert phase1_v2_control.should_skip_phase1_v2_final_export(
        phase1_v2_final_blocked=True,
        tail_stop_blocks_final=False,
    )


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
            "--tail_safety_cvar_expansion_block_final_delta",
            "4.0",
            "--tail_safety_cvar_expansion_block_best_delta",
            "6.0",
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
    assert args.tail_safety_cvar_expansion_block_final_delta == 4.0
    assert args.tail_safety_cvar_expansion_block_best_delta == 6.0
    assert args.os_eff_min_budget == 0.20
    assert args.u_tri_state_required is True
    assert args.source_episode_density_gate is True
    assert args.feasibility_gate is True


def test_phase1_source_val_selection_rejects_heldout_joint_best_even_in_dry_run():
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "runs/tmp",
            "--dry_run",
            "--best_metric",
            "joint_safe",
            "--phase1_source_val_selection_only",
            "true",
        ]
    )

    with pytest.raises(ValueError, match="source-only checkpoint selection"):
        train(args)


def test_phase1_source_val_selection_cannot_be_disabled_even_in_dry_run():
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "runs/tmp",
            "--dry_run",
            "--best_metric",
            "clean_val_tx",
            "--phase1_source_val_selection_only",
            "false",
        ]
    )

    with pytest.raises(ValueError, match="must remain true"):
        train(args)


def test_u_tri_state_required_cannot_bypass_all_query_geometry_routing():
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "runs/tmp",
            "--dry_run",
            "--u_tri_state_required",
            "true",
            "--u_geometry_all_valid_queries",
            "false",
        ]
    )

    with pytest.raises(ValueError, match="cannot bypass geometry-first routing"):
        train(args)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ({"tail_stopped": True, "export_failed": True, "final_blocked": True,
          "selected_checkpoint_exists": False, "heldout_eval_status": "FAILED"}, "STOPPED_TAIL"),
        ({"tail_stopped": False, "export_failed": True, "final_blocked": True,
          "selected_checkpoint_exists": False, "heldout_eval_status": "FAILED"}, "FAILED_EXPORT"),
        ({"tail_stopped": False, "export_failed": False, "final_blocked": True,
          "selected_checkpoint_exists": True, "heldout_eval_status": "COMPLETE"}, "NON_PROMOTABLE_GUARD_BLOCKED"),
        ({"tail_stopped": False, "export_failed": False, "final_blocked": False,
          "selected_checkpoint_exists": False, "heldout_eval_status": "NOT_RUN"}, "NO_SAFE_CHECKPOINT"),
        ({"tail_stopped": False, "export_failed": False, "final_blocked": False,
          "selected_checkpoint_exists": True, "heldout_eval_status": "FAILED"}, "HELDOUT_EVAL_INCOMPLETE"),
        ({"tail_stopped": False, "export_failed": False, "final_blocked": False,
          "selected_checkpoint_exists": True, "heldout_eval_status": "COMPLETE",
          "p0_mechanisms_ready": False}, "NON_PROMOTABLE_P0_DISABLED"),
        ({"tail_stopped": False, "export_failed": False, "final_blocked": False,
          "selected_checkpoint_exists": True, "heldout_eval_status": "COMPLETE",
          "endpoint_export_ready": False}, "NON_PROMOTABLE_ENDPOINT_NOT_EXPORTED"),
        ({"tail_stopped": False, "export_failed": False, "final_blocked": False,
          "selected_checkpoint_exists": True, "heldout_eval_status": "COMPLETE"}, "COMPLETE"),
    ],
)
def test_phase1_terminal_status_is_fail_closed(inputs, expected):
    assert _resolve_phase1_terminal_status(**inputs) == expected


def test_phase1_training_loop_never_reads_heldout_test_views():
    source = (CODE_ROOT / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

    assert "test_ran_this_epoch = False" in source
    assert "should_run_training_test(" not in source


def test_batch_telemetry_append_remains_inside_training_batch_loop():
    source = (CODE_ROOT / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")
    line = next(row for row in source.splitlines() if "epoch_logs.append(" in row)

    assert line.startswith("            epoch_logs.append(")


def test_u_tri_state_route_defers_tail_reference_and_promotion_until_pseudo_stage():
    source = (CODE_ROOT / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

    assert 'promotion_stage_ready = not bool(getattr(args, "u_tri_state_required", False)) or phase == "pseudo"' in source
    assert 'phase1_v2_reasons.append("US_STAGE_NOT_READY")' in source


def test_u_geometry_route_filters_pseudo_ce_entropy_and_direct_metric_to_paired_core():
    source = (CODE_ROOT / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

    assert "routed_pseudo_mask = mask & u_geometry_core_mask" in source
    assert "entropy_per_sample[u_geometry_core_mask].mean()" in source
    assert "dm_mask = dm_mask & u_geometry_core_mask" in source
    assert "combined_core = torch.stack([row[0] for row in state_rows], dim=0).all(dim=0)" in source
    assert "combined_outside = torch.stack([row[2] for row in state_rows], dim=0).any(dim=0)" in source
    assert "combined_ambiguous = (~combined_core) & (~combined_outside)" in source
