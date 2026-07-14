from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import build_arg_parser  # noqa: E402
from scripts import launch_phase1_dgleo_p0factorial8_20260714 as launcher  # noqa: E402


def _parsed(row):
    command = launcher.build_command(
        row,
        root=Path("/tmp/cvs"),
        python=Path("/tmp/python"),
        run_id="dry",
        wisig_pkl=Path("/tmp/ManySig.pkl"),
        teacher_ckpt=Path("/tmp/teacher.pth"),
    )
    return build_arg_parser().parse_args(command[3:])


def _row(a: bool, b: bool, c: bool):
    return next(
        row
        for row in launcher.build_matrix()
        if row["factors"] == {"A": a, "B": b, "C": c}
    )


def test_matrix_is_complete_factorial_with_fixed_random_gpu_permutation():
    rows = launcher.build_matrix()
    assert len(rows) == 8
    assert len({row["candidate_id"] for row in rows}) == 8
    assert {tuple(row["factors"][name] for name in ("A", "B", "C")) for row in rows} == set(
        itertools.product((False, True), repeat=3)
    )
    assert tuple(int(row["gpu"]) for row in rows) == launcher.GPU_PERMUTATION
    assert launcher.GPU_PERMUTATION == (1, 0, 2, 6, 5, 4, 7, 3)
    assert sorted(launcher.GPU_PERMUTATION) == list(range(8))
    assert launcher.GPU_PERMUTATION != tuple(range(8))
    assert {int(row["seed"]) for row in rows} == {launcher.SEED}


def test_factor_a_changes_positive_first_own_class_core_only():
    off = _parsed(_row(False, False, False))
    on = _parsed(_row(True, False, False))
    assert off.direct_metric_positive_first is False
    assert on.direct_metric_positive_first is True
    assert off.direct_metric_hierarchical_class_gate is False
    assert on.direct_metric_hierarchical_class_gate is True
    assert off.direct_metric_known_coverage_weight == 0.0
    assert on.direct_metric_known_coverage_weight > 0.0
    assert off.direct_metric_virtual_detach is True
    assert on.direct_metric_virtual_detach is False
    assert off.direct_metric_require_effective_negative_grad is False
    assert on.direct_metric_require_effective_negative_grad is True
    assert off.direct_metric_reference_bank is False
    assert on.direct_metric_reference_bank is True
    assert on.direct_metric_core_tpr_weight > 0.0
    assert off.lambda_zid_receiver_invariance == on.lambda_zid_receiver_invariance == 0.0
    assert off.direct_metric_component_inter_margin_weight == on.direct_metric_component_inter_margin_weight == 0.0


def test_factor_b_changes_tx_conditioned_invariance_local_component_and_worst_view_only():
    off = _parsed(_row(False, False, False))
    on = _parsed(_row(False, True, False))
    assert off.lambda_zid_receiver_invariance == 0.0
    assert on.lambda_zid_receiver_invariance > 0.0
    assert on.lambda_zid_day_invariance > 0.0
    assert on.lambda_zid_channel_invariance > 0.0
    assert on.lambda_u_zid_receiver_invariance > 0.0
    assert on.lambda_u_zid_day_invariance > 0.0
    assert on.lambda_u_zid_channel_invariance > 0.0
    assert off.source_episode_local_compact_weight == 0.0
    assert on.source_episode_local_compact_weight > 0.0
    assert on.source_episode_local_invariant_weight > 0.0
    assert on.source_episode_local_inter_weight > 0.0
    assert on.source_episode_local_overlap_weight > 0.0
    assert on.source_episode_local_accept_weight > 0.0
    assert on.source_episode_local_density_weight > 0.0
    assert off.group_ce_mode == "smooth_dro_capped"
    assert on.group_ce_mode == "dual_worst"
    assert on.source_episode_sat_weight > off.source_episode_sat_weight
    assert on.direct_metric_sat_pair_weight > off.direct_metric_sat_pair_weight
    assert off.direct_metric_positive_first is on.direct_metric_positive_first is False
    assert off.os_budget_scope == on.os_budget_scope == "all_shared"


def test_factor_c_changes_query_geometry_zid_budget_and_risk_only():
    off = _parsed(_row(False, False, False))
    on = _parsed(_row(False, False, True))
    assert off.direct_metric_component_inter_margin_weight == 0.0
    assert on.direct_metric_component_inter_margin_weight > 0.0
    assert off.direct_metric_component_overlap_weight == 0.0
    assert on.direct_metric_component_overlap_weight > 0.0
    assert off.os_budget_scope == "all_shared"
    assert on.os_budget_scope == "zid_path"
    assert off.os_eff_min_budget == off.os_eff_max_budget == 0.0
    assert 0.0 < on.os_eff_min_budget < on.os_eff_max_budget
    assert off.os_budget_controller is False
    assert on.os_budget_controller is True
    assert off.os_gradient_surgery is False
    assert on.os_gradient_surgery is True
    assert off.os_objective_budget_controller is False
    assert on.os_objective_budget_controller is True
    assert off.unlabeled_risk_buffer is False
    assert on.unlabeled_risk_buffer is True
    assert off.lambda_risk_energy_out == 0.0
    assert on.lambda_risk_energy_out > 0.0
    assert off.lambda_zid_receiver_invariance == on.lambda_zid_receiver_invariance == 0.0
    assert off.direct_metric_positive_first is on.direct_metric_positive_first is False


@pytest.mark.parametrize("factors", itertools.product((False, True), repeat=3))
def test_all_candidates_share_internal_p0_source_only_final_protocol(factors):
    row = _row(*factors)
    args = _parsed(row)
    assert row["source_only"] is True
    assert row["phase1_proxy_only"] is True
    assert row["checkpoint_selection"] == "final_only"
    assert args.epochs == 120
    assert args.label_epochs == 0
    assert args.pseudo_epochs == 120
    assert args.checkpoint_selection == "final_only"
    assert args.phase1_distribution_audit_only is False
    assert args.phase1_export_diagnostic_on_block is True
    assert args.use_concat_sat_channel_aug is True
    assert args.concat_sat_ce_only is False
    assert args.concat_sat_deduplicate_tx_ce is True
    assert args.concat_sat_teacher_clean_only is True
    assert args.eval_sat_channel is True
    assert args.eval_sat_scenarios == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    assert args.test_eval_start_epoch > args.epochs
    assert args.test_eval_interval == 0
    assert args.test_eval_final_window == 0
    assert args.source_val_heavy_eval_start_epoch == 10
    assert args.source_val_heavy_eval_interval == 10
    assert args.source_val_heavy_eval_final_window == 20
    assert args.source_val_heavy_eval_final_interval == 2
    assert args.u_direct_include_outside_known is False
    assert args.u_outside_stop_gradient is True


def test_one_process_per_gpu_matrix_and_inherited_resource_gate(monkeypatch):
    rows = launcher.build_matrix()
    assert {gpu: sum(int(row["gpu"]) == gpu for row in rows) for gpu in range(8)} == {
        gpu: 1 for gpu in range(8)
    }
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return {"blocked": {}, "gpus": {str(gpu): {} for gpu in range(8)}}

    monkeypatch.setattr(launcher.dual, "gpu_launch_snapshot", snapshot)
    result = launcher.wait_for_gpu_slots(
        run_id="unit_p0factorial",
        max_total_compute_per_gpu=launcher.MAX_TOTAL_COMPUTE_PER_GPU,
        min_free_memory_mib=launcher.MIN_FREE_MEMORY_MIB,
        allow_unrelated_compute=True,
        timeout_seconds=1,
        poll_seconds=1,
    )
    assert result["blocked"] == {}
    assert calls == [
        {
            "run_id": "unit_p0factorial",
            "gpus": list(range(8)),
            "max_total_compute_per_gpu": launcher.MAX_TOTAL_COMPUTE_PER_GPU,
            "min_free_memory_mib": launcher.MIN_FREE_MEMORY_MIB,
            "allow_unrelated_compute": True,
        }
    ]
    payload = launcher.matrix_payload(rows, "unit_p0factorial", launcher.WALL_HOURS)
    assert payload["planned_processes_per_gpu"] == 1
    assert payload["one_candidate_per_gpu"] is True


def test_matrix_validation_rejects_two_processes_on_one_gpu():
    rows = launcher.build_matrix()
    rows[1]["gpu"] = rows[0]["gpu"]
    with pytest.raises(ValueError, match="exactly one candidate process per GPU"):
        launcher.validate_matrix(rows)
