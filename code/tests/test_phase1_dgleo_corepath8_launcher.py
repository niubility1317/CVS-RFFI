from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import build_arg_parser  # noqa: E402
from scripts import launch_phase1_dgleo_corepath8_20260714 as launcher  # noqa: E402


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


def test_corepath_matrix_is_one_same_seed_candidate_per_gpu():
    rows = launcher.build_matrix()
    assert len(rows) == 8
    assert sorted(int(row["gpu"]) for row in rows) == list(range(8))
    assert {int(row["seed"]) for row in rows} == {launcher.SEED}
    assert all(row["checkpoint_selection"] == "final_only" for row in rows)


def test_corepath_control_preserves_legacy_mechanisms_for_attribution():
    args = _parsed(launcher.build_matrix()[0])
    assert args.id_feature_key == "feat_joint"
    assert args.direct_metric_hierarchical_combine == "product"
    assert args.direct_metric_reference_bank is False
    assert args.pseudo_temporal_mode == "batch_neighbor"
    assert args.concat_sat_deduplicate_tx_ce is False
    assert args.os_budget_scope == "all_shared"


def test_corepath_stable_enables_complete_p0_closed_loop():
    args = _parsed(launcher.build_matrix()[6])
    assert args.id_feature_key == "feat_cls"
    assert args.direct_metric_hierarchical_combine == "smooth_min"
    assert args.direct_metric_reference_bank is True
    assert args.direct_metric_gate_reference_detach is True
    assert args.direct_metric_core_tpr_weight >= 5.0
    assert args.direct_metric_source_radius_cap_deg == 18.0
    assert args.source_episode_radius_cap_deg == 18.0
    assert args.source_episode_leave_domain_target_deg == 16.0
    assert args.pseudo_temporal_mode == "epoch_bank"
    assert args.concat_sat_deduplicate_tx_ce is True
    assert args.concat_sat_teacher_clean_only is True
    assert args.os_budget_scope == "zid_path"
    assert args.os_objective_max_scale == 8.0
    assert args.direct_metric_zid_p99_target_deg == 70.0
    assert args.tail_safety_p99_target_deg == 70.0
    assert args.eval_sat_scenarios == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
