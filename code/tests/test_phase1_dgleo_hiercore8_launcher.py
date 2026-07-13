from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import build_arg_parser  # noqa: E402
from scripts import launch_phase1_dgleo_hiercore8_20260713 as launcher  # noqa: E402


def test_hiercore_matrix_is_one_candidate_per_gpu_and_has_ablation():
    assert launcher.dual is not None
    rows = launcher.build_matrix()
    assert len(rows) == 8
    assert sorted(int(row["gpu"]) for row in rows) == list(range(8))
    assert sum(not bool(row["config"]["hierarchy"]) for row in rows) == 1


def test_hiercore_command_enables_direct_p0_mechanisms_from_epoch_one():
    row = launcher.build_matrix()[0]
    command = launcher.build_command(row, root=Path("/tmp/cvs"), python=Path("/tmp/python"), run_id="dry", wisig_pkl=Path("/tmp/ManySig.pkl"), teacher_ckpt=Path("/tmp/teacher.pth"))
    parsed = build_arg_parser().parse_args(command[3:])
    assert parsed.checkpoint_selection == "final_only"
    assert parsed.direct_metric_start_epoch == 1
    assert parsed.source_episode_start_epoch == 1
    assert parsed.direct_metric_hierarchical_class_gate is True
    assert parsed.direct_metric_global_quantile_weight > 0.0
    assert parsed.direct_metric_component_inter_margin_weight > 0.0
    assert parsed.direct_metric_component_overlap_weight > 0.0
    assert parsed.source_episode_local_overlap_weight > 0.0
    assert parsed.source_episode_leave_domain_target_weight > 0.0
    assert parsed.u_direct_include_ambiguous is True
    assert parsed.os_budget_target_reserve > 0.0
    assert parsed.phase1_export_diagnostic_on_block is True
    assert parsed.eval_sat_scenarios == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
