from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts import launch_phase1_dgleo_p0closed8_20260713 as launcher  # noqa: E402


def test_p0closed8_matrix_is_one_candidate_per_gpu_with_normalized_objective_shares():
    rows = launcher.build_matrix()
    assert len(rows) == 8
    assert sorted(int(row["gpu"]) for row in rows) == list(range(8))
    assert len({row["candidate_id"] for row in rows}) == 8
    for row in rows:
        cfg = row["config"]
        assert abs(
            cfg["objective_boundary"]
            + cfg["objective_source"]
            + cfg["objective_invariant"]
            + cfg["objective_u"]
            - 1.0
        ) < 1e-8


def test_p0closed8_commands_enable_all_p0_closure_mechanisms():
    rows = launcher.build_matrix()
    command = launcher.build_command(
        rows[0],
        root=Path("/tmp/cvs"),
        python=Path("/tmp/python"),
        run_id="dry",
        wisig_pkl=Path("/tmp/ManySig.pkl"),
        teacher_ckpt=Path("/tmp/teacher.pth"),
    )
    text = " ".join(str(token) for token in command)
    for required in (
        "--epochs 120",
        "--checkpoint_selection final_only",
        "--direct_metric_virtual_detach true",
        "--direct_metric_gate_reference_detach false",
        "--os_gradient_protect_closed true",
        "--os_objective_budget_controller true",
        "--u_geometry_all_valid_queries true",
        "--tail_safety_training_stop_enabled false",
        "--tail_safety_absolute_violation_drives_state false",
        "--tail_safety_reference_requires_absolute_safe false",
        "--eval_sat_on all",
        "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
    ):
        assert required in text
    from SSDG.train_ssdg import build_arg_parser

    parsed = build_arg_parser().parse_args(command[3:])
    assert parsed.epochs == 120
    assert parsed.os_objective_budget_controller is True
    assert parsed.direct_metric_gate_reference_detach is False
    assert parsed.eval_sat_on == "all"


def test_p0closed8_dry_run_emits_eight_unique_commands():
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "code/scripts/launch_phase1_dgleo_p0closed8_20260713.py"), "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["candidate_count"] == 8
    assert payload["unique_command_count"] == 8
    assert payload["gpu_total_counts"] == {str(gpu): 1 for gpu in range(8)}


def test_capacity_queue_dry_run_binds_verified_launcher_without_polling_gpu():
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "code/scripts/queue_phase1_dgleo_p0closed8_20260713.py"),
            "--dry-run",
            "--root",
            str(PROJECT_ROOT),
            "--python",
            sys.executable,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["launch_command"][0] == str(Path(sys.executable).resolve())
    assert payload["launch_command"][1].endswith("launch_phase1_dgleo_p0closed8_20260713.py")
    assert payload["launch_command"][-2:] == ["--wall-hours", "10.0"]
