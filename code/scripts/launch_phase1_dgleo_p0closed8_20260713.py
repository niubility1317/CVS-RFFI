from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import launch_phase1_dgleo_dualguard16_20260712 as dual
import launch_phase1_dgleo_jointp0_leoweak8_20260713 as joint


DEFAULT_RUN_ID = "phase1_dgleo_p0closed8_20260713"
DEFAULT_ROOT = dual.DEFAULT_ROOT
DEFAULT_PYTHON = dual.DEFAULT_PYTHON
WALL_HOURS = 10.0
SEED = 713201
LEO_WEAK = "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"


BASE: Dict[str, Any] = deepcopy(joint.BASE)
BASE.update(
    {
        "source_w": 0.080,
        "dm_lambda": 0.040,
        "proxy_w": 0.025,
        "u_dm": 0.020,
        "u_q": 0.022,
        "u_min": 4,
        "os_min_budget": 0.14,
        "os_max_budget": 0.22,
        "os_min_closed_scale": 0.95,
        "objective_boundary": 0.35,
        "objective_source": 0.25,
        "objective_invariant": 0.25,
        "objective_u": 0.15,
        "objective_max_scale": 32.0,
    }
)


CELLS: Sequence[Mapping[str, Any]] = (
    {"cell": "C0_BALANCED", "role": "balanced_objective_budget", "overrides": {}},
    {
        "cell": "C1_SOURCE_HEAVY",
        "role": "source_episode_gradient_pressure",
        "overrides": {"objective_boundary": 0.25, "objective_source": 0.35, "source_w": 0.110},
    },
    {
        "cell": "C2_INVARIANT_HEAVY",
        "role": "receiver_day_invariant_core_pressure",
        "overrides": {
            "objective_boundary": 0.25,
            "objective_invariant": 0.35,
            "l_rx_inv": 0.26,
            "l_day_inv": 0.18,
            "proto_domain": 1.20,
        },
    },
    {
        "cell": "C3_BOUNDARY_ALIGNED",
        "role": "fixed_virtual_differentiable_gate_pressure",
        "overrides": {
            "objective_boundary": 0.40,
            "objective_source": 0.20,
            "dm_lambda": 0.055,
            "proxy_w": 0.035,
        },
    },
    {
        "cell": "C4_U_GEOMETRY",
        "role": "unlabeled_tristate_geometry_pressure",
        "overrides": {
            "objective_boundary": 0.30,
            "objective_source": 0.20,
            "objective_u": 0.25,
            "u_dm": 0.030,
            "u_q": 0.030,
        },
    },
    {
        "cell": "C5_SAT_INVARIANT",
        "role": "clean_sat_receiver_invariant_pressure",
        "overrides": {
            "objective_boundary": 0.25,
            "objective_source": 0.20,
            "objective_invariant": 0.40,
            "l_channel_inv": 0.34,
            "u_channel_inv": 0.24,
            "source_sat_weight": 1.25,
            "pair_weight": 1.25,
        },
    },
    {
        "cell": "C6_INTEGRATED_AGGRESSIVE",
        "role": "high_open_budget_integrated_pressure",
        "overrides": {
            "objective_boundary": 0.30,
            "objective_source": 0.30,
            "os_min_budget": 0.18,
            "os_max_budget": 0.26,
            "source_w": 0.120,
            "dm_lambda": 0.055,
            "objective_max_scale": 48.0,
        },
    },
    {
        "cell": "C7_DG_PROTECTED",
        "role": "closed_gradient_protected_joint_pressure",
        "overrides": {
            "objective_boundary": 0.30,
            "objective_invariant": 0.30,
            "teacher_clean": 2.50,
            "teacher_sat": 1.50,
            "os_min_closed_scale": 0.98,
        },
    },
)


def build_matrix() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gpu, cell in enumerate(CELLS):
        config = deepcopy(BASE)
        config.update(dict(cell.get("overrides", {})))
        rows.append(
            {
                "candidate_id": f"P0C_{cell['cell']}",
                "cell": str(cell["cell"]),
                "role": str(cell["role"]),
                "replicate": 1,
                "seed": SEED,
                "gpu": gpu,
                "source_only": True,
                "phase1_proxy_only": True,
                "checkpoint_selection": "final_only",
                "config": config,
            }
        )
    validate_matrix(rows)
    return rows


def validate_matrix(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 8 or len({str(row["candidate_id"]) for row in rows}) != 8:
        raise ValueError("P0 closure matrix must contain eight unique candidates")
    if Counter(int(row["gpu"]) for row in rows) != Counter({gpu: 1 for gpu in range(8)}):
        raise ValueError("P0 closure matrix must assign exactly one candidate to every GPU")
    for row in rows:
        cfg = row["config"]
        shares = [
            float(cfg["objective_boundary"]),
            float(cfg["objective_source"]),
            float(cfg["objective_invariant"]),
            float(cfg["objective_u"]),
        ]
        if abs(sum(shares) - 1.0) > 1e-8 or min(shares) <= 0.0:
            raise ValueError(f"objective shares must be positive and sum to one: {row['candidate_id']}")
        if not 0.14 <= float(cfg["os_min_budget"]) <= float(cfg["os_max_budget"]) <= 0.26:
            raise ValueError(f"invalid protected open budget: {row['candidate_id']}")


def build_command(
    row: Mapping[str, Any],
    *,
    root: Path,
    python: Path,
    run_id: str,
    wisig_pkl: Path,
    teacher_ckpt: Path,
) -> List[str]:
    command = joint.build_command(
        row,
        root=root,
        python=python,
        run_id=run_id,
        wisig_pkl=wisig_pkl,
        teacher_ckpt=teacher_ckpt,
    )
    cfg = dict(row["config"])
    replacements = {
        "--epochs": 120,
        "--label_epochs": 0,
        "--pseudo_epochs": 120,
        "--direct_metric_virtual_detach": "true",
        "--direct_metric_gate_reference_detach": "false",
        "--direct_metric_warmup_epochs": 2,
        "--source_episode_warmup_epochs": 2,
        "--source_episode_structural_warmup_epochs": 4,
        "--proxy_unknown_warmup_epochs": 2,
        "--u_geometry_all_valid_queries": "true",
        "--u_direct_metric_min_selected": cfg["u_min"],
        "--os_gradient_protect_closed": "true",
        "--os_objective_budget_controller": "true",
        "--os_objective_boundary_share": cfg["objective_boundary"],
        "--os_objective_source_share": cfg["objective_source"],
        "--os_objective_invariant_share": cfg["objective_invariant"],
        "--os_objective_u_share": cfg["objective_u"],
        "--os_objective_min_scale": 0.05,
        "--os_objective_max_scale": cfg["objective_max_scale"],
        "--os_eff_min_budget": cfg["os_min_budget"],
        "--os_eff_max_budget": cfg["os_max_budget"],
        "--os_budget_min_closed_scale": cfg["os_min_closed_scale"],
        "--tail_safety_absolute_violation_drives_state": "false",
        "--tail_safety_training_stop_enabled": "false",
        "--tail_safety_reference_requires_absolute_safe": "false",
        "--eval_sat_on": "all",
        "--sat_train_scenario": "leo_clear_weak",
        "--sat_train_scenarios": LEO_WEAK,
        "--eval_sat_scenarios": LEO_WEAK,
        "--sat_protocol_disjoint_required": "false",
    }
    for name, value in replacements.items():
        dual._set_arg(command, name, value)
    return command


def matrix_payload(rows: Sequence[Mapping[str, Any]], run_id: str, wall_hours: float) -> Dict[str, Any]:
    payload = joint.matrix_payload(rows, run_id, wall_hours)
    payload.update(
        {
            "schema": "phase1_dgleo_p0closed8_matrix_v1",
            "open_gradient_budget_range": [0.14, 0.26],
            "objective_gradient_budgeting": True,
            "tail_training_stop_enabled": False,
            "satellite_eval_scope": "all_named_loaders_with_receiver_floors",
        }
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch eight Phase1 P0 closure experiments, one per GPU.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--wisig-pkl", default="")
    parser.add_argument("--teacher-ckpt", default="")
    parser.add_argument("--wall-hours", type=float, default=WALL_HOURS)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--launch-settle-seconds", type=float, default=3.0)
    parser.add_argument("--emit-matrix", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < float(args.wall_hours) <= WALL_HOURS:
        raise ValueError(f"--wall-hours must be in (0,{WALL_HOURS}]")
    rows = build_matrix()
    if args.emit_matrix:
        Path(args.emit_matrix).write_text(
            json.dumps(matrix_payload(rows, args.run_id, args.wall_hours), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.dry_run:
        root = Path(args.root)
        python = Path(args.python)
        wisig = Path(args.wisig_pkl or root / "Dataset_WigSig" / "ManySig.pkl")
        teacher = Path(
            args.teacher_ckpt
            or root / "runs" / "phase1_adv3_mechanism32_queue_20260701" / "ADV3B02_CORE90_SOFT_E200" / "best_joint_safe_ssdg.pth"
        )
        commands = [
            build_command(row, root=root, python=python, run_id=args.run_id, wisig_pkl=wisig, teacher_ckpt=teacher)
            for row in rows
        ]
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "candidate_count": len(rows),
                    "gpu_total_counts": dict(sorted(Counter(int(row["gpu"]) for row in rows).items())),
                    "unique_command_count": len({tuple(command) for command in commands}),
                    "first_command": shlex.join(commands[0]),
                    "last_command": shlex.join(commands[-1]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    original_build = dual.build_command
    original_payload = dual.matrix_payload
    try:
        dual.build_command = build_command
        dual.matrix_payload = matrix_payload
        return dual.run_matrix(args, rows)
    finally:
        dual.build_command = original_build
        dual.matrix_payload = original_payload


if __name__ == "__main__":
    raise SystemExit(main())
