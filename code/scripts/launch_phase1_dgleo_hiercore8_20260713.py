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

import launch_phase1_dgleo_p0closed8_20260713 as previous

# The capacity-aware scheduler consumes the shared process helpers through the
# launcher module. Keep this explicit alias so wrapper launchers remain drop-in.
dual = previous.dual


DEFAULT_RUN_ID = "phase1_dgleo_hiercore8_20260713"
DEFAULT_ROOT = previous.DEFAULT_ROOT
DEFAULT_PYTHON = previous.DEFAULT_PYTHON
WALL_HOURS = 10.0
SEED = 713301
LEO_WEAK = previous.LEO_WEAK


BASE: Dict[str, Any] = deepcopy(previous.BASE)
BASE.update(
    {
        "source_w": 0.10,
        "dm_lambda": 0.05,
        "proxy_w": 0.025,
        "u_dm": 0.025,
        "u_q": 0.025,
        "os_min_budget": 0.16,
        "os_max_budget": 0.24,
        "os_reserve": 0.015,
        "source_center": 0.45,
        "source_overlap": 0.30,
        "source_leave": 0.35,
        "dm_global": 0.80,
        "dm_inter": 0.70,
        "dm_overlap": 0.70,
        "u_tail_pair": 0.25,
        "u_outside_pair": 0.15,
        "hierarchy": True,
        "u_ambiguous": True,
        "group_mode": "dual_worst",
    }
)


CELLS: Sequence[Mapping[str, Any]] = (
    {"cell": "H0_FULL_STABLE", "role": "full_hierarchical_joint_stable", "overrides": {}},
    {
        "cell": "H1_CORE_STRONG",
        "role": "invariant_core_and_leave_domain_cvar",
        "overrides": {"source_center": 0.70, "source_leave": 0.55, "objective_source": 0.32, "objective_boundary": 0.28},
    },
    {
        "cell": "H2_COMPONENT_SAFE",
        "role": "nearest_component_inter_and_overlap",
        "overrides": {"source_overlap": 0.55, "dm_inter": 1.10, "dm_overlap": 1.10},
    },
    {
        "cell": "H3_BOUNDARY_PARITY",
        "role": "global_quantile_and_hierarchical_boundary",
        "overrides": {"dm_global": 1.30, "dm_inter": 0.90, "objective_boundary": 0.38, "objective_source": 0.22},
    },
    {
        "cell": "H4_U_TRI_ACTIVE",
        "role": "trusted_core_ambiguous_tail_outside_pairing",
        "overrides": {"u_dm": 0.04, "u_q": 0.035, "u_tail_pair": 0.45, "u_outside_pair": 0.25, "objective_boundary": 0.30, "objective_u": 0.25, "objective_source": 0.20},
    },
    {
        "cell": "H5_DG_SAT_FLOOR",
        "role": "dual_worst_group_and_satellite_invariance",
        "overrides": {"l_channel_inv": 0.34, "u_channel_inv": 0.24, "pair_weight": 1.35, "source_sat_weight": 1.30, "objective_invariant": 0.35, "objective_boundary": 0.25},
    },
    {
        "cell": "H6_FULL_AGGRESSIVE",
        "role": "full_hierarchical_high_open_budget",
        "overrides": {"source_w": 0.14, "dm_lambda": 0.07, "os_min_budget": 0.20, "os_max_budget": 0.28, "os_reserve": 0.02, "source_center": 0.75, "source_overlap": 0.55, "source_leave": 0.60},
    },
    {
        "cell": "H7_NO_HIERARCHY_ABL",
        "role": "hierarchical_class_gate_ablation",
        "overrides": {"hierarchy": False, "dm_global": 0.0},
    },
)


def build_matrix() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gpu, cell in enumerate(CELLS):
        config = deepcopy(BASE)
        config.update(dict(cell.get("overrides", {})))
        rows.append(
            {
                "candidate_id": f"HC_{cell['cell']}",
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
    if len(rows) != 8 or Counter(int(row["gpu"]) for row in rows) != Counter(range(8)):
        raise ValueError("hiercore matrix must contain exactly one candidate per GPU")
    for row in rows:
        cfg = row["config"]
        shares = [float(cfg[key]) for key in ("objective_boundary", "objective_source", "objective_invariant", "objective_u")]
        if abs(sum(shares) - 1.0) > 1e-8 or min(shares) <= 0.0:
            raise ValueError(f"objective shares invalid: {row['candidate_id']}")
        if not 0.16 <= float(cfg["os_min_budget"]) <= float(cfg["os_max_budget"]) <= 0.28:
            raise ValueError(f"open gradient budget invalid: {row['candidate_id']}")


def build_command(row: Mapping[str, Any], *, root: Path, python: Path, run_id: str, wisig_pkl: Path, teacher_ckpt: Path) -> List[str]:
    command = previous.build_command(row, root=root, python=python, run_id=run_id, wisig_pkl=wisig_pkl, teacher_ckpt=teacher_ckpt)
    cfg = dict(row["config"])
    replacements = {
        "--source_episode_local_center_target_deg": 8.0,
        "--source_episode_local_invariant_weight": cfg["source_center"],
        "--source_episode_local_inter_margin_deg": 48.0,
        "--source_episode_local_overlap_weight": cfg["source_overlap"],
        "--source_episode_local_overlap_margin_deg": 5.0,
        "--source_episode_leave_domain_target_deg": 42.0,
        "--source_episode_leave_domain_target_weight": cfg["source_leave"],
        "--source_episode_structural_cvar_alpha": 0.20,
        "--direct_metric_hierarchical_class_gate": str(bool(cfg["hierarchy"])).lower(),
        "--direct_metric_global_quantile_weight": cfg["dm_global"],
        "--direct_metric_component_inter_margin_weight": cfg["dm_inter"],
        "--direct_metric_component_inter_margin_deg": 48.0,
        "--direct_metric_component_overlap_weight": cfg["dm_overlap"],
        "--direct_metric_component_overlap_margin_deg": 5.0,
        "--u_direct_include_ambiguous": str(bool(cfg["u_ambiguous"])).lower(),
        "--u_tri_tail_pair_weight": cfg["u_tail_pair"],
        "--u_tri_outside_pair_weight": cfg["u_outside_pair"],
        "--u_tri_tail_pair_target_deg": 10.0,
        "--u_tri_outside_pair_target_deg": 16.0,
        "--group_ce_mode": cfg["group_mode"],
        "--os_budget_target_reserve": cfg["os_reserve"],
        "--phase1_export_diagnostic_on_block": "true",
        "--source_val_heavy_eval_start_epoch": 10,
        "--source_val_heavy_eval_interval": 10,
        "--source_val_heavy_eval_final_window": 20,
        "--source_val_heavy_eval_final_interval": 2,
        "--eval_sat_scenarios": LEO_WEAK,
    }
    for name, value in replacements.items():
        previous.dual._set_arg(command, name, value)
    return command


def matrix_payload(rows: Sequence[Mapping[str, Any]], run_id: str, wall_hours: float) -> Dict[str, Any]:
    payload = previous.matrix_payload(rows, run_id, wall_hours)
    payload.update(
        {
            "schema": "phase1_dgleo_hiercore8_matrix_v1",
            "mechanism": "hierarchical_class_core_plus_local_support",
            "open_gradient_budget_range": [0.16, 0.28],
            "diagnostic_endpoint_export_on_guard_block": True,
            "claim_boundary": "PHASE1_SOURCE_ONLY_PROXY_DIAGNOSTIC_NO_TRUE_UNKNOWN_CLAIM",
        }
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch eight Phase1 hierarchical-core P0 experiments.")
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
    rows = build_matrix()
    root = Path(args.root)
    python = Path(args.python)
    wisig = Path(args.wisig_pkl or root / "Dataset_WigSig" / "ManySig.pkl")
    teacher = Path(args.teacher_ckpt or root / "runs" / "phase1_adv3_mechanism32_queue_20260701" / "ADV3B02_CORE90_SOFT_E200" / "best_joint_safe_ssdg.pth")
    if args.emit_matrix:
        Path(args.emit_matrix).write_text(json.dumps(matrix_payload(rows, args.run_id, args.wall_hours), indent=2, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        commands = [build_command(row, root=root, python=python, run_id=args.run_id, wisig_pkl=wisig, teacher_ckpt=teacher) for row in rows]
        print(json.dumps({"run_id": args.run_id, "candidate_count": len(rows), "gpu_total_counts": dict(sorted(Counter(int(row["gpu"]) for row in rows).items())), "unique_command_count": len({tuple(command) for command in commands}), "first_command": shlex.join(commands[0]), "last_command": shlex.join(commands[-1])}, indent=2, sort_keys=True))
        return 0
    original_build = previous.dual.build_command
    original_payload = previous.dual.matrix_payload
    try:
        previous.dual.build_command = build_command
        previous.dual.matrix_payload = matrix_payload
        return previous.dual.run_matrix(args, rows)
    finally:
        previous.dual.build_command = original_build
        previous.dual.matrix_payload = original_payload


if __name__ == "__main__":
    raise SystemExit(main())
