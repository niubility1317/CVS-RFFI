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


DEFAULT_RUN_ID = "phase1_dgleo_jointp0_leoweak8r2_20260713"
DEFAULT_ROOT = dual.DEFAULT_ROOT
DEFAULT_PYTHON = dual.DEFAULT_PYTHON
WALL_HOURS = 10.0
SEED = 713101
LEO_WEAK = "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
_DUAL_BUILD_COMMAND = dual.build_command


BASE: Dict[str, Any] = deepcopy(dual.BASE)
BASE.update(
    {
        "lr": 0.000040,
        "source_w": 0.050,
        "dm_lambda": 0.025,
        "proxy_w": 0.015,
        "u_dm": 0.012,
        "u_q": 0.015,
        "u_start": 1,
        "u_min": 4,
        "source_local_compact": 0.22,
        "source_local_invariant": 0.28,
        "source_local_inter": 0.12,
        "source_local_accept": 0.18,
        "source_local_density": 0.08,
        "l_rx_inv": 0.18,
        "l_day_inv": 0.12,
        "l_channel_inv": 0.22,
        "u_rx_inv": 0.10,
        "u_day_inv": 0.07,
        "u_channel_inv": 0.14,
        "u_dom": 0.24,
        "u_adv": 0.12,
        "u_sat": 0.40,
        "sat_cls": 1.00,
        "sat_cons": 0.12,
        "teacher_clean": 2.20,
        "teacher_sat": 1.35,
        "teacher_zid": 0.35,
        "domain_w": 1.40,
        "adv_w": 0.22,
        "os_min_budget": 0.10,
        "os_max_budget": 0.18,
        "os_min_closed_scale": 0.90,
        "source_sat_weight": 1.0,
        "proto_w": 0.080,
        "proto_domain": 0.90,
        "proto_push": 0.15,
        "open_world_w": 0.040,
        "zid_compact_w": 0.025,
        "pair_weight": 0.90,
    }
)


CELLS: Sequence[Mapping[str, Any]] = (
    {"cell": "J0_STABLE_10_18", "role": "full_joint_stable_budget", "overrides": {}},
    {
        "cell": "J1_BALANCED_12_20",
        "role": "full_joint_balanced_budget",
        "overrides": {"os_min_budget": 0.12, "os_max_budget": 0.20, "source_w": 0.060, "dm_lambda": 0.030},
    },
    {
        "cell": "J2_CORE_MEMORY",
        "role": "full_joint_persistent_core_alignment",
        "overrides": {"os_min_budget": 0.14, "os_max_budget": 0.22, "source_w": 0.080, "proto_w": 0.120, "proto_domain": 1.20},
    },
    {
        "cell": "J3_FIXED_RISK_PRESSURE",
        "role": "full_joint_acceptance_pressure",
        "overrides": {"os_min_budget": 0.14, "os_max_budget": 0.22, "dm_lambda": 0.050, "proxy_w": 0.030, "open_world_w": 0.060},
    },
    {
        "cell": "J4_SAT_INVARIANT",
        "role": "full_joint_clean_sat_identity_alignment",
        "overrides": {"os_min_budget": 0.12, "os_max_budget": 0.20, "l_channel_inv": 0.30, "u_channel_inv": 0.20, "pair_weight": 1.30, "source_sat_weight": 1.25},
    },
    {
        "cell": "J5_U_TRI_STRONG",
        "role": "full_joint_unlabeled_tristate",
        "overrides": {"os_min_budget": 0.12, "os_max_budget": 0.22, "u_dm": 0.025, "u_q": 0.025, "u_dom": 0.30, "u_sat": 0.50},
    },
    {
        "cell": "J6_AGGRESSIVE_18_28",
        "role": "full_joint_aggressive_open_budget",
        "overrides": {"os_min_budget": 0.18, "os_max_budget": 0.28, "source_w": 0.100, "dm_lambda": 0.060, "proxy_w": 0.040, "proto_w": 0.140, "open_world_w": 0.080},
    },
    {
        "cell": "J7_GUARDED_FULL",
        "role": "full_joint_high_pressure_with_dg_guard",
        "overrides": {"os_min_budget": 0.15, "os_max_budget": 0.24, "source_w": 0.080, "dm_lambda": 0.050, "proxy_w": 0.030, "proto_w": 0.120, "teacher_clean": 2.50, "teacher_sat": 1.50, "os_min_closed_scale": 0.95},
    },
)


def build_matrix() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gpu, cell in enumerate(CELLS):
        config = deepcopy(BASE)
        config.update(dict(cell.get("overrides", {})))
        rows.append(
            {
                "candidate_id": f"JP0_{cell['cell']}",
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
        raise ValueError("jointp0 matrix must contain eight unique candidates")
    if Counter(int(row["gpu"]) for row in rows) != Counter({gpu: 1 for gpu in range(8)}):
        raise ValueError("jointp0 matrix must assign exactly one candidate to every GPU")
    for row in rows:
        cfg = row["config"]
        if row["checkpoint_selection"] != "final_only" or not bool(row["source_only"]):
            raise ValueError(f"protocol violation: {row['candidate_id']}")
        if min(float(cfg[key]) for key in ("source_w", "dm_lambda", "proxy_w")) <= 0.0:
            raise ValueError(f"all open losses must be active: {row['candidate_id']}")
        if float(cfg["os_min_budget"]) < 0.10 or float(cfg["os_max_budget"]) > 0.28:
            raise ValueError(f"open gradient budget outside protected exploration range: {row['candidate_id']}")


def build_command(
    row: Mapping[str, Any],
    *,
    root: Path,
    python: Path,
    run_id: str,
    wisig_pkl: Path,
    teacher_ckpt: Path,
) -> List[str]:
    command = _DUAL_BUILD_COMMAND(
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
        "--use_tx_rx_balanced_sampler": "true",
        "--balanced_sampler_tx_per_batch": 6,
        "--balanced_sampler_domain_per_batch": 6,
        "--balanced_sampler_samples_per_cell": 3,
        "--balanced_sampler_replacement": "true",
        "--phase1_distribution_audit_only": "false",
        "--use_proto_memory": "true",
        "--lambda_proto": cfg["proto_w"],
        "--proto_momentum": 0.90,
        "--proto_domain_align_weight": cfg["proto_domain"],
        "--proto_push_weight": cfg["proto_push"],
        "--proto_min_count": 3,
        "--lambda_open_world_feat": cfg["open_world_w"],
        "--ow_feat_start_epoch": 1,
        "--ow_feat_warmup_epochs": 4,
        "--lambda_zid_compact": cfg["zid_compact_w"],
        "--zid_compact_start_epoch": 1,
        "--zid_compact_warmup_epochs": 8,
        "--source_episode_start_epoch": 1,
        "--source_episode_warmup_epochs": 4,
        "--source_episode_structural_start_epoch": 1,
        "--source_episode_structural_warmup_epochs": 8,
        "--source_episode_local_min_samples": 3,
        "--source_episode_local_radius_floor_deg": 4.0,
        "--source_episode_clean_weight": 1.0,
        "--source_episode_sat_weight": cfg["source_sat_weight"],
        "--source_episode_multiview_normalize": "true",
        "--direct_metric_start_epoch": 1,
        "--direct_metric_warmup_epochs": 4,
        "--direct_metric_sat_pair_weight": cfg["pair_weight"],
        "--direct_metric_zid_p50_target_deg": 30,
        "--direct_metric_zid_p95_target_deg": 58,
        "--direct_metric_zid_p99_target_deg": 80,
        "--direct_metric_zid_tail_cvar_target_deg": 70,
        "--direct_metric_source_overflow_target": 0.65,
        "--direct_metric_proxy_vaccept_target": 0.42,
        "--direct_metric_bridge_accept_target": 0.30,
        "--direct_metric_low_density_accept_target": 0.10,
        "--direct_metric_tail_accept_target": 0.32,
        "--direct_metric_overflow_accept_target": 0.18,
        "--direct_metric_radius_inter_ratio_target": 0.82,
        "--proxy_unknown_start_epoch": 1,
        "--proxy_unknown_warmup_epochs": 4,
        "--u_direct_metric_start_epoch": 1,
        "--u_quarantine_start_epoch": 1,
        "--u_direct_metric_min_selected": cfg["u_min"],
        "--u_direct_idle_blocks_promotion": "true",
        "--u_tri_state_required": "true",
        "--u_tri_max_outside_rate": 0.95,
        "--os_eff_min_budget": cfg["os_min_budget"],
        "--os_eff_max_budget": cfg["os_max_budget"],
        "--os_budget_min_closed_scale": cfg["os_min_closed_scale"],
        "--os_budget_max_scale": 8.0,
        "--phase1_v2_os_eff_all_phases": "true",
        "--max_grad_norm": 5.0,
        "--source_val_dg_health_guard": "true",
        "--source_val_dg_health_start_epoch": 5,
        "--source_val_dg_health_warning_drop_pp": 2.0,
        "--source_val_dg_health_stop_drop_pp": 6.0,
        "--source_val_dg_health_floor": 80.0,
        "--source_val_dg_health_min_open_scale": 0.35,
        "--source_val_dg_health_stop_patience": 2,
        "--source_val_heavy_eval_start_epoch": 10,
        "--source_val_heavy_eval_interval": 10,
        "--source_val_heavy_eval_final_window": 20,
        "--source_val_heavy_eval_final_interval": 2,
        "--tail_safety_p95_target_deg": 60,
        "--tail_safety_p99_target_deg": 82,
        "--tail_safety_cvar_target_deg": 72,
        "--tail_safety_proxy_vaccept_target": 0.50,
        "--sat_train_scenario": "leo_clear_weak",
        "--sat_train_scenarios": LEO_WEAK,
        "--eval_sat_scenarios": LEO_WEAK,
        "--sat_protocol_disjoint_required": "false",
        "--sat_cons_start_epoch": 1,
        "--sat_view_schedule": (
            "1@0.55:leo_clear_weak;21@0.75:leo_clear_weak,leo_low_elev_weak,leo_rain_weak;"
            "61@0.90:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
        ),
    }
    for name, value in replacements.items():
        dual._set_arg(command, name, value)
    return command


def matrix_payload(rows: Sequence[Mapping[str, Any]], run_id: str, wall_hours: float) -> Dict[str, Any]:
    return {
        "schema": "phase1_dgleo_jointp0_leoweak8_matrix_v1",
        "run_id": run_id,
        "candidate_count": len(rows),
        "gpu_total_counts": dict(sorted(Counter(int(row["gpu"]) for row in rows).items())),
        "max_active_per_gpu": 1,
        "wall_clock_limit_hours": float(wall_hours),
        "checkpoint_selection": "final_only",
        "satellite_train_eval_protocol": "leo_weak_same_family_independent_random_stress",
        "open_loss_start_epoch": 1,
        "open_gradient_budget_range": [0.10, 0.28],
        "claim_boundary": "PHASE1_SOURCE_ONLY_PROXY_DIAGNOSTIC_NO_TRUE_UNKNOWN_SUCCESS_CLAIM",
        "candidates": list(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch eight joint P0 Phase1 experiments, one per GPU.")
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
