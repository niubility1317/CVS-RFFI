from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import launch_phase1_dgleo_hiercore8_20260713 as previous


dual = previous.dual
DEFAULT_RUN_ID = "phase1_dgleo_corepath8_20260714"
DEFAULT_ROOT = previous.DEFAULT_ROOT
DEFAULT_PYTHON = previous.DEFAULT_PYTHON
WALL_HOURS = 10.0
SEED = previous.SEED
LEO_WEAK = previous.LEO_WEAK
GPU_COUNT = 8
MAX_TOTAL_COMPUTE_PER_GPU = 2
MIN_FREE_MEMORY_MIB = 10000
RESOURCE_WAIT_TIMEOUT_SECONDS = 3 * 3600
RESOURCE_POLL_SECONDS = 60.0


BASE: Dict[str, Any] = deepcopy(previous.BASE)
BASE.update(
    {
        "id_feature": "feat_joint",
        "hierarchy_combine": "product",
        "reference_bank": False,
        "core_tpr_weight": 0.0,
        "core_tpr_target": 0.85,
        "source_radius_cap": 0.0,
        "source_leave_target": 42.0,
        "temporal_mode": "batch_neighbor",
        "dedup_tx_ce": False,
        "teacher_clean_only": False,
        "budget_scope": "all_shared",
        "strict_contract": False,
        "objective_max_scale": 32.0,
    }
)


CELLS: Sequence[Mapping[str, Any]] = (
    {
        "cell": "R0_R3_REPLAY",
        "role": "same_seed_legacy_mechanism_control_with_tail_fix",
        "overrides": {},
    },
    {
        "cell": "R1_ID_CORE",
        "role": "ungated_identity_core_only",
        "overrides": {"id_feature": "feat_cls"},
    },
    {
        "cell": "R2_FROZEN_GATE",
        "role": "identity_core_frozen_reference_smooth_gate_known_tpr",
        "overrides": {
            "id_feature": "feat_cls",
            "hierarchy_combine": "smooth_min",
            "reference_bank": True,
            "core_tpr_weight": 4.0,
            "strict_contract": True,
        },
    },
    {
        "cell": "R3_OVERFLOW_ALIGNED",
        "role": "fixed_radius_overflow_and_leave_domain_alignment",
        "overrides": {
            "id_feature": "feat_cls",
            "hierarchy_combine": "smooth_min",
            "reference_bank": True,
            "core_tpr_weight": 4.0,
            "source_radius_cap": 18.0,
            "source_leave_target": 16.0,
            "strict_contract": True,
        },
    },
    {
        "cell": "R4_U_EPOCH_BANK",
        "role": "cross_epoch_unlabeled_temporal_bank",
        "overrides": {
            "id_feature": "feat_cls",
            "hierarchy_combine": "smooth_min",
            "reference_bank": True,
            "core_tpr_weight": 4.0,
            "source_radius_cap": 18.0,
            "source_leave_target": 16.0,
            "temporal_mode": "epoch_bank",
            "strict_contract": True,
        },
    },
    {
        "cell": "R5_CONCAT_DEDUP",
        "role": "concat_sa_tx_ce_dedup_and_clean_teacher",
        "overrides": {
            "id_feature": "feat_cls",
            "hierarchy_combine": "smooth_min",
            "reference_bank": True,
            "core_tpr_weight": 4.0,
            "source_radius_cap": 18.0,
            "source_leave_target": 16.0,
            "temporal_mode": "epoch_bank",
            "dedup_tx_ce": True,
            "teacher_clean_only": True,
            "strict_contract": True,
        },
    },
    {
        "cell": "R6_FULL_STABLE",
        "role": "full_p0_closed_loop_stable",
        "overrides": {
            "id_feature": "feat_cls",
            "hierarchy_combine": "smooth_min",
            "reference_bank": True,
            "core_tpr_weight": 5.0,
            "source_radius_cap": 18.0,
            "source_leave_target": 16.0,
            "temporal_mode": "epoch_bank",
            "dedup_tx_ce": True,
            "teacher_clean_only": True,
            "budget_scope": "zid_path",
            "strict_contract": True,
            "objective_max_scale": 8.0,
            "os_min_budget": 0.18,
            "os_max_budget": 0.24,
            "objective_boundary": 0.30,
            "objective_source": 0.30,
            "objective_invariant": 0.25,
            "objective_u": 0.15,
        },
    },
    {
        "cell": "R7_FULL_AGGRESSIVE",
        "role": "full_p0_closed_loop_aggressive",
        "overrides": {
            "id_feature": "feat_cls",
            "hierarchy_combine": "smooth_min",
            "reference_bank": True,
            "core_tpr_weight": 7.0,
            "source_radius_cap": 16.0,
            "source_leave_target": 14.0,
            "temporal_mode": "epoch_bank",
            "dedup_tx_ce": True,
            "teacher_clean_only": True,
            "budget_scope": "zid_path",
            "strict_contract": True,
            "objective_max_scale": 8.0,
            "source_w": 0.14,
            "dm_lambda": 0.07,
            "os_min_budget": 0.20,
            "os_max_budget": 0.28,
            "objective_boundary": 0.32,
            "objective_source": 0.33,
            "objective_invariant": 0.20,
            "objective_u": 0.15,
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
                "candidate_id": f"CP_{cell['cell']}",
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
        raise ValueError("corepath matrix must assign exactly one candidate per GPU")
    for row in rows:
        cfg = row["config"]
        shares = [float(cfg[name]) for name in ("objective_boundary", "objective_source", "objective_invariant", "objective_u")]
        if abs(sum(shares) - 1.0) > 1e-8 or min(shares) <= 0.0:
            raise ValueError(f"invalid objective shares: {row['candidate_id']}")
        if not 0.16 <= float(cfg["os_min_budget"]) <= float(cfg["os_max_budget"]) <= 0.28:
            raise ValueError(f"invalid open gradient budget: {row['candidate_id']}")
        if bool(cfg["reference_bank"]) and str(cfg["hierarchy_combine"]) != "smooth_min":
            raise ValueError(f"frozen reference candidates require smooth_min: {row['candidate_id']}")


def build_command(
    row: Mapping[str, Any],
    *,
    root: Path,
    python: Path,
    run_id: str,
    wisig_pkl: Path,
    teacher_ckpt: Path,
) -> List[str]:
    command = previous.build_command(
        row,
        root=root,
        python=python,
        run_id=run_id,
        wisig_pkl=wisig_pkl,
        teacher_ckpt=teacher_ckpt,
    )
    cfg = dict(row["config"])
    strict = bool(cfg["strict_contract"])
    replacements = {
        "--id_feature_key": cfg["id_feature"],
        "--direct_metric_hierarchical_combine": cfg["hierarchy_combine"],
        "--direct_metric_gate_reference_detach": "true" if cfg["reference_bank"] else "false",
        "--direct_metric_reference_bank": str(bool(cfg["reference_bank"])).lower(),
        "--direct_metric_reference_refresh_epochs": 10,
        "--direct_metric_reference_per_component": 4,
        "--direct_metric_core_tpr_target": cfg["core_tpr_target"],
        "--direct_metric_core_tpr_weight": cfg["core_tpr_weight"],
        "--direct_metric_source_radius_cap_deg": cfg["source_radius_cap"],
        "--source_episode_radius_cap_deg": cfg["source_radius_cap"] if cfg["source_radius_cap"] > 0 else 30.0,
        "--source_episode_leave_domain_target_deg": cfg["source_leave_target"],
        "--pseudo_temporal_mode": cfg["temporal_mode"],
        "--pseudo_temporal_bank_min_streak": 2,
        "--concat_sat_deduplicate_tx_ce": str(bool(cfg["dedup_tx_ce"])).lower(),
        "--concat_sat_teacher_clean_only": str(bool(cfg["teacher_clean_only"])).lower(),
        "--os_budget_scope": cfg["budget_scope"],
        "--os_objective_max_scale": cfg["objective_max_scale"],
        "--os_eff_min_budget": cfg["os_min_budget"],
        "--os_eff_max_budget": cfg["os_max_budget"],
        "--os_objective_boundary_share": cfg["objective_boundary"],
        "--os_objective_source_share": cfg["objective_source"],
        "--os_objective_invariant_share": cfg["objective_invariant"],
        "--os_objective_u_share": cfg["objective_u"],
        "--direct_metric_zid_p50_target_deg": 28.0,
        "--direct_metric_zid_p95_target_deg": 54.0 if strict else 58.0,
        "--direct_metric_zid_p99_target_deg": 70.0 if strict else 80.0,
        "--direct_metric_zid_tail_cvar_target_deg": 56.0 if strict else 70.0,
        "--direct_metric_source_overflow_target": 0.45 if strict else 0.65,
        "--direct_metric_proxy_vaccept_target": 0.35 if strict else 0.50,
        "--tail_safety_p95_target_deg": 54.0 if strict else 60.0,
        "--tail_safety_p99_target_deg": 70.0 if strict else 82.0,
        "--tail_safety_cvar_target_deg": 56.0 if strict else 72.0,
        "--tail_safety_proxy_vaccept_target": 0.35 if strict else 0.50,
        "--tail_safety_p99_expansion_block_final_delta": 2.0,
        "--tail_safety_p99_expansion_block_best_delta": 3.5,
        "--eval_sat_scenarios": LEO_WEAK,
    }
    for name, value in replacements.items():
        dual._set_arg(command, name, value)
    return command


def matrix_payload(rows: Sequence[Mapping[str, Any]], run_id: str, wall_hours: float) -> Dict[str, Any]:
    payload = previous.matrix_payload(rows, run_id, wall_hours)
    payload.update(
        {
            "schema": "phase1_dgleo_corepath8_matrix_v1",
            "mechanism": "invariant_identity_core_plus_frozen_reference_acceptance",
            "same_seed_as_hiercore8_r3": True,
            "one_candidate_per_gpu": True,
            "resource_gate": {
                "gpu_count": GPU_COUNT,
                "max_total_compute_per_gpu": MAX_TOTAL_COMPUTE_PER_GPU,
                "min_free_memory_mib": MIN_FREE_MEMORY_MIB,
                "required_free_slots_per_gpu": 1,
                "counts_all_nvidia_compute_clients": True,
                "blocks_same_run_id": True,
                "allows_one_unrelated_compute_client": True,
            },
            "final_checkpoint_only": True,
            "claim_boundary": "PHASE1_SOURCE_ONLY_PROXY_DIAGNOSTIC_NO_TRUE_UNKNOWN_CLAIM",
        }
    )
    return payload


def wait_for_gpu_slots(
    *,
    run_id: str,
    max_total_compute_per_gpu: int,
    min_free_memory_mib: int,
    allow_unrelated_compute: bool,
    timeout_seconds: float,
    poll_seconds: float,
) -> Dict[str, Any]:
    if max_total_compute_per_gpu < 1:
        raise ValueError("max_total_compute_per_gpu must be >= 1")
    if timeout_seconds < 0 or poll_seconds <= 0:
        raise ValueError("invalid GPU resource wait timing")
    deadline = time.monotonic() + timeout_seconds
    while True:
        snapshot = dual.gpu_launch_snapshot(
            run_id=run_id,
            gpus=list(range(GPU_COUNT)),
            max_total_compute_per_gpu=max_total_compute_per_gpu,
            min_free_memory_mib=min_free_memory_mib,
            allow_unrelated_compute=allow_unrelated_compute,
        )
        blocked = snapshot["blocked"]
        print(
            json.dumps(
                {
                    "event": "GPU_SLOT_CHECK",
                    "snapshot": snapshot,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not blocked:
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU_SLOT_WAIT_TIMEOUT blocked={blocked}")
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch eight Phase1 invariant-core P0 experiments.")
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
    parser.add_argument("--max-total-compute-per-gpu", type=int, default=MAX_TOTAL_COMPUTE_PER_GPU)
    parser.add_argument("--min-free-memory-mib", type=int, default=MIN_FREE_MEMORY_MIB)
    parser.add_argument(
        "--allow-unrelated-compute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resource-wait-timeout-seconds", type=float, default=RESOURCE_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--resource-poll-seconds", type=float, default=RESOURCE_POLL_SECONDS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_matrix()
    root = Path(args.root)
    python = Path(args.python)
    wisig = Path(args.wisig_pkl or root / "Dataset_WigSig" / "ManySig.pkl")
    teacher = Path(
        args.teacher_ckpt
        or root / "runs" / "phase1_adv3_mechanism32_queue_20260701" / "ADV3B02_CORE90_SOFT_E200" / "best_joint_safe_ssdg.pth"
    )
    if args.emit_matrix:
        Path(args.emit_matrix).write_text(
            json.dumps(matrix_payload(rows, args.run_id, args.wall_hours), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.dry_run:
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
    launch_occupancy = wait_for_gpu_slots(
        run_id=str(args.run_id),
        max_total_compute_per_gpu=int(args.max_total_compute_per_gpu),
        min_free_memory_mib=int(args.min_free_memory_mib),
        allow_unrelated_compute=bool(args.allow_unrelated_compute),
        timeout_seconds=float(args.resource_wait_timeout_seconds),
        poll_seconds=float(args.resource_poll_seconds),
    )
    print(
        json.dumps(
            {
                "event": "GPU_SLOT_READY",
                "gpu_snapshot_before_phase1_launch": launch_occupancy,
                "phase1_candidates": len(rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
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
