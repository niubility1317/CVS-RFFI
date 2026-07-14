from __future__ import annotations

import argparse
import itertools
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

import launch_phase1_dgleo_corepath8_20260714 as previous


dual = previous.dual
DEFAULT_RUN_ID = "phase1_dgleo_p0factorial8_20260714"
DEFAULT_ROOT = previous.DEFAULT_ROOT
DEFAULT_PYTHON = previous.DEFAULT_PYTHON
WALL_HOURS = previous.WALL_HOURS
SEED = previous.SEED
LEO_WEAK = previous.LEO_WEAK
GPU_COUNT = previous.GPU_COUNT
MAX_TOTAL_COMPUTE_PER_GPU = previous.MAX_TOTAL_COMPUTE_PER_GPU
MIN_FREE_MEMORY_MIB = previous.MIN_FREE_MEMORY_MIB
RESOURCE_WAIT_TIMEOUT_SECONDS = previous.RESOURCE_WAIT_TIMEOUT_SECONDS
RESOURCE_POLL_SECONDS = previous.RESOURCE_POLL_SECONDS
GPU_PERMUTATION_SEED = 20260714
GPU_PERMUTATION = (1, 0, 2, 6, 5, 4, 7, 3)


BASE: Dict[str, Any] = deepcopy(previous.BASE)
BASE.update(
    {
        "id_feature": "feat_cls",
        "hierarchy_combine": "smooth_min",
        "reference_bank": False,
        "core_tpr_weight": 0.0,
        "known_coverage_weight": 0.0,
        "positive_first": False,
        "hierarchical_class_gate": False,
        "virtual_detach": True,
        "effective_negative_grad_guard": False,
        "l_rx_inv": 0.0,
        "l_day_inv": 0.0,
        "l_channel_inv": 0.0,
        "u_rx_inv": 0.0,
        "u_day_inv": 0.0,
        "u_channel_inv": 0.0,
        "source_local_compact": 0.0,
        "source_local_invariant": 0.0,
        "source_local_inter": 0.0,
        "source_local_overlap": 0.0,
        "source_local_accept": 0.0,
        "source_local_density": 0.0,
        "source_sat_weight": 1.0,
        "sat_pair_weight": 0.0,
        "group_mode": "smooth_dro_capped",
        "query_inter_weight": 0.0,
        "query_overlap_weight": 0.0,
        "budget_scope": "all_shared",
        "os_min_budget": 0.0,
        "os_max_budget": 0.0,
        "os_budget_controller": False,
        "risk_buffer": False,
        "risk_energy_weight": 0.0,
    }
)


FACTOR_A: Mapping[str, Any] = {
    "reference_bank": True,
    "core_tpr_weight": 4.0,
    "known_coverage_weight": 1.0,
    "positive_first": True,
    "hierarchical_class_gate": True,
    "virtual_detach": False,
    "effective_negative_grad_guard": True,
}
FACTOR_B: Mapping[str, Any] = {
    "l_rx_inv": 0.18,
    "l_day_inv": 0.12,
    "l_channel_inv": 0.22,
    "u_rx_inv": 0.10,
    "u_day_inv": 0.07,
    "u_channel_inv": 0.14,
    "source_local_compact": 0.24,
    "source_local_invariant": 0.30,
    "source_local_inter": 0.16,
    "source_local_overlap": 0.24,
    "source_local_accept": 0.18,
    "source_local_density": 0.10,
    "source_sat_weight": 1.30,
    "sat_pair_weight": 1.25,
    "group_mode": "dual_worst",
}
FACTOR_C: Mapping[str, Any] = {
    "query_inter_weight": 0.90,
    "query_overlap_weight": 0.90,
    "budget_scope": "zid_path",
    "os_min_budget": 0.16,
    "os_max_budget": 0.24,
    "os_budget_controller": True,
    "risk_buffer": True,
    "risk_energy_weight": 0.02,
}


def _set_boolean_flag(command: List[str], positive: str, negative: str, enabled: bool) -> None:
    command[:] = [token for token in command if token not in (positive, negative)]
    command.append(positive if enabled else negative)


def build_matrix() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    factor_levels = itertools.product((False, True), repeat=3)
    for index, (factor_a, factor_b, factor_c) in enumerate(factor_levels):
        config = deepcopy(BASE)
        for enabled, profile in (
            (factor_a, FACTOR_A),
            (factor_b, FACTOR_B),
            (factor_c, FACTOR_C),
        ):
            if enabled:
                config.update(profile)
        factor_code = f"A{int(factor_a)}B{int(factor_b)}C{int(factor_c)}"
        rows.append(
            {
                "candidate_id": f"P0F_{factor_code}",
                "cell": factor_code,
                "role": "three_package_full_factorial",
                "factors": {"A": factor_a, "B": factor_b, "C": factor_c},
                "replicate": 1,
                "seed": SEED,
                "gpu": GPU_PERMUTATION[index],
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
        raise ValueError("P0 factorial matrix must contain eight unique candidates")
    if Counter(int(row["gpu"]) for row in rows) != Counter({gpu: 1 for gpu in range(GPU_COUNT)}):
        raise ValueError("P0 factorial matrix must assign exactly one candidate process per GPU")
    factor_cells = {
        tuple(bool(row["factors"][name]) for name in ("A", "B", "C"))
        for row in rows
    }
    if factor_cells != set(itertools.product((False, True), repeat=3)):
        raise ValueError("P0 factorial matrix must cover every A/B/C level exactly once")
    for row in rows:
        if not bool(row["source_only"]) or row["checkpoint_selection"] != "final_only":
            raise ValueError(f"common protocol violation: {row['candidate_id']}")
        cfg = row["config"]
        factor_a = bool(row["factors"]["A"])
        factor_c = bool(row["factors"]["C"])
        if bool(cfg["reference_bank"]) != factor_a:
            raise ValueError(f"factor A reference-bank mismatch: {row['candidate_id']}")
        if bool(cfg["effective_negative_grad_guard"]) != factor_a:
            raise ValueError(f"factor A effective-gradient mismatch: {row['candidate_id']}")
        if factor_c and not 0.0 < float(cfg["os_min_budget"]) <= float(cfg["os_max_budget"]):
            raise ValueError(f"factor C gradient budget invalid: {row['candidate_id']}")
        if not factor_c and (float(cfg["os_min_budget"]) != 0.0 or float(cfg["os_max_budget"]) != 0.0):
            raise ValueError(f"factor C-off gradient budget must be disabled: {row['candidate_id']}")


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
    replacements = {
        "--epochs": 120,
        "--label_epochs": 0,
        "--pseudo_epochs": 120,
        "--checkpoint_selection": "final_only",
        "--phase1_distribution_audit_only": "false",
        "--phase1_export_diagnostic_on_block": "true",
        "--concat_sat_deduplicate_tx_ce": "true",
        "--concat_sat_teacher_clean_only": "true",
        "--sat_train_scenario": "leo_clear_weak",
        "--sat_train_scenarios": LEO_WEAK,
        "--eval_sat_scenarios": LEO_WEAK,
        "--sat_protocol_disjoint_required": "false",
        "--source_val_heavy_eval_start_epoch": 10,
        "--source_val_heavy_eval_interval": 10,
        "--source_val_heavy_eval_final_window": 20,
        "--source_val_heavy_eval_final_interval": 2,
        "--u_geometry_all_valid_queries": "true",
        "--u_tri_state_required": "true",
        "--u_direct_include_outside_known": "false",
        "--u_outside_stop_gradient": "true",
        "--direct_metric_hierarchical_class_gate": str(bool(cfg["hierarchical_class_gate"])).lower(),
        "--direct_metric_hierarchical_combine": "smooth_min",
        "--direct_metric_reference_bank": str(bool(cfg["reference_bank"])).lower(),
        "--direct_metric_gate_reference_detach": "true",
        "--direct_metric_reference_refresh_epochs": 10,
        "--direct_metric_reference_per_component": 4,
        "--direct_metric_core_tpr_weight": cfg["core_tpr_weight"],
        "--direct_metric_known_coverage_weight": cfg["known_coverage_weight"],
        "--direct_metric_positive_first": str(bool(cfg["positive_first"])).lower(),
        "--direct_metric_virtual_detach": str(bool(cfg["virtual_detach"])).lower(),
        "--direct_metric_require_effective_negative_grad": str(
            bool(cfg["effective_negative_grad_guard"])
        ).lower(),
        "--lambda_zid_receiver_invariance": cfg["l_rx_inv"],
        "--lambda_zid_day_invariance": cfg["l_day_inv"],
        "--lambda_zid_channel_invariance": cfg["l_channel_inv"],
        "--lambda_u_zid_receiver_invariance": cfg["u_rx_inv"],
        "--lambda_u_zid_day_invariance": cfg["u_day_inv"],
        "--lambda_u_zid_channel_invariance": cfg["u_channel_inv"],
        "--zid_invariance_min_groups": 2,
        "--zid_invariance_min_samples_per_group": 2,
        "--zid_channel_pair_weight": 1.25,
        "--source_episode_local_compact_weight": cfg["source_local_compact"],
        "--source_episode_local_invariant_weight": cfg["source_local_invariant"],
        "--source_episode_local_inter_weight": cfg["source_local_inter"],
        "--source_episode_local_inter_margin_deg": 48.0,
        "--source_episode_local_overlap_weight": cfg["source_local_overlap"],
        "--source_episode_local_overlap_margin_deg": 5.0,
        "--source_episode_local_accept_weight": cfg["source_local_accept"],
        "--source_episode_local_density_weight": cfg["source_local_density"],
        "--source_episode_local_min_samples": 3,
        "--source_episode_sat_weight": cfg["source_sat_weight"],
        "--direct_metric_sat_pair_weight": cfg["sat_pair_weight"],
        "--group_ce_mode": cfg["group_mode"],
        "--direct_metric_component_inter_margin_weight": cfg["query_inter_weight"],
        "--direct_metric_component_inter_margin_deg": 48.0,
        "--direct_metric_component_overlap_weight": cfg["query_overlap_weight"],
        "--direct_metric_component_overlap_margin_deg": 5.0,
        "--os_budget_scope": cfg["budget_scope"],
        "--os_eff_min_budget": cfg["os_min_budget"],
        "--os_eff_max_budget": cfg["os_max_budget"],
        "--os_budget_controller": str(bool(cfg["os_budget_controller"])).lower(),
        "--os_gradient_surgery": str(bool(cfg["os_budget_controller"])).lower(),
        "--os_objective_budget_controller": str(bool(cfg["os_budget_controller"])).lower(),
        "--os_gradient_protect_closed": str(bool(cfg["os_budget_controller"])).lower(),
        "--unlabeled_risk_buffer": str(bool(cfg["risk_buffer"])).lower(),
        "--risk_maxprob_min": 0.70,
        "--risk_density_percentile": 10.0,
        "--risk_geo_margin_min_deg": 2.0,
        "--lambda_risk_energy_out": cfg["risk_energy_weight"],
    }
    for name, value in replacements.items():
        dual._set_arg(command, name, value)
    _set_boolean_flag(
        command,
        "--use_concat_sat_channel_aug",
        "--no_use_concat_sat_channel_aug",
        True,
    )
    _set_boolean_flag(command, "--concat_sat_ce_only", "--no_concat_sat_ce_only", False)
    return command


def matrix_payload(rows: Sequence[Mapping[str, Any]], run_id: str, wall_hours: float) -> Dict[str, Any]:
    payload = previous.matrix_payload(rows, run_id, wall_hours)
    payload.update(
        {
            "schema": "phase1_dgleo_p0factorial8_matrix_v1",
            "design": "2x2x2_full_factorial_A_positive_core_B_tx_conditioned_invariance_C_query_risk",
            "factor_order": ["A", "B", "C"],
            "gpu_permutation_seed": GPU_PERMUTATION_SEED,
            "gpu_permutation": list(GPU_PERMUTATION),
            "planned_processes_per_gpu": 1,
            "one_candidate_per_gpu": True,
            "epochs": 120,
            "checkpoint_selection": "final_only",
            "source_val_re_evaluation": {
                "epochs_1_to_100_interval": 10,
                "final_20_epochs_interval": 2,
            },
            "satellite_test_protocol": "leo_weak_final_only",
            "unlabeled_outside_policy": "stop_gradient",
            "claim_boundary": "PHASE1_SOURCE_ONLY_INTERNAL_P0_DIAGNOSTIC_NO_DEPLOYMENT_SUCCESS_CLAIM",
        }
    )
    return payload


wait_for_gpu_slots = previous.wait_for_gpu_slots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Phase1 A/B/C 2^3 P0 factorial, one process per GPU.")
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
        or root
        / "runs"
        / "phase1_adv3_mechanism32_queue_20260701"
        / "ADV3B02_CORE90_SOFT_E200"
        / "best_joint_safe_ssdg.pth"
    )
    if args.emit_matrix:
        Path(args.emit_matrix).write_text(
            json.dumps(matrix_payload(rows, args.run_id, args.wall_hours), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.dry_run:
        commands = [
            build_command(
                row,
                root=root,
                python=python,
                run_id=args.run_id,
                wisig_pkl=wisig,
                teacher_ckpt=teacher,
            )
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
                "planned_processes_per_gpu": 1,
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
