from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import launch_phase1_dgleo_p1verify128_20260710 as p1base


DEFAULT_RUN_ID = "phase1_dgleo_dualguard16_20260712"
DEFAULT_ROOT = Path("/home/szu2070436088/2510044040/CV-SincNet")
DEFAULT_PYTHON = Path("/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python")
PAIRED_SEEDS = (712101, 712211)
WALL_HOURS = 10.0


BASE: Dict[str, Any] = deepcopy(p1base.BASE)
BASE.update(
    {
        "lr": 0.000043,
        "u_dom": 0.18,
        "u_adv": 0.08,
        "u_sat": 0.30,
        "u_dm": 0.0025,
        "u_q": 0.0030,
        "u_start": 80,
        "u_min": 8,
        "l_rx_inv": 0.10,
        "l_day_inv": 0.07,
        "l_channel_inv": 0.11,
        "u_rx_inv": 0.05,
        "u_day_inv": 0.035,
        "u_channel_inv": 0.07,
        "dm_lambda": 0.0040,
        "dm_quantile": 0.70,
        "dm_source": 0.75,
        "dm_proxy": 0.80,
        "dm_bridge": 0.75,
        "dm_lowden": 0.60,
        "dm_tail": 0.80,
        "dm_overflow": 0.85,
        "dm_ratio": 0.75,
        "dm_satpair": 0.55,
        "dm_p95": 62.0,
        "dm_p99": 80.0,
        "dm_cvar": 72.0,
        "source_local_compact": 0.12,
        "source_local_invariant": 0.10,
        "source_local_inter": 0.08,
        "source_local_accept": 0.10,
        "source_local_density": 0.06,
        "source_w": 0.010,
        "proxy_w": 0.0040,
        "sat_cls": 1.00,
        "sat_cons": 0.10,
        "teacher_clean": 3.00,
        "teacher_sat": 1.70,
        "teacher_zid": 0.58,
        "domain_w": 1.40,
        "adv_w": 0.20,
        "os_min_budget": 0.06,
        "os_max_budget": 0.22,
        "os_min_closed_scale": 1.0,
        "source_sat_weight": 1.0,
        "u_direct_required": True,
    }
)


CELLS: Sequence[Mapping[str, Any]] = (
    {
        "cell": "C0_DG_ANCHOR",
        "role": "historical_dg_recovery_control",
        "overrides": {
            "source_w": 0.0,
            "dm_lambda": 0.0,
            "u_dm": 0.0,
            "source_local_compact": 0.0,
            "source_local_invariant": 0.0,
            "source_local_inter": 0.0,
            "source_local_accept": 0.0,
            "source_local_density": 0.0,
            "u_direct_required": False,
        },
    },
    {
        "cell": "C1_LEAVE_DOMAIN_ONLY",
        "role": "bounded_leave_domain_without_structural_components",
        "overrides": {
            "source_w": 0.008,
            "source_local_compact": 0.0,
            "source_local_invariant": 0.0,
            "source_local_inter": 0.0,
            "source_local_accept": 0.0,
            "source_local_density": 0.0,
        },
    },
    {
        "cell": "C2_LOCAL_NO_DENSITY",
        "role": "local_component_density_ablation",
        "overrides": {"source_local_density": 0.0},
    },
    {
        "cell": "C3_LOCAL_BALANCED",
        "role": "bounded_local_component_stable",
        "overrides": {},
    },
    {
        "cell": "C4_LOCAL_STRONG_PROTECT",
        "role": "bounded_local_component_aggressive_with_dg_protection",
        "overrides": {
            "source_w": 0.016,
            "source_local_compact": 0.20,
            "source_local_invariant": 0.15,
            "source_local_inter": 0.12,
            "source_local_accept": 0.18,
            "source_local_density": 0.10,
            "dm_lambda": 0.006,
            "teacher_clean": 3.25,
            "teacher_sat": 1.85,
            "os_max_budget": 0.18,
        },
    },
    {
        "cell": "C5_SAT_GEOM_STRONG",
        "role": "satellite_view_geometry_focus",
        "overrides": {
            "source_sat_weight": 1.50,
            "l_channel_inv": 0.15,
            "u_channel_inv": 0.10,
            "teacher_sat": 1.90,
            "sat_cons": 0.12,
        },
    },
    {
        "cell": "C6_U_DOMAIN_QUAR",
        "role": "unlabeled_domain_satellite_quarantine_without_u_direct",
        "overrides": {
            "u_dom": 0.24,
            "u_adv": 0.12,
            "u_sat": 0.42,
            "u_dm": 0.0,
            "u_q": 0.0055,
            "u_rx_inv": 0.08,
            "u_day_inv": 0.055,
            "u_channel_inv": 0.11,
            "u_direct_required": False,
        },
    },
    {
        "cell": "C7_FULL_JOINT",
        "role": "full_bounded_geometry_and_unlabeled_direct_joint",
        "overrides": {
            "dm_lambda": 0.006,
            "proxy_w": 0.0050,
            "u_dom": 0.24,
            "u_adv": 0.12,
            "u_sat": 0.42,
            "u_dm": 0.0045,
            "u_q": 0.0050,
            "u_rx_inv": 0.08,
            "u_day_inv": 0.055,
            "u_channel_inv": 0.11,
            "os_max_budget": 0.22,
        },
    },
)


def build_matrix() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gpu, cell in enumerate(CELLS):
        for replicate, seed in enumerate(PAIRED_SEEDS, start=1):
            config = deepcopy(BASE)
            config.update(dict(cell.get("overrides", {})))
            rows.append(
                {
                    "candidate_id": f"DG16_{cell['cell']}_S{replicate}",
                    "cell": str(cell["cell"]),
                    "role": str(cell["role"]),
                    "replicate": replicate,
                    "seed": int(seed),
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
    if len(rows) != 16:
        raise ValueError(f"expected 16 candidates, got {len(rows)}")
    if len({str(row["candidate_id"]) for row in rows}) != 16:
        raise ValueError("candidate IDs are not unique")
    if Counter(int(row["gpu"]) for row in rows) != Counter({gpu: 2 for gpu in range(8)}):
        raise ValueError("matrix must assign exactly two candidates to every GPU")
    if Counter(str(row["cell"]) for row in rows) != Counter({str(cell["cell"]): 2 for cell in CELLS}):
        raise ValueError("matrix must use two paired seeds per mechanism cell")
    for row in rows:
        if not bool(row["source_only"]) or row["checkpoint_selection"] != "final_only":
            raise ValueError(f"protocol violation: {row['candidate_id']}")


def _set_arg(command: List[str], name: str, value: Any) -> None:
    if name in command:
        index = command.index(name)
        if index + 1 >= len(command) or str(command[index + 1]).startswith("--"):
            raise ValueError(f"cannot replace flag-style argument: {name}")
        command[index + 1] = str(value)
    else:
        command.extend([name, str(value)])


def build_command(
    row: Mapping[str, Any],
    *,
    root: Path,
    python: Path,
    run_id: str,
    wisig_pkl: Path,
    teacher_ckpt: Path,
) -> List[str]:
    command = p1base.build_command(
        row,
        root=root,
        python=python,
        run_id=run_id,
        wisig_pkl=wisig_pkl,
        teacher_ckpt=teacher_ckpt,
    )
    config = dict(row["config"])
    replacements = {
        "--epochs": 120,
        "--label_epochs": 75,
        "--pseudo_epochs": 45,
        "--source_val_heavy_eval_start_epoch": 20,
        "--source_val_heavy_eval_interval": 20,
        "--source_val_heavy_eval_final_window": 20,
        "--source_val_heavy_eval_final_interval": 5,
        "--tail_safety_p95_target_deg": 65,
        "--tail_safety_p99_target_deg": 82,
        "--tail_safety_cvar_target_deg": 75,
        "--tail_safety_proxy_vaccept_target": 0.60,
        "--tail_safety_warning_patience": 2,
        "--tail_safety_rollback_patience": 1,
        "--tail_safety_absolute_violation_drives_state": "false",
        "--tail_safety_training_stop_enabled": "false",
        "--tail_safety_reference_requires_absolute_safe": "false",
        "--os_eff_min_budget": config["os_min_budget"],
        "--os_eff_max_budget": config["os_max_budget"],
        "--os_budget_min_closed_scale": config["os_min_closed_scale"],
        "--max_grad_norm": 5.0,
        "--source_val_dg_health_guard": "true",
        "--source_val_dg_health_start_epoch": 20,
        "--source_val_dg_health_warning_drop_pp": 3.0,
        "--source_val_dg_health_stop_drop_pp": 10.0,
        "--source_val_dg_health_floor": 70.0,
        "--source_val_dg_health_min_open_scale": 0.15,
        "--source_val_dg_health_stop_patience": 1,
        "--source_episode_start_epoch": 30,
        "--source_episode_warmup_epochs": 35,
        "--source_episode_structural_start_epoch": 40,
        "--source_episode_structural_warmup_epochs": 40,
        "--source_episode_local_min_samples": 2,
        "--source_episode_local_radius_floor_deg": 4.0,
        "--source_episode_local_density_beta": 0.25,
        "--source_episode_local_density_cap": 1.5,
        "--source_episode_local_term_cap": 3.0,
        "--source_episode_clean_weight": 1.0,
        "--source_episode_sat_weight": config["source_sat_weight"],
        "--source_episode_multiview_normalize": "true",
        "--direct_metric_start_epoch": 50,
        "--direct_metric_warmup_epochs": 35,
        "--direct_metric_zid_p50_target_deg": 35,
        "--direct_metric_zid_p95_target_deg": config["dm_p95"],
        "--direct_metric_zid_p99_target_deg": config["dm_p99"],
        "--direct_metric_zid_tail_cvar_target_deg": config["dm_cvar"],
        "--direct_metric_source_overflow_target": 0.70,
        "--direct_metric_proxy_vaccept_target": 0.45,
        "--direct_metric_bridge_accept_target": 0.35,
        "--direct_metric_low_density_accept_target": 0.12,
        "--direct_metric_tail_accept_target": 0.35,
        "--direct_metric_overflow_accept_target": 0.20,
        "--direct_metric_radius_inter_ratio_target": 0.85,
        "--direct_metric_core_accept_target": 0.70,
        "--proxy_unknown_start_epoch": 60,
        "--proxy_unknown_warmup_epochs": 30,
        "--u_direct_metric_start_epoch": config["u_start"],
        "--u_quarantine_start_epoch": 76,
        "--u_direct_metric_min_selected": config["u_min"],
        "--u_direct_idle_blocks_promotion": "true" if config["u_direct_required"] else "false",
        "--u_tri_state_required": "true",
        "--u_tri_max_outside_rate": 0.90,
        "--feasibility_gate": "true",
        "--feasibility_stage": "audit",
        "--feasibility_relaxed_pass": "false",
        "--feasibility_local_pass": "false",
        "--sat_view_schedule": (
            "1@0.45:leo_clear_weak;31@0.72:leo_clear_weak,leo_low_elev_weak,leo_rain_weak;"
            "76@0.88:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
        ),
    }
    for name, value in replacements.items():
        _set_arg(command, name, value)
    return command


def matrix_payload(rows: Sequence[Mapping[str, Any]], run_id: str, wall_hours: float) -> Dict[str, Any]:
    return {
        "schema": "phase1_dgleo_dualguard16_matrix_v1",
        "run_id": run_id,
        "candidate_count": len(rows),
        "cell_count": len(CELLS),
        "paired_seeds": list(PAIRED_SEEDS),
        "gpu_total_counts": dict(sorted(Counter(int(row["gpu"]) for row in rows).items())),
        "max_active_per_gpu": 2,
        "wall_clock_limit_hours": float(wall_hours),
        "checkpoint_selection": "final_only",
        "satellite_train_eval_protocol": "disjoint_family_and_implementation",
        "claim_boundary": "PHASE1_SOURCE_ONLY_PROXY_DIAGNOSTIC_NO_TRUE_UNKNOWN_SUCCESS_CLAIM",
        "candidates": list(rows),
    }


def _terminal_status(out_dir: Path, returncode: int) -> str:
    path = out_dir / "phase1_terminal_status.json"
    if path.is_file():
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("status") or "UNKNOWN")
        except Exception:
            pass
    return "PROCESS_COMPLETE" if returncode == 0 else "PROCESS_FAILED_NO_TERMINAL"


def _scheduler_outcome(
    terminal_results: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    timed_out: bool,
) -> tuple[Dict[str, Any], int]:
    status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in terminal_results)
    returncode_counts = Counter(int(row.get("returncode", -1)) for row in terminal_results)
    non_complete = [
        str(row.get("candidate_id") or "UNKNOWN")
        for row in terminal_results
        if str(row.get("status") or "UNKNOWN") != "COMPLETE" or int(row.get("returncode", -1)) != 0
    ]
    missing_count = max(0, int(expected_count) - len(terminal_results))
    if timed_out:
        status = "WALL_CLOCK_TIMEOUT"
        exit_code = 124
    elif missing_count:
        status = "SCHEDULER_INCOMPLETE"
        exit_code = 1
    elif non_complete:
        status = "CHILD_FAILURE"
        exit_code = 1
    else:
        status = "COMPLETE"
        exit_code = 0
    return (
        {
            "status": status,
            "terminal_count": len(terminal_results),
            "missing_terminal_count": missing_count,
            "candidate_status_counts": dict(sorted(status_counts.items())),
            "child_returncode_counts": {
                str(code): count for code, count in sorted(returncode_counts.items())
            },
            "non_complete_candidates": non_complete,
        },
        exit_code,
    )


def _terminate_process_groups(active: Mapping[int, Mapping[str, Any]], grace_seconds: float = 60.0) -> None:
    for pid in list(active):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(1.0, float(grace_seconds))
    while time.monotonic() < deadline and any(state["process"].poll() is None for state in active.values()):
        time.sleep(2.0)
    for pid, state in active.items():
        if state["process"].poll() is None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_matrix(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> int:
    root = Path(args.root)
    python = Path(args.python)
    wisig = Path(args.wisig_pkl or root / "Dataset_WigSig" / "ManySig.pkl")
    p1base.validate_source_wisig_pkl(wisig)
    teacher = Path(
        args.teacher_ckpt
        or root
        / "runs"
        / "phase1_adv3_mechanism32_queue_20260701"
        / "ADV3B02_CORE90_SOFT_E200"
        / "best_joint_safe_ssdg.pth"
    )
    for required in (python, wisig, teacher, root / "code" / "SSDG" / "train_ssdg.py"):
        if not required.is_file():
            raise FileNotFoundError(required)
    observed = p1base._pmon_pids()
    occupied = {gpu: sorted(pids) for gpu, pids in observed.items() if pids}
    if occupied:
        raise RuntimeError(f"dualguard16 requires empty GPUs before launch, found: {occupied}")

    log_root = root / "logs" / args.run_id
    run_root = root / "runs" / args.run_id
    log_root.mkdir(parents=True, exist_ok=False)
    run_root.mkdir(parents=True, exist_ok=False)
    (log_root / "candidate_matrix.json").write_text(
        json.dumps(matrix_payload(rows, args.run_id, args.wall_hours), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    active: Dict[int, Dict[str, Any]] = {}
    events_path = log_root / "scheduler_events.tsv"
    start_monotonic = time.monotonic()
    deadline = start_monotonic + float(args.wall_hours) * 3600.0
    timed_out = False
    terminal_results: List[Dict[str, Any]] = []
    with events_path.open("w", encoding="utf-8", newline="") as events:
        writer = csv.writer(events, delimiter="\t")
        writer.writerow(["timestamp", "event", "candidate_id", "gpu", "seed", "pid", "returncode", "status", "log"])
        for row in rows:
            gpu = int(row["gpu"])
            out_dir = run_root / str(row["candidate_id"])
            log_path = log_root / f"{row['candidate_id']}.out"
            out_dir.mkdir()
            command = build_command(
                row,
                root=root,
                python=python,
                run_id=args.run_id,
                wisig_pkl=wisig,
                teacher_ckpt=teacher,
            )
            handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = f"{root / 'code'}:{root}:{env.get('PYTHONPATH', '')}"
            process = subprocess.Popen(
                command,
                cwd=str(root),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[process.pid] = {
                "process": process,
                "row": row,
                "handle": handle,
                "out_dir": out_dir,
                "log_path": log_path,
            }
            writer.writerow(
                [time.strftime("%Y-%m-%dT%H:%M:%S%z"), "LAUNCHED", row["candidate_id"], gpu, row["seed"], process.pid, "", "RUNNING", log_path]
            )
            events.flush()
            time.sleep(float(args.launch_settle_seconds))

        while active:
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_groups(active)
            for pid, state in list(active.items()):
                code = state["process"].poll()
                if code is None:
                    continue
                state["handle"].close()
                status = "WALL_CLOCK_TIMEOUT" if timed_out else _terminal_status(state["out_dir"], int(code))
                row = state["row"]
                writer.writerow(
                    [time.strftime("%Y-%m-%dT%H:%M:%S%z"), "TERMINAL", row["candidate_id"], row["gpu"], row["seed"], pid, code, status, state["log_path"]]
                )
                terminal_results.append(
                    {
                        "candidate_id": str(row["candidate_id"]),
                        "returncode": int(code),
                        "status": str(status),
                    }
                )
                events.flush()
                del active[pid]
            if active and not timed_out:
                time.sleep(float(args.poll_seconds))

    outcome, scheduler_exit_code = _scheduler_outcome(
        terminal_results,
        expected_count=len(rows),
        timed_out=timed_out,
    )
    summary = {
        "run_id": args.run_id,
        "wall_hours_limit": float(args.wall_hours),
        "elapsed_hours": (time.monotonic() - start_monotonic) / 3600.0,
        "candidate_count": len(rows),
        **outcome,
    }
    (log_root / "scheduler_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return scheduler_exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a 16-row bounded Phase1 dual-objective validation matrix.")
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
        raise ValueError(f"--wall-hours must be in (0, {WALL_HOURS}]")
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
            or root
            / "runs"
            / "phase1_adv3_mechanism32_queue_20260701"
            / "ADV3B02_CORE90_SOFT_E200"
            / "best_joint_safe_ssdg.pth"
        )
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
                    "cell_count": len(CELLS),
                    "gpu_total_counts": dict(sorted(Counter(int(row["gpu"]) for row in rows).items())),
                    "wall_clock_limit_hours": float(args.wall_hours),
                    "unique_command_count": len({tuple(command) for command in commands}),
                    "first_command": shlex.join(commands[0]),
                    "last_command": shlex.join(commands[-1]),
                    "claim": "PHASE1_SOURCE_ONLY_PROXY_DIAGNOSTIC_NO_TRUE_UNKNOWN_SUCCESS_CLAIM",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return run_matrix(args, rows)


if __name__ == "__main__":
    raise SystemExit(main())
