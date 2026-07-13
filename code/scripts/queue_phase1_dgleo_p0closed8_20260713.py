from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import launch_phase1_dgleo_p0closed8_20260713 as launcher


DEFAULT_ROOT = Path("/home/szu2070436088/2510044040/CV-SincNet")
DEFAULT_PYTHON = Path("/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python")
DEFAULT_RUN_ID = "phase1_dgleo_p0closed8_20260713"


def _run_text(command: Sequence[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}")
    return completed.stdout


def _gpu_uuid_to_index() -> Dict[str, int]:
    output = _run_text(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    mapping: Dict[str, int] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[parts[1]] = int(parts[0])
    if not mapping:
        raise RuntimeError("nvidia-smi returned no GPU index/UUID mapping")
    return mapping


def _compute_processes() -> List[Dict[str, Any]]:
    uuid_to_index = _gpu_uuid_to_index()
    output = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: List[Dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        rows.append(
            {
                "gpu": uuid_to_index.get(parts[0]),
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "used_memory_mib": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1,
            }
        )
    return rows


def _gpu_occupancy(
    compute: Sequence[Mapping[str, Any]], own_pids: Sequence[int], gpu_count: int = 8
) -> Dict[int, Dict[str, Any]]:
    owned = {int(pid) for pid in own_pids}
    occupancy: Dict[int, Dict[str, Any]] = {
        gpu: {"total_count": 0, "own_count": 0, "external_count": 0, "own_pids": [], "external_pids": []}
        for gpu in range(gpu_count)
    }
    for row in compute:
        gpu = row.get("gpu")
        if gpu is None or int(gpu) not in occupancy:
            continue
        gpu = int(gpu)
        pid = int(row["pid"])
        occupancy[gpu]["total_count"] += 1
        if pid in owned:
            occupancy[gpu]["own_count"] += 1
            occupancy[gpu]["own_pids"].append(pid)
        else:
            occupancy[gpu]["external_count"] += 1
            occupancy[gpu]["external_pids"].append(pid)
    return occupancy


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _candidate_state(
    rows: Sequence[Mapping[str, Any]],
    pending: Mapping[str, Mapping[str, Any]],
    active: Mapping[int, Mapping[str, Any]],
    terminal: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    active_by_id = {str(state["row"]["candidate_id"]): (pid, state) for pid, state in active.items()}
    result: List[Dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        item: Dict[str, Any] = {"candidate_id": candidate_id, "gpu": int(row["gpu"])}
        if candidate_id in terminal:
            item.update(terminal[candidate_id])
        elif candidate_id in active_by_id:
            pid, state = active_by_id[candidate_id]
            item.update(
                {
                    "status": "RUNNING",
                    "pid": int(pid),
                    "launched_at": state["launched_at"],
                    "elapsed_hours": (time.monotonic() - float(state["launched_monotonic"])) / 3600.0,
                }
            )
        elif candidate_id in pending:
            item["status"] = "PENDING_GPU_SLOT"
        else:
            item["status"] = "UNTRACKED_ERROR"
        result.append(item)
    return result


def _terminate_owned_process(pid: int, state: Mapping[str, Any]) -> None:
    launcher.dual._terminate_process_groups({int(pid): state}, grace_seconds=60.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch one fixed Phase1 candidate per GPU when that GPU has a free training slot."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--wisig-pkl", default="")
    parser.add_argument("--teacher-ckpt", default="")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument("--max-wait-hours", type=float, default=12.0)
    parser.add_argument("--wall-hours", type=float, default=10.0)
    parser.add_argument("--max-concurrent-per-gpu", type=int, default=2)
    parser.add_argument("--launch-settle-seconds", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if float(args.poll_seconds) < 5.0 or int(args.stable_polls) < 1:
        raise ValueError("poll-seconds must be >=5 and stable-polls must be >=1")
    if not 0.0 < float(args.max_wait_hours) <= 24.0:
        raise ValueError("max-wait-hours must be in (0,24]")
    if not 0.0 < float(args.wall_hours) <= 10.0:
        raise ValueError("wall-hours must be in (0,10]")
    if not 1 <= int(args.max_concurrent_per_gpu) <= 2:
        raise ValueError("max-concurrent-per-gpu must be 1 or 2")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    root = Path(args.root).resolve()
    python = Path(args.python).resolve()
    wisig = Path(args.wisig_pkl or root / "Dataset_WigSig" / "ManySig.pkl")
    teacher = Path(
        args.teacher_ckpt
        or root
        / "runs"
        / "phase1_adv3_mechanism32_queue_20260701"
        / "ADV3B02_CORE90_SOFT_E200"
        / "best_joint_safe_ssdg.pth"
    )
    rows = launcher.build_matrix()
    commands = {
        str(row["candidate_id"]): launcher.build_command(
            row,
            root=root,
            python=python,
            run_id=str(args.run_id),
            wisig_pkl=wisig,
            teacher_ckpt=teacher,
        )
        for row in rows
    }
    state_path = root / "logs" / f"{args.run_id}_queue_state.json"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "phase1_p0closed8_capacity_queue_v2",
                    "run_id": str(args.run_id),
                    "state_path": str(state_path),
                    "launcher": str(Path(launcher.__file__).resolve()),
                    "candidate_count": len(rows),
                    "candidate_gpu_map": {str(row["candidate_id"]): int(row["gpu"]) for row in rows},
                    "unique_command_count": len({tuple(command) for command in commands.values()}),
                    "max_concurrent_per_gpu": int(args.max_concurrent_per_gpu),
                    "launch_mode": "fixed_candidate_per_gpu_capacity_aware",
                    "foreign_processes_count_as_candidates": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for required in (python, wisig, teacher, root / "code" / "SSDG" / "train_ssdg.py"):
        if not required.is_file():
            raise FileNotFoundError(required)
    launcher.dual.p1base.validate_source_wisig_pkl(wisig)

    log_root = root / "logs" / str(args.run_id)
    run_root = root / "runs" / str(args.run_id)
    log_root.mkdir(parents=True, exist_ok=False)
    run_root.mkdir(parents=True, exist_ok=False)
    (log_root / "candidate_matrix.json").write_text(
        json.dumps(launcher.matrix_payload(rows, str(args.run_id), float(args.wall_hours)), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    pending: MutableMapping[str, Mapping[str, Any]] = {
        str(row["candidate_id"]): row for row in rows
    }
    active: MutableMapping[int, Dict[str, Any]] = {}
    terminal: MutableMapping[str, Dict[str, Any]] = {}
    stable_by_gpu = {gpu: 0 for gpu in range(8)}
    wait_started = time.monotonic()
    wait_deadline = wait_started + float(args.max_wait_hours) * 3600.0
    events_path = log_root / "scheduler_events.tsv"

    with events_path.open("w", encoding="utf-8", newline="") as events:
        writer = csv.writer(events, delimiter="\t")
        writer.writerow(
            ["timestamp", "event", "candidate_id", "gpu", "seed", "pid", "returncode", "status", "log"]
        )
        while pending or active:
            now = time.monotonic()
            for pid, state in list(active.items()):
                process = state["process"]
                timed_out = now >= float(state["deadline_monotonic"])
                if timed_out and process.poll() is None:
                    _terminate_owned_process(pid, state)
                code = process.poll()
                if code is None:
                    continue
                state["handle"].close()
                candidate_id = str(state["row"]["candidate_id"])
                status = (
                    "WALL_CLOCK_TIMEOUT"
                    if timed_out
                    else launcher.dual._terminal_status(state["out_dir"], int(code))
                )
                terminal[candidate_id] = {
                    "status": status,
                    "pid": int(pid),
                    "returncode": int(code),
                    "launched_at": state["launched_at"],
                    "terminal_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "TERMINAL",
                        candidate_id,
                        state["row"]["gpu"],
                        state["row"]["seed"],
                        pid,
                        code,
                        status,
                        state["log_path"],
                    ]
                )
                events.flush()
                del active[pid]

            compute = _compute_processes()
            occupancy = _gpu_occupancy(compute, list(active), gpu_count=8)
            for candidate_id, row in list(pending.items()):
                gpu = int(row["gpu"])
                has_capacity = occupancy[gpu]["total_count"] < int(args.max_concurrent_per_gpu)
                stable_by_gpu[gpu] = stable_by_gpu[gpu] + 1 if has_capacity else 0
                if stable_by_gpu[gpu] < int(args.stable_polls):
                    continue

                out_dir = run_root / candidate_id
                log_path = log_root / f"{candidate_id}.out"
                out_dir.mkdir()
                handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTHONPATH"] = f"{root / 'code'}:{root}:{env.get('PYTHONPATH', '')}"
                launched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                process = subprocess.Popen(
                    commands[candidate_id],
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
                    "launched_at": launched_at,
                    "launched_monotonic": time.monotonic(),
                    "deadline_monotonic": time.monotonic() + float(args.wall_hours) * 3600.0,
                }
                del pending[candidate_id]
                writer.writerow(
                    [
                        launched_at,
                        "LAUNCHED_SLOT_AWARE",
                        candidate_id,
                        gpu,
                        row["seed"],
                        process.pid,
                        "",
                        "RUNNING",
                        log_path,
                    ]
                )
                events.flush()
                print(
                    f"[QUEUE] launched candidate={candidate_id} gpu={gpu} pid={process.pid} "
                    f"prelaunch_total={occupancy[gpu]['total_count']} prelaunch_external={occupancy[gpu]['external_count']}",
                    flush=True,
                )
                time.sleep(float(args.launch_settle_seconds))

            if pending and time.monotonic() >= wait_deadline:
                for candidate_id, row in list(pending.items()):
                    terminal[candidate_id] = {
                        "status": "WAIT_TIMEOUT_NO_GPU_SLOT",
                        "returncode": 75,
                        "terminal_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    writer.writerow(
                        [
                            time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "WAIT_TIMEOUT",
                            candidate_id,
                            row["gpu"],
                            row["seed"],
                            "",
                            75,
                            "WAIT_TIMEOUT_NO_GPU_SLOT",
                            "",
                        ]
                    )
                    del pending[candidate_id]
                events.flush()

            own_pids = list(active)
            compute = _compute_processes()
            occupancy = _gpu_occupancy(compute, own_pids, gpu_count=8)
            candidates = _candidate_state(rows, pending, active, terminal)
            if pending and active:
                status = "PARTIAL_RUNNING_WAITING_GPU_SLOTS"
            elif pending:
                status = "WAITING_GPU_SLOTS"
            elif active:
                status = "ALL_LAUNCHED_RUNNING"
            elif all(item["status"] == "COMPLETE" for item in candidates):
                status = "COMPLETE"
            else:
                status = "TERMINAL_WITH_FAILURES"
            payload = {
                "schema": "phase1_p0closed8_capacity_queue_v2",
                "run_id": str(args.run_id),
                "status": status,
                "candidate_count": len(rows),
                "own_launched_or_terminal_count": len(rows) - len(pending),
                "own_running_count": len(active),
                "own_terminal_count": len(terminal),
                "pending_count": len(pending),
                "max_concurrent_per_gpu": int(args.max_concurrent_per_gpu),
                "foreign_processes_count_as_candidates": False,
                "gpu_occupancy": occupancy,
                "candidates": candidates,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            _write_state(state_path, payload)
            print(
                f"[QUEUE] status={status} own_running={len(active)} pending={len(pending)} "
                f"terminal={len(terminal)} external_compute={sum(x['external_count'] for x in occupancy.values())}",
                flush=True,
            )
            if pending or active:
                time.sleep(float(args.poll_seconds))

    terminal_results = [
        {
            "candidate_id": candidate_id,
            "returncode": int(item.get("returncode", 1)),
            "status": str(item["status"]),
        }
        for candidate_id, item in terminal.items()
    ]
    outcome, exit_code = launcher.dual._scheduler_outcome(
        terminal_results, expected_count=len(rows), timed_out=False
    )
    summary = {
        "run_id": str(args.run_id),
        "scheduler": "capacity_aware_fixed_candidate_per_gpu_v2",
        "max_concurrent_per_gpu": int(args.max_concurrent_per_gpu),
        "foreign_processes_count_as_candidates": False,
        "candidate_count": len(rows),
        **outcome,
    }
    (log_root / "scheduler_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
