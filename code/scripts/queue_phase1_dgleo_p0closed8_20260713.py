from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_ROOT = Path("/home/szu2070436088/2510044040/CV-SincNet")
DEFAULT_PYTHON = Path("/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python")
DEFAULT_RUN_ID = "phase1_dgleo_p0closed8_20260713"
BLOCKER_PATTERNS = (
    "run_cvs_publication_matrix",
    "run_cvs_baseline_queue.sh",
    "baselines.drift.train",
    "launch_phase1_dgleo_",
)


def _compute_processes() -> List[Dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    rows: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        rows.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "used_memory_mib": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1,
            }
        )
    return rows


def _blocker_processes(self_pid: int) -> List[Dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,cmd="],
        text=True,
        capture_output=True,
        check=True,
    )
    rows: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        pid = int(fields[0])
        cmdline = fields[2]
        if pid == int(self_pid) or "queue_phase1_dgleo_p0closed8_20260713.py" in cmdline:
            continue
        matched = [pattern for pattern in BLOCKER_PATTERNS if pattern in cmdline]
        if matched:
            rows.append({"pid": pid, "ppid": int(fields[1]), "patterns": matched, "cmdline": cmdline})
    return rows


def _write_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wait for an empty N607 lane, then launch Phase1 P0 closure.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--max-wait-hours", type=float, default=12.0)
    parser.add_argument("--wall-hours", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    python = Path(args.python).resolve()
    launcher = root / "code" / "scripts" / "launch_phase1_dgleo_p0closed8_20260713.py"
    for required in (root, python, launcher):
        if not required.exists():
            raise FileNotFoundError(required)
    if float(args.poll_seconds) < 5.0 or int(args.stable_polls) < 1:
        raise ValueError("poll-seconds must be >=5 and stable-polls must be >=1")
    if not 0.0 < float(args.max_wait_hours) <= 24.0:
        raise ValueError("max-wait-hours must be in (0,24]")
    if not 0.0 < float(args.wall_hours) <= 10.0:
        raise ValueError("wall-hours must be in (0,10]")

    state_path = root / "logs" / f"{args.run_id}_queue_state.json"
    command = [
        str(python),
        str(launcher),
        "--run-id",
        str(args.run_id),
        "--root",
        str(root),
        "--python",
        str(python),
        "--wall-hours",
        str(float(args.wall_hours)),
    ]
    if args.dry_run:
        print(json.dumps({"state_path": str(state_path), "launch_command": command}, indent=2))
        return 0

    started = time.monotonic()
    deadline = started + float(args.max_wait_hours) * 3600.0
    stable = 0
    while time.monotonic() < deadline:
        compute = _compute_processes()
        blockers = _blocker_processes(os.getpid())
        clear = not compute and not blockers
        stable = stable + 1 if clear else 0
        payload = {
            "schema": "phase1_p0closed8_queue_state_v1",
            "run_id": str(args.run_id),
            "status": "READY_STABILIZING" if clear else "WAITING_FOR_EXISTING_JOBS",
            "stable_polls": int(stable),
            "required_stable_polls": int(args.stable_polls),
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            "compute_processes": compute,
            "blocker_processes": blockers,
            "launch_command": command,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_state(state_path, payload)
        print(
            f"[QUEUE] status={payload['status']} stable={stable}/{int(args.stable_polls)} "
            f"compute={len(compute)} blockers={len(blockers)}",
            flush=True,
        )
        if stable >= int(args.stable_polls):
            payload["status"] = "LAUNCHING"
            _write_state(state_path, payload)
            return subprocess.call(command, cwd=str(root))
        time.sleep(float(args.poll_seconds))

    _write_state(
        state_path,
        {
            "schema": "phase1_p0closed8_queue_state_v1",
            "run_id": str(args.run_id),
            "status": "WAIT_TIMEOUT_NO_LAUNCH",
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return 75


if __name__ == "__main__":
    raise SystemExit(main())
