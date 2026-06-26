"""Read-only N607 training process inventory for CV-SincNet automation.

The monitor automation uses this script to avoid brittle one-off ``ps | grep``
checks.  It collects a bounded remote process/GPU snapshot, then classifies
CV-SincNet training from process tree, GPU compute PID, project path, and
training CLI evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import time
from typing import Any


DEFAULT_REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
DEFAULT_SSH_CONFIG = pathlib.Path(__file__).resolve().parent / "n607_ssh_config"
DEFAULT_BRIDGE_KEY = "C:/Users/lh594/.ssh/id_ed25519_lab_bridge_172_31_105_18"
DEFAULT_N607_KEY = "C:/Users/lh594/.ssh/id_ed25519_n607"
DEFAULT_BRIDGE_HOST = "administrator@172.31.105.18"
DEFAULT_N607_HOST = "szu2070436088@172.31.111.215"


TRAINING_ENTRY_PATTERNS = [
    "train.py",
    "code/train.py",
    "training_test_eval.py",
    "train_target_adapt.py",
    "train_federated.py",
    "train_cen31_distill.py",
    "train_cvs.py",
    "train_fjmp.py",
    "train_sgc",
    "train_recon",
    "distill_recon",
    "eval_recon_frontend",
]

LAUNCHER_PATTERNS = [
    "launch_",
    "run_",
    "_8gpu",
    "_4gpu",
    "_6gpu",
    "queue",
    "nohup",
]

NOISE_PATTERNS = [
    "grep",
    "sed",
    "awk",
    "nvidia-smi",
    "dbus",
    "at-spi",
    "sshd:",
    "n607_training_inventory.py",
]

FEDERATED_PATTERNS = [
    "--train_mode fed",
    "train_mode=fed",
    "fedavg",
    "fedprox",
    "fedcvs",
    "federated",
    "fl_round",
    "--fl_",
    "vmb",
    "receiver_agnostic",
    "federated_config.json",
]

CENTRALIZED_HINTS = [
    "--train_mode centralized",
    "train_mode=centralized",
    "centralized",
    "cen",
    "cen31",
    "cena31",
    "student",
    "distill",
    "sgc",
    "ssdg",
    "target_adapt",
    "fjmp",
]


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return "" if value is None else str(value)


def _lower_cmd(proc: dict[str, Any]) -> str:
    parts = [
        _as_text(proc.get("cmdline")),
        _as_text(proc.get("exe")),
        _as_text(proc.get("cwd")),
        _as_text(proc.get("environ")),
    ]
    return " ".join(parts).lower()


def _cmd(proc: dict[str, Any]) -> str:
    return _as_text(proc.get("cmdline"))


def _pid(proc: dict[str, Any]) -> str:
    return str(proc.get("pid", ""))


def _ppid(proc: dict[str, Any]) -> str:
    return str(proc.get("ppid", ""))


def is_noise(proc: dict[str, Any]) -> bool:
    cmd = _lower_cmd(proc)
    if not cmd:
        return True
    return any(pattern in cmd for pattern in NOISE_PATTERNS)


def is_project_related(proc: dict[str, Any], remote_root: str) -> bool:
    root = remote_root.lower()
    cmd = _lower_cmd(proc)
    cwd = str(proc.get("cwd") or "").lower()
    return cwd.startswith(root) or root in cmd


def is_pythonish(proc: dict[str, Any]) -> bool:
    cmd = _lower_cmd(proc)
    return any(token in cmd for token in ["python", "torchrun", "accelerate"])


def has_training_entry(proc: dict[str, Any]) -> bool:
    cmd = _lower_cmd(proc)
    if any(pattern in cmd for pattern in TRAINING_ENTRY_PATTERNS):
        return True
    if re.search(r"\b-m\s+[\w.]*?(train|distill|eval)[\w.]*", cmd):
        return True
    if "--run_name" in cmd and any(flag in cmd for flag in ["--output_dir", "--log_dir", "--epochs", "--train_mode"]):
        return True
    if "/logs/" in cmd and "/runs/" in cmd and "cuda_visible_devices" in cmd:
        return True
    return False


def classify_lane(proc: dict[str, Any]) -> str:
    cmd = _lower_cmd(proc)
    if any(pattern in cmd for pattern in FEDERATED_PATTERNS):
        return "federated_vmb"
    if any(pattern in cmd for pattern in CENTRALIZED_HINTS):
        return "centralized"
    if any(pattern in cmd for pattern in TRAINING_ENTRY_PATTERNS):
        return "centralized"
    return "unknown"


def build_children(processes: list[dict[str, Any]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for proc in processes:
        children.setdefault(_ppid(proc), []).append(_pid(proc))
    return children


def descendant_pids(pid: str, children: dict[str, list[str]]) -> set[str]:
    found: set[str] = set()
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        if child in found:
            continue
        found.add(child)
        stack.extend(children.get(child, []))
    return found


def ancestor_pids(pid: str, process_by_pid: dict[str, dict[str, Any]]) -> list[str]:
    ancestors: list[str] = []
    seen: set[str] = set()
    current = process_by_pid.get(pid)
    while current:
        parent = _ppid(current)
        if not parent or parent == "0" or parent in seen:
            break
        seen.add(parent)
        ancestors.append(parent)
        current = process_by_pid.get(parent)
    return ancestors


def gpu_pid_set(snapshot: dict[str, Any]) -> set[str]:
    pids: set[str] = set()
    for row in snapshot.get("gpu_compute", []) or []:
        pid = str(row.get("pid") or "").strip()
        if pid and pid.lower() not in {"none", "[not supported]"}:
            pids.add(pid)
    return pids


def gpu_by_pid(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in snapshot.get("gpu_compute", []) or []:
        pid = str(row.get("pid") or "").strip()
        if pid:
            result.setdefault(pid, []).append(row)
    return result


def classify_snapshot(snapshot: dict[str, Any], remote_root: str = DEFAULT_REMOTE_ROOT) -> dict[str, Any]:
    processes = snapshot.get("processes", []) or []
    process_by_pid = {_pid(proc): proc for proc in processes if _pid(proc)}
    children = build_children(processes)
    gpu_pids = gpu_pid_set(snapshot)
    gpus = gpu_by_pid(snapshot)

    active: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    active_pids: set[str] = set()

    for proc in processes:
        pid = _pid(proc)
        if not pid or is_noise(proc):
            continue
        project_related = is_project_related(proc, remote_root)
        descendants = descendant_pids(pid, children)
        gpu_related = pid in gpu_pids or bool(descendants & gpu_pids)
        entry = has_training_entry(proc)
        launcher = any(pattern in _lower_cmd(proc) for pattern in LAUNCHER_PATTERNS)
        pythonish = is_pythonish(proc)

        reasons: list[str] = []
        if project_related:
            reasons.append("project_path")
        if pid in gpu_pids:
            reasons.append("gpu_compute_pid")
        if descendants & gpu_pids:
            reasons.append("gpu_compute_descendant")
        if entry:
            reasons.append("training_entry_or_run_args")
        if launcher:
            reasons.append("launcher_or_queue")

        is_training = False
        if project_related and entry:
            is_training = True
        elif project_related and gpu_related and pythonish:
            is_training = True
        elif project_related and launcher and descendants & gpu_pids:
            is_training = True
        elif gpu_related and entry:
            is_training = True

        if is_training:
            lane = classify_lane(proc)
            active_pids.add(pid)
            active.append(
                {
                    "pid": pid,
                    "ppid": _ppid(proc),
                    "lane": lane,
                    "confidence": "high" if pid in gpu_pids or entry else "medium",
                    "reasons": reasons,
                    "gpu_compute": gpus.get(pid, []),
                    "cwd": proc.get("cwd"),
                    "exe": proc.get("exe"),
                    "cmdline": _cmd(proc),
                    "ancestor_pids": ancestor_pids(pid, process_by_pid),
                    "descendant_gpu_pids": sorted(descendants & gpu_pids),
                }
            )
        elif project_related and gpu_related:
            excluded.append(
                {
                    "pid": pid,
                    "ppid": _ppid(proc),
                    "reason": "project_gpu_process_without_training_entry",
                    "cwd": proc.get("cwd"),
                    "cmdline": _cmd(proc),
                }
            )

    # Attach project launcher ancestors for context, without double-counting them
    # as lane activity unless they already matched above.
    launcher_context: list[dict[str, Any]] = []
    for proc in processes:
        pid = _pid(proc)
        if not pid or is_noise(proc):
            continue
        descendants = descendant_pids(pid, children)
        if not descendants & active_pids:
            continue
        if not is_project_related(proc, remote_root):
            continue
        if not any(pattern in _lower_cmd(proc) for pattern in LAUNCHER_PATTERNS):
            continue
        launcher_context.append(
            {
                "pid": pid,
                "ppid": _ppid(proc),
                "reason": "launcher_parent_of_active_training",
                "cwd": proc.get("cwd"),
                "cmdline": _cmd(proc),
                "active_child_pids": sorted(descendants & active_pids),
            }
        )

    centralized_active = any(item["lane"] == "centralized" for item in active)
    federated_active = any(item["lane"] == "federated_vmb" for item in active)
    unknown_active = any(item["lane"] == "unknown" for item in active)

    monitor_state = 0 if centralized_active and federated_active else 1
    if unknown_active:
        # Unknown project GPU training blocks optimizer to avoid false idle.
        monitor_state = 0

    return {
        "schema_version": 1,
        "remote_root": remote_root,
        "collected_at": snapshot.get("collected_at"),
        "host": snapshot.get("host"),
        "gpu_compute": snapshot.get("gpu_compute", []),
        "active_training_processes": active,
        "launcher_context": launcher_context,
        "excluded_ambiguous_process": excluded,
        "centralized_active": centralized_active or unknown_active,
        "federated_vmb_active": federated_active or unknown_active,
        "unknown_training_active": unknown_active,
        "monitor_state": monitor_state,
        "completed_detected_lanes": {
            "centralized": not (centralized_active or unknown_active),
            "federated_vmb": not (federated_active or unknown_active),
        },
        "classification_note": (
            "unknown project GPU training counts as active for both lanes to avoid false optimizer entry"
            if unknown_active
            else "lane activity is based on classified project training processes"
        ),
    }


REMOTE_COLLECTOR = r"""
import json
import os
import socket
import subprocess
import time

ROOT = __REMOTE_ROOT__

def read_text(path, binary=False, limit=20000):
    try:
        mode = "rb" if binary else "r"
        with open(path, mode) as f:
            data = f.read(limit)
        if binary:
            return data
        return data
    except Exception:
        return None

def read_link(path):
    try:
        return os.readlink(path)
    except Exception:
        return None

def parse_status_ppid(pid):
    text = read_text(f"/proc/{pid}/status") or ""
    for line in text.splitlines():
        if line.startswith("PPid:"):
            return line.split(":", 1)[1].strip()
    return ""

def split_nulls(data):
    if data is None:
        return []
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    return [part for part in data.split("\0") if part]

def collect_processes():
    rows = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = name
        cmdline = split_nulls(read_text(f"/proc/{pid}/cmdline", binary=True))
        environ = split_nulls(read_text(f"/proc/{pid}/environ", binary=True, limit=8000))
        cwd = read_link(f"/proc/{pid}/cwd")
        exe = read_link(f"/proc/{pid}/exe")
        if not cmdline and not cwd and not exe:
            continue
        rows.append({
            "pid": pid,
            "ppid": parse_status_ppid(pid),
            "cwd": cwd,
            "exe": exe,
            "cmdline": cmdline,
            "environ": [item for item in environ if item.startswith(("CUDA_VISIBLE_DEVICES=", "PYTHONPATH=", "RUN_ID=", "LOG_ROOT=", "RUNS_ROOT="))],
        })
    return rows

def collect_gpu_compute():
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
    except Exception as exc:
        return [{"error": str(exc)}]
    if proc.returncode != 0:
        return [{"error": proc.stderr.strip() or f"nvidia-smi exit {proc.returncode}"}]
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append({
                "pid": parts[0],
                "gpu_uuid": parts[1],
                "process_name": parts[2],
                "used_memory_mib": parts[3],
            })
    return rows

print(json.dumps({
    "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "host": socket.gethostname(),
    "remote_root": ROOT,
    "gpu_compute": collect_gpu_compute(),
    "processes": collect_processes(),
}, ensure_ascii=False))
"""


def remote_command(remote_root: str) -> str:
    collector = REMOTE_COLLECTOR.replace("__REMOTE_ROOT__", repr(remote_root))
    return "if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi; \"$PY\" - <<'PY'\n" + collector + "\nPY"


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "command": command,
        }


def direct_ssh_command(args: argparse.Namespace, command: str) -> list[str]:
    return [
        "ssh",
        "-F",
        str(args.ssh_config),
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        args.target_alias,
        command,
    ]


def bridge_ssh_command(args: argparse.Namespace, command: str) -> list[str]:
    proxy = (
        f"ssh -i {args.bridge_key} -o BatchMode=yes -o IdentitiesOnly=yes "
        f"-o StrictHostKeyChecking=accept-new -W %h:%p {args.bridge_host}"
    )
    return [
        "ssh",
        "-i",
        args.n607_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ProxyCommand={proxy}",
        args.n607_host,
        command,
    ]


def load_json_stdout(result: dict[str, Any]) -> dict[str, Any] | None:
    text = result.get("stdout") or ""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def collect_remote_inventory(args: argparse.Namespace) -> dict[str, Any]:
    probe = remote_command(args.remote_root)
    attempts = []
    for route, builder in [("direct", direct_ssh_command), ("bridge", bridge_ssh_command)]:
        if route == "bridge" and args.no_bridge:
            continue
        result = run_command(builder(args, probe), args.timeout)
        attempts.append(
            {
                "route": route,
                "exit_code": result["exit_code"],
                "timed_out": result["timed_out"],
                "stderr_tail": (result.get("stderr") or "")[-2000:],
            }
        )
        if result["exit_code"] == 0 and not result["timed_out"]:
            snapshot = load_json_stdout(result)
            if snapshot is None:
                attempts[-1]["parse_error"] = "no JSON object found in stdout"
                continue
            classified = classify_snapshot(snapshot, args.remote_root)
            classified["route_used"] = route
            classified["route_attempts"] = attempts
            return classified
        if route == "direct" and args.direct_only:
            break
    return {
        "schema_version": 1,
        "route_used": None,
        "route_attempts": attempts,
        "monitor_state": None,
        "centralized_active": "unknown",
        "federated_vmb_active": "unknown",
        "unknown_training_active": "unknown",
        "completed_detected_lanes": {"centralized": "unknown", "federated_vmb": "unknown"},
        "active_training_processes": [],
        "excluded_ambiguous_process": [],
        "error": "ROUTE_GATE_FAILED",
    }


def load_snapshot(path: str) -> dict[str, Any]:
    if path == "-":
        return json.loads(sys.stdin.read())
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--ssh-config", type=pathlib.Path, default=DEFAULT_SSH_CONFIG)
    parser.add_argument("--target-alias", default="N607")
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--no-bridge", action="store_true")
    parser.add_argument("--bridge-key", default=DEFAULT_BRIDGE_KEY)
    parser.add_argument("--n607-key", default=DEFAULT_N607_KEY)
    parser.add_argument("--bridge-host", default=DEFAULT_BRIDGE_HOST)
    parser.add_argument("--n607-host", default=DEFAULT_N607_HOST)
    parser.add_argument("--from-snapshot", help="Classify a local JSON snapshot instead of SSH collection. Use '-' for stdin.")
    parser.add_argument("--print-remote-command", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_remote_command:
        print(remote_command(args.remote_root))
        return 0
    if args.from_snapshot:
        result = classify_snapshot(load_snapshot(args.from_snapshot), args.remote_root)
    else:
        result = collect_remote_inventory(args)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if not result.get("error") else 2


if __name__ == "__main__":
    raise SystemExit(main())
