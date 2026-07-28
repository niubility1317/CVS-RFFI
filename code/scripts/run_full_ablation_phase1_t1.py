#!/usr/bin/env python3
"""Run the frozen Phase1 T1 matrix with at most two processes per GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cvsrffi.full_ablation_spec import (
    DESIGN_ID,
    GPU_COUNT,
    PHASE1_T1_ARMS,
    SLOTS_PER_GPU,
    validate_plan_rows,
)


class Phase1RunnerError(RuntimeError):
    """Raised when the immutable Phase1 release contract is violated."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_plan_hash(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {
            key: value
            for key, value in dict(plan).items()
            if key != "sealed_content_sha256"
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_phase1_release_plan(
    plan: Mapping[str, Any],
    *,
    require_launch_authority: bool,
) -> None:
    if plan.get("schema") != "cvs.full_ablation.plan.v1":
        raise Phase1RunnerError("unexpected plan schema")
    if plan.get("design_id") != DESIGN_ID or plan.get("phase") != "phase1":
        raise Phase1RunnerError("plan is not the Phase1 full-ablation T1 matrix")
    rows = list(plan.get("rows") or [])
    validate_plan_rows(rows)
    expected_ids = {arm.ablation_id for arm in PHASE1_T1_ARMS}
    if {row.get("ablation_id") for row in rows} != expected_ids:
        raise Phase1RunnerError("Phase1 plan arm set drift")
    if len(rows) != 30:
        raise Phase1RunnerError("Phase1 T1 release must contain exactly 30 rows")
    if len({int(row["train_seed"]) for row in rows}) != 5:
        raise Phase1RunnerError("Phase1 T1 release must contain five paired seeds")
    if any(row.get("git_commit") != plan.get("git_commit") for row in rows):
        raise Phase1RunnerError("row Git commit differs from plan Git commit")
    if require_launch_authority:
        if plan.get("formal_launch_authority") is not True:
            raise Phase1RunnerError("plan lacks formal launch authority")
        if not str(plan.get("run_id", "")).strip():
            raise Phase1RunnerError("sealed plan lacks run_id")
        sealed_hash = str(plan.get("sealed_content_sha256", "")).lower()
        if (
            len(sealed_hash) != 64
            or sealed_hash != _canonical_plan_hash(plan)
        ):
            raise Phase1RunnerError("sealed plan content hash is missing or invalid")
        seed_registry_hash = str(
            plan.get("seed_registry_sha256", "")
        ).lower()
        if len(seed_registry_hash) != 64 or any(
            char not in "0123456789abcdef"
            for char in seed_registry_hash
        ):
            raise Phase1RunnerError("sealed plan lacks seed-registry hash")
        if any(
            row.get("executor_status") != "LOCAL_VERIFIED"
            for row in rows
        ):
            raise Phase1RunnerError("one or more Phase1 executors are not LOCAL_VERIFIED")
        review = plan.get("independent_review") or {}
        if review.get("p0_count") != 0 or review.get("p1_count") != 0:
            raise Phase1RunnerError("independent review is not P0=0,P1=0")


def build_phase1_command(
    row: Mapping[str, Any],
    *,
    run_id: str,
    python_executable: str,
    train_script: Path,
    wisig_pkl: Path,
    output_dir: Path,
    sealed_plan_sha256: str = "",
    seed_registry_sha256: str = "",
) -> list[str]:
    command = [
        str(python_executable),
        "-u",
        str(train_script),
        "--wisig_pkl",
        str(wisig_pkl),
        "--output_dir",
        str(output_dir),
        "--run_id",
        str(run_id),
        "--candidate_id",
        str(row["ablation_id"]),
        "--formal_ablation",
        "true",
        "--ablation_id",
        str(row["ablation_id"]),
        "--git_commit",
        str(row["git_commit"]),
        "--row_key",
        str(row["row_key"]),
        "--sealed_plan_sha256",
        str(sealed_plan_sha256),
        "--seed_registry_sha256",
        str(seed_registry_sha256),
        "--seed",
        str(int(row["train_seed"])),
        "--device",
        "cuda:0",
    ]
    if str(row.get("config_hash", "")).strip():
        command.extend(
            ["--expected_config_hash", str(row["config_hash"])]
        )
    return command


def normalize_exception_fingerprint(log_text: str) -> str:
    lines = [line.strip() for line in str(log_text).splitlines() if line.strip()]
    exception_lines = [
        line
        for line in lines
        if "error" in line.lower()
        or "exception" in line.lower()
        or "traceback" in line.lower()
    ]
    selected = "\n".join((exception_lines or lines)[-12:])
    selected = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", selected)
    selected = re.sub(r"[A-Za-z]:[\\/][^\s:]+", "<PATH>", selected)
    selected = re.sub(r"/[^\s:]+", "<PATH>", selected)
    selected = re.sub(r"\b\d+\b", "<N>", selected)
    selected = re.sub(r"\s+", " ", selected).strip().lower()
    return hashlib.sha256(selected.encode("utf-8")).hexdigest()


def validate_phase1_row_completion(
    *,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    output_dir: Path,
    return_code: int,
) -> dict[str, Any]:
    terminal_path = output_dir / "phase1_terminal_status.json"
    receipt_path = output_dir / "phase1_training_completion_receipt.json"
    if not terminal_path.is_file() or not receipt_path.is_file():
        raise Phase1RunnerError("row lacks terminal or completion receipt")
    terminal = _load_json(terminal_path)
    receipt = _load_json(receipt_path)
    expected = {
        "run_id": str(plan["run_id"]),
        "row_key": str(row["row_key"]),
        "ablation_id": str(row["ablation_id"]),
        "git_commit": str(plan["git_commit"]),
        "sealed_plan_sha256": str(plan["sealed_content_sha256"]),
        "seed_registry_sha256": str(plan["seed_registry_sha256"]),
    }
    for key, value in expected.items():
        if str(receipt.get(key, "")) != value:
            raise Phase1RunnerError(
                f"row completion receipt identity drift: {key}"
            )
    if int(receipt.get("train_seed", -1)) != int(row["train_seed"]):
        raise Phase1RunnerError("row completion receipt train-seed drift")
    if (
        str(receipt.get("resolved_config_hash", ""))
        != str(row.get("config_hash", ""))
        or str(receipt.get("method_config_hash", ""))
        != str(row.get("method_config_hash", ""))
    ):
        raise Phase1RunnerError("row completion receipt config-hash drift")
    actual_terminal_hash = hashlib.sha256(
        terminal_path.read_bytes()
    ).hexdigest()
    if (
        str(receipt.get("terminal_manifest_sha256", ""))
        != actual_terminal_hash
    ):
        raise Phase1RunnerError("row terminal-manifest hash drift")
    resource_path = output_dir / "phase1_resource_summary.json"
    if (
        not resource_path.is_file()
        or str(receipt.get("resource_summary_sha256", ""))
        != hashlib.sha256(resource_path.read_bytes()).hexdigest()
    ):
        raise Phase1RunnerError("row resource-summary hash drift")
    for key, expected_hash in dict(
        receipt.get("prototype_hashes") or {}
    ).items():
        prototype_path = str(
            (receipt.get("prototype_paths") or {}).get(key, "")
        )
        if (
            not prototype_path
            or not Path(prototype_path).is_file()
            or hashlib.sha256(Path(prototype_path).read_bytes()).hexdigest()
            != str(expected_hash)
        ):
            raise Phase1RunnerError("row prototype artifact hash drift")
    split_receipt = dict(receipt.get("source_split_receipt") or {})
    if (
        len(str(split_receipt.get("split_manifest_sha256", ""))) != 64
        or int(
            split_receipt.get(
                "source_target_receiver_overlap_count",
                -1,
            )
        )
        != 0
    ):
        raise Phase1RunnerError("row completion receipt split evidence invalid")
    if (
        int(return_code) != 0
        or int(receipt.get("exit_code", -1)) != 0
        or str(receipt.get("terminal_status", "")) != "COMPLETE"
        or str(terminal.get("status", "")) != "COMPLETE"
    ):
        raise Phase1RunnerError("row terminal status is not COMPLETE")
    return receipt


def verify_release_checkout(
    plan: Mapping[str, Any],
    repo_root: Path,
) -> None:
    root = repo_root.resolve()
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    expected_commit = str(plan.get("git_commit", "")).strip().lower()
    if actual_commit != expected_commit:
        raise Phase1RunnerError(
            f"checkout commit drift: expected={expected_commit} actual={actual_commit}"
        )
    release_files = dict(plan.get("release_files") or {})
    if not release_files:
        raise Phase1RunnerError("sealed plan lacks release file hashes")
    for relative_path, expected_hash in release_files.items():
        path = (root / str(relative_path)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise Phase1RunnerError("release file escapes repository root") from exc
        if not path.is_file():
            raise Phase1RunnerError(f"release file is missing: {relative_path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != str(expected_hash).lower():
            raise Phase1RunnerError(
                f"release file hash drift: {relative_path}"
            )


def _gpu_process_pids() -> dict[int, set[int]]:
    gpu_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    uuid_to_index: dict[str, int] = {}
    for line in gpu_rows:
        index, uuid = [part.strip() for part in line.split(",", 1)]
        uuid_to_index[uuid] = int(index)
    result = {index: set() for index in range(GPU_COUNT)}
    app_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for line in app_rows:
        if not line.strip():
            continue
        uuid, pid = [part.strip() for part in line.split(",", 1)]
        if uuid in uuid_to_index and pid.isdigit():
            result[uuid_to_index[uuid]].add(int(pid))
    return result


class _Capacity:
    def __init__(self, poll_seconds: float):
        self.poll_seconds = float(poll_seconds)
        self.locks = [threading.Lock() for _ in range(GPU_COUNT)]
        self.owned: dict[int, dict[int, subprocess.Popen]] = {
            gpu: {} for gpu in range(GPU_COUNT)
        }

    def launch(
        self,
        gpu: int,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout,
        stop_event: threading.Event,
    ) -> subprocess.Popen:
        while not stop_event.is_set():
            with self.locks[gpu]:
                live_owned = {
                    pid: process
                    for pid, process in self.owned[gpu].items()
                    if process.poll() is None
                }
                self.owned[gpu] = live_owned
                visible = _gpu_process_pids()[gpu]
                external = visible - set(live_owned)
                if len(external) + len(live_owned) < SLOTS_PER_GPU:
                    process = subprocess.Popen(
                        list(command),
                        cwd=str(cwd),
                        env=dict(env),
                        stdout=stdout,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self.owned[gpu][int(process.pid)] = process
                    return process
            stop_event.wait(self.poll_seconds)
        raise Phase1RunnerError("dispatch stopped before row launch")

    def release(self, gpu: int, pid: int) -> None:
        with self.locks[gpu]:
            self.owned[gpu].pop(int(pid), None)

    def terminate_owned(self, grace_seconds: float = 20.0) -> None:
        owned_processes: list[subprocess.Popen] = []
        for gpu in range(GPU_COUNT):
            with self.locks[gpu]:
                owned_processes.extend(self.owned[gpu].values())
        live_processes = [
            process
            for process in owned_processes
            if process.poll() is None
        ]
        for process in live_processes:
            try:
                os.killpg(int(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + max(0.0, float(grace_seconds))
        while time.time() < deadline:
            if all(process.poll() is not None for process in live_processes):
                return
            time.sleep(0.25)
        for process in live_processes:
            if process.poll() is not None:
                continue
            try:
                os.killpg(int(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_release(args: argparse.Namespace, plan: Mapping[str, Any]) -> int:
    validate_phase1_release_plan(plan, require_launch_authority=True)
    verify_release_checkout(plan, Path(args.repo_root))
    run_root = Path(args.run_root).resolve()
    log_root = Path(args.log_root).resolve()
    if run_root.exists() or log_root.exists():
        raise FileExistsError("refusing to overwrite an existing run or log root")
    run_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    (log_root / "status").mkdir()
    _exclusive_json(log_root / "sealed_plan.json", dict(plan))
    rows_by_slot: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in plan["rows"]:
        worker = row["worker"]
        rows_by_slot[(int(worker["gpu"]), int(worker["slot"]))].append(row)
    capacity = _Capacity(args.poll_seconds)
    stop_event = threading.Event()
    failure_lock = threading.Lock()
    failures: dict[str, list[str]] = defaultdict(list)
    statuses: list[dict[str, Any]] = []
    status_lock = threading.Lock()
    thread_errors: list[dict[str, Any]] = []

    def run_slot(gpu: int, slot: int) -> None:
        for row in rows_by_slot[(gpu, slot)]:
            if stop_event.is_set():
                return
            row_key = str(row["row_key"])
            output_dir = run_root / row_key
            log_path = log_root / f"{row_key}.out"
            pid_path = log_root / f"{row_key}.pid"
            status_path = log_root / "status" / f"{row_key}.json"
            if any(path.exists() for path in (output_dir, log_path, pid_path, status_path)):
                raise FileExistsError(f"row identity collision: {row_key}")
            command = build_phase1_command(
                row,
                run_id=str(plan["run_id"]),
                python_executable=args.python,
                train_script=Path(args.train_script).resolve(),
                wisig_pkl=Path(args.wisig_pkl).resolve(),
                output_dir=output_dir,
                sealed_plan_sha256=str(
                    plan["sealed_content_sha256"]
                ),
                seed_registry_sha256=str(
                    plan["seed_registry_sha256"]
                ),
            )
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(args.repo_root).resolve() / "code"), str(Path(args.repo_root).resolve())]
            )
            output_dir.mkdir()
            started = time.time()
            with log_path.open("x", encoding="utf-8", newline="\n") as log_handle:
                process = capacity.launch(
                    gpu,
                    command,
                    cwd=Path(args.repo_root).resolve(),
                    env=env,
                    stdout=log_handle,
                    stop_event=stop_event,
                )
                pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
                return_code = int(process.wait())
            capacity.release(gpu, int(process.pid))
            terminal_exists = (output_dir / "phase1_terminal_status.json").is_file()
            receipt_valid = False
            completion_error = ""
            try:
                validate_phase1_row_completion(
                    row=row,
                    plan=plan,
                    output_dir=output_dir,
                    return_code=return_code,
                )
                receipt_valid = True
            except Exception as exc:
                completion_error = str(exc)
            status = {
                "row_key": row_key,
                "ablation_id": row["ablation_id"],
                "train_seed": int(row["train_seed"]),
                "gpu": gpu,
                "slot": slot,
                "pid": int(process.pid),
                "return_code": return_code,
                "terminal_manifest_exists": terminal_exists,
                "completion_receipt_valid": receipt_valid,
                "completion_error": completion_error,
                "elapsed_seconds": time.time() - started,
            }
            if not receipt_valid:
                fingerprint = normalize_exception_fingerprint(
                    log_path.read_text(encoding="utf-8", errors="replace")
                    + "\n"
                    + completion_error
                )
                status["exception_fingerprint"] = fingerprint
                with failure_lock:
                    failures[fingerprint].append(row_key)
                    if len(set(failures[fingerprint])) >= 2:
                        stop_event.set()
                        capacity.terminate_owned()
            _exclusive_json(status_path, status)
            with status_lock:
                statuses.append(status)

    def guarded_run_slot(gpu: int, slot: int) -> None:
        try:
            run_slot(gpu, slot)
        except Exception as exc:
            stop_event.set()
            capacity.terminate_owned()
            with status_lock:
                thread_errors.append(
                    {
                        "gpu": gpu,
                        "slot": slot,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    threads = [
        threading.Thread(target=guarded_run_slot, args=slot_key, daemon=False)
        for slot_key in sorted(rows_by_slot)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = {
        "schema": "cvs.full_ablation.phase1_runner_summary.v1",
        "run_id": plan["run_id"],
        "row_count": len(plan["rows"]),
        "completed_count": len(statuses),
        "success_count": sum(
            bool(status["completion_receipt_valid"])
            for status in statuses
        ),
        "failed_count": sum(
            not bool(status["completion_receipt_valid"])
            for status in statuses
        ),
        "systemic_stop": stop_event.is_set(),
        "thread_errors": thread_errors,
        "failure_fingerprints": failures,
        "statuses": sorted(statuses, key=lambda item: item["row_key"]),
    }
    _exclusive_json(log_root / "runner_summary.json", summary)
    if stop_event.is_set() or thread_errors:
        return 20
    return 0 if summary["success_count"] == len(plan["rows"]) else 10


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--wisig-pkl", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--train-script", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = _load_json(Path(args.plan).resolve())
    validate_phase1_release_plan(
        plan,
        require_launch_authority=bool(args.execute),
    )
    if not args.execute:
        commands = [
            build_phase1_command(
                row,
                run_id=str(plan.get("run_id") or "UNSEALED_DRY_RUN"),
                python_executable=args.python,
                train_script=Path(args.train_script),
                wisig_pkl=Path(args.wisig_pkl),
                output_dir=Path(args.run_root) / row["row_key"],
                sealed_plan_sha256=str(
                    plan.get("sealed_content_sha256", "")
                ),
                seed_registry_sha256=str(
                    plan.get("seed_registry_sha256", "")
                ),
            )
            for row in plan["rows"]
        ]
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "row_count": len(commands),
                    "slot_count": len(
                        {
                            (row["worker"]["gpu"], row["worker"]["slot"])
                            for row in plan["rows"]
                        }
                    ),
                    "commands": commands,
                },
                ensure_ascii=False,
            )
        )
        return 0
    return run_release(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
