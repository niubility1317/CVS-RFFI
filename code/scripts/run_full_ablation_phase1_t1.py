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
import sys
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
    build_phase1_t1_rows,
    validate_plan_rows,
)


class Phase1RunnerError(RuntimeError):
    """Raised when the immutable Phase1 release contract is violated."""


class Phase1ProtocolError(Phase1RunnerError):
    """Raised for a P0 identity, protocol, or artifact integrity violation."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_p0_protocol_failure(
    log_text: str,
    completion_exception: Exception | None,
) -> bool:
    if isinstance(completion_exception, Phase1ProtocolError):
        return True
    normalized = str(log_text).lower()
    markers = (
        "phase1ablationconfigerror",
        "formal phase1",
        "resolved phase1 config hash differs",
        "checkpoint selection drift",
        "source-only checkpoint selection forbids",
        "dataset receipt drift",
        "environment receipt drift",
    )
    return any(marker in normalized for marker in markers)


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
    registered_seeds = [
        int(value)
        for value in list(
            plan.get("registered_phase1_train_seeds") or []
        )
    ]
    if len(registered_seeds) != 5 or len(set(registered_seeds)) != 5:
        raise Phase1RunnerError(
            "plan lacks five registered Phase1 seeds"
        )
    expected_rows = build_phase1_t1_rows(
        registered_seeds,
        git_commit=str(plan.get("git_commit", "")),
    )
    canonical_fields = (
        "ablation_id",
        "train_seed",
        "row_key",
        "worker",
        "split_fractions",
        "epochs",
        "checkpoint_selection",
        "method_config_hash",
    )
    actual_by_key = {
        str(row["row_key"]): row for row in rows
    }
    expected_by_key = {
        str(row["row_key"]): row for row in expected_rows
    }
    if set(actual_by_key) != set(expected_by_key):
        raise Phase1RunnerError(
            "Phase1 plan is not the exact registered 6x5 Cartesian matrix"
        )
    for row_key, expected_row in expected_by_key.items():
        actual_row = actual_by_key[row_key]
        if any(
            actual_row.get(field) != expected_row.get(field)
            for field in canonical_fields
        ):
            raise Phase1RunnerError(
                f"Phase1 canonical row drift: {row_key}"
            )
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
        wisig_hash = str(plan.get("wisig_pkl_sha256", "")).lower()
        if len(wisig_hash) != 64 or any(
            char not in "0123456789abcdef"
            for char in wisig_hash
        ):
            raise Phase1RunnerError("sealed plan lacks WiSig SHA256")
        if plan.get("python_environment_id") != "CVS-RFFI":
            raise Phase1RunnerError(
                "sealed plan requires the verified CVS-RFFI environment"
            )
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
    wisig_pkl_sha256: str = "",
    dataset_receipt_path: str = "",
    dataset_receipt_sha256: str = "",
    environment_receipt_path: str = "",
    environment_receipt_sha256: str = "",
    python_environment_id: str = "",
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
        "--wisig_pkl_sha256",
        str(wisig_pkl_sha256),
        "--dataset_receipt_path",
        str(dataset_receipt_path),
        "--dataset_receipt_sha256",
        str(dataset_receipt_sha256),
        "--environment_receipt_sha256",
        str(environment_receipt_sha256),
        "--environment_receipt_path",
        str(environment_receipt_path),
        "--python_environment_id",
        str(python_environment_id),
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
    dataset_receipt_sha256: str = "",
    environment_receipt_sha256: str = "",
    dataset_receipt_path: Path | None = None,
    environment_receipt_path: Path | None = None,
) -> dict[str, Any]:
    terminal_path = output_dir / "phase1_terminal_status.json"
    receipt_path = output_dir / "phase1_training_completion_receipt.json"
    if not terminal_path.is_file() or not receipt_path.is_file():
        raise Phase1RunnerError("row lacks terminal or completion receipt")
    terminal = _load_json(terminal_path)
    receipt = _load_json(receipt_path)
    terminal_status = str(terminal.get("status", ""))
    receipt_status = str(receipt.get("terminal_status", ""))
    if "NON_PROMOTABLE_P0_DISABLED" in {
        terminal_status,
        receipt_status,
    }:
        raise Phase1ProtocolError(
            "row terminal status is NON_PROMOTABLE_P0_DISABLED"
        )
    if terminal_status != receipt_status:
        raise Phase1ProtocolError(
            "row terminal status differs from completion receipt"
        )
    public_receipts = (
        (
            "dataset",
            dataset_receipt_path,
            str(dataset_receipt_sha256),
        ),
        (
            "environment",
            environment_receipt_path,
            str(environment_receipt_sha256),
        ),
    )
    for label, public_path, expected_hash in public_receipts:
        if (
            public_path is None
            or not Path(public_path).is_file()
            or _sha256_path(Path(public_path)) != expected_hash
        ):
            raise Phase1ProtocolError(
                f"row public {label} receipt hash drift"
            )
        public_payload = _load_json(Path(public_path))
        if (
            public_payload != dict(receipt.get(f"{label}_receipt") or {})
            or public_payload
            != dict(terminal.get(f"{label}_receipt") or {})
        ):
            raise Phase1ProtocolError(
                f"row public {label} receipt content drift"
            )
    expected = {
        "run_id": str(plan["run_id"]),
        "row_key": str(row["row_key"]),
        "ablation_id": str(row["ablation_id"]),
        "git_commit": str(plan["git_commit"]),
        "sealed_plan_sha256": str(plan["sealed_content_sha256"]),
        "seed_registry_sha256": str(plan["seed_registry_sha256"]),
        "wisig_pkl_sha256": str(plan["wisig_pkl_sha256"]),
        "dataset_receipt_sha256": str(dataset_receipt_sha256),
        "environment_receipt_sha256": str(
            environment_receipt_sha256
        ),
    }
    for key, value in expected.items():
        if str(receipt.get(key, "")) != value:
            raise Phase1ProtocolError(
                f"row completion receipt identity drift: {key}"
            )
    if int(receipt.get("train_seed", -1)) != int(row["train_seed"]):
        raise Phase1ProtocolError("row completion receipt train-seed drift")
    if (
        str(receipt.get("resolved_config_hash", ""))
        != str(row.get("config_hash", ""))
        or str(receipt.get("method_config_hash", ""))
        != str(row.get("method_config_hash", ""))
    ):
        raise Phase1ProtocolError("row completion receipt config-hash drift")
    actual_terminal_hash = hashlib.sha256(
        terminal_path.read_bytes()
    ).hexdigest()
    if (
        str(receipt.get("terminal_manifest_sha256", ""))
        != actual_terminal_hash
    ):
        raise Phase1ProtocolError("row terminal-manifest hash drift")
    resource_path = output_dir / "phase1_resource_summary.json"
    if (
        not resource_path.is_file()
        or str(receipt.get("resource_summary_sha256", ""))
        != hashlib.sha256(resource_path.read_bytes()).hexdigest()
    ):
        raise Phase1ProtocolError("row resource-summary hash drift")
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
            raise Phase1ProtocolError("row prototype artifact hash drift")
    prototype_hashes = dict(receipt.get("prototype_hashes") or {})
    if set(prototype_hashes) != {
        "prototype_path",
        "prototype_json_path",
    } or any(
        len(str(value)) != 64
        for value in prototype_hashes.values()
    ):
        raise Phase1ProtocolError(
            "row prototype artifact hashes are incomplete"
        )
    checkpoint_path = Path(
        str(terminal.get("selected_checkpoint", ""))
    )
    selected_checkpoint_sha256 = str(
        receipt.get("selected_checkpoint_sha256", "")
    )
    if (
        not checkpoint_path.is_file()
        or len(selected_checkpoint_sha256) != 64
        or _sha256_path(checkpoint_path)
        != selected_checkpoint_sha256
        or str(terminal.get("selected_checkpoint_sha256", ""))
        != selected_checkpoint_sha256
    ):
        raise Phase1ProtocolError(
            "row selected-checkpoint hash drift"
        )
    split_receipt = dict(receipt.get("source_split_receipt") or {})
    split_payload = {
        key: value
        for key, value in split_receipt.items()
        if key != "split_manifest_sha256"
    }
    if (
        len(str(split_receipt.get("split_manifest_sha256", ""))) != 64
        or _canonical_payload_hash(split_payload)
        != str(split_receipt.get("split_manifest_sha256", ""))
        or split_receipt
        != dict(terminal.get("source_split_receipt") or {})
        or str(split_receipt.get("wisig_pkl_sha256", ""))
        != str(plan["wisig_pkl_sha256"])
        or any(
            len(str(split_receipt.get(key, ""))) != 64
            for key in (
                "labeled_indices_sha256",
                "unlabeled_indices_sha256",
                "source_validation_indices_sha256",
            )
        )
        or int(
            split_receipt.get(
                "source_target_receiver_overlap_count",
                -1,
            )
        )
        != 0
    ):
        raise Phase1ProtocolError("row completion receipt split evidence invalid")
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
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise Phase1RunnerError(
            "release checkout has tracked file drift"
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
        actual_hash = _sha256_path(path)
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
    repo_root = Path(args.repo_root).resolve()
    verify_release_checkout(plan, repo_root)
    expected_train_script = (
        repo_root / "code" / "SSDG" / "train_ssdg.py"
    ).resolve()
    if Path(args.train_script).resolve() != expected_train_script:
        raise Phase1RunnerError(
            "execute mode requires the reviewed train_ssdg.py"
        )
    if Path(args.python).resolve() != Path(sys.executable).resolve():
        raise Phase1RunnerError(
            "child Python must equal the reviewed runner interpreter"
        )
    expected_environment_id = str(
        plan.get("python_environment_id", "")
    ).strip()
    if (
        not expected_environment_id
        or Path(sys.prefix).name.lower()
        != expected_environment_id.lower()
    ):
        raise Phase1RunnerError(
            "formal Phase1 runner environment differs from the sealed plan"
        )
    wisig_path = Path(args.wisig_pkl).resolve()
    if not wisig_path.is_file():
        raise Phase1RunnerError("WiSig pickle is missing")
    actual_wisig_hash = _sha256_path(wisig_path)
    if actual_wisig_hash != str(plan["wisig_pkl_sha256"]):
        raise Phase1ProtocolError("WiSig pickle SHA256 drift")
    run_root = Path(args.run_root).resolve()
    log_root = Path(args.log_root).resolve()
    if run_root.exists() or log_root.exists():
        raise FileExistsError("refusing to overwrite an existing run or log root")
    run_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    (log_root / "status").mkdir()
    _exclusive_json(log_root / "sealed_plan.json", dict(plan))
    wisig_stat = wisig_path.stat()
    dataset_receipt_path = log_root / "dataset_receipt.json"
    _exclusive_json(
        dataset_receipt_path,
        {
            "schema": "cvs.phase1.dataset_receipt.v1",
            "sealed_plan_sha256": str(
                plan["sealed_content_sha256"]
            ),
            "wisig_pkl_sha256": actual_wisig_hash,
            "wisig_pkl_path": str(wisig_path),
            "wisig_pkl_size": int(wisig_stat.st_size),
            "wisig_pkl_mtime_ns": int(wisig_stat.st_mtime_ns),
        },
    )
    dataset_receipt_sha256 = hashlib.sha256(
        dataset_receipt_path.read_bytes()
    ).hexdigest()
    environment_receipt_path = log_root / "environment_receipt.json"
    import numpy as np
    import torch

    _exclusive_json(
        environment_receipt_path,
        {
            "schema": "cvs.phase1.python_environment_receipt.v1",
            "environment_id": expected_environment_id,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "python_prefix": str(Path(sys.prefix).resolve()),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": int(torch.cuda.device_count()),
            "numpy_version": str(np.__version__),
        },
    )
    environment_receipt_sha256 = hashlib.sha256(
        environment_receipt_path.read_bytes()
    ).hexdigest()
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
                wisig_pkl_sha256=actual_wisig_hash,
                dataset_receipt_path=str(dataset_receipt_path),
                dataset_receipt_sha256=dataset_receipt_sha256,
                environment_receipt_path=str(
                    environment_receipt_path
                ),
                environment_receipt_sha256=(
                    environment_receipt_sha256
                ),
                python_environment_id=expected_environment_id,
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
            completion_exception: Exception | None = None
            try:
                validate_phase1_row_completion(
                    row=row,
                    plan=plan,
                    output_dir=output_dir,
                    return_code=return_code,
                    dataset_receipt_sha256=(
                        dataset_receipt_sha256
                    ),
                    environment_receipt_sha256=(
                        environment_receipt_sha256
                    ),
                    dataset_receipt_path=dataset_receipt_path,
                    environment_receipt_path=(
                        environment_receipt_path
                    ),
                )
                receipt_valid = True
            except Exception as exc:
                completion_exception = exc
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
                "p0_protocol_violation": isinstance(
                    completion_exception,
                    Phase1ProtocolError,
                ),
                "elapsed_seconds": time.time() - started,
            }
            if not receipt_valid:
                log_text = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                p0_protocol_violation = is_p0_protocol_failure(
                    log_text,
                    completion_exception,
                )
                status["p0_protocol_violation"] = (
                    p0_protocol_violation
                )
                fingerprint = normalize_exception_fingerprint(
                    log_text + "\n" + completion_error
                )
                status["exception_fingerprint"] = fingerprint
                with failure_lock:
                    failures[fingerprint].append(row_key)
                    if p0_protocol_violation or len(
                        set(failures[fingerprint])
                    ) >= 2:
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
