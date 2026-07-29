#!/usr/bin/env python3
"""Run a sealed Phase2 matrix with two bounded processes per GPU."""

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

from cvsrffi.full_ablation_spec import GPU_COUNT, SLOTS_PER_GPU
from cvsrffi.stage2_ablation_release import (
    RUNNER_SUMMARY_SCHEMA,
    SCORE_COMPLETION_SCHEMA,
    TERMINAL_ROW_SCHEMA,
    Stage2AblationReleaseError,
    canonical_json_bytes,
    sha256_object,
    validate_sealed_stage2_plan,
)
from cvsrffi.stage2_prediction_artifact import (
    PredictionArtifactError,
    verify_prediction_artifact,
)
from cvsrffi.stage2_ablation_row_executor import ROW_EXECUTION_SCHEMA
from cvsrffi.stage2_ablation_truth_scorer import (
    SAME_ROW_SCORE_SCHEMA,
    build_failed_row_record,
    write_row_record_exclusive,
)


class Stage2AblationRunnerError(RuntimeError):
    """Raised when a sealed Phase2 physical row does not close."""


class Stage2AblationProtocolError(Stage2AblationRunnerError):
    """Raised for a release-binding or protocol P0 failure."""


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise Stage2AblationRunnerError(
            f"JSON root must be an object: {path}"
        )
    return dict(value)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing runner evidence")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def normalize_exception_fingerprint(log_text: str) -> str:
    lines = [
        line.strip()
        for line in str(log_text).splitlines()
        if line.strip()
    ]
    exception_lines = [
        line
        for line in lines
        if any(
            token in line.lower()
            for token in ("error", "exception", "traceback")
        )
    ]
    selected = "\n".join((exception_lines or lines)[-12:])
    selected = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", selected)
    selected = re.sub(r"[A-Za-z]:[\\/][^\s:]+", "<PATH>", selected)
    selected = re.sub(r"/[^\s:]+", "<PATH>", selected)
    selected = re.sub(r"\b\d+\b", "<N>", selected)
    selected = re.sub(r"\s+", " ", selected).strip().lower()
    return hashlib.sha256(selected.encode("utf-8")).hexdigest()


def is_p0_protocol_failure(
    log_text: str,
    error: BaseException | None,
) -> bool:
    message = (
        str(log_text)
        + "\n"
        + (str(error) if error is not None else "")
    ).lower()
    markers = (
        "p0_protocol_violation",
        "query_truth_opened\":true",
        '"query_fit_access":true',
        '"clean_source_runtime_access":true',
        "output overwrite risk",
        "checkout commit drift",
        "tracked file drift",
        "candidate lock drift",
    )
    return isinstance(error, Stage2AblationProtocolError) or any(
        marker in message for marker in markers
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
    if set(result) != set(range(GPU_COUNT)):
        raise Stage2AblationRunnerError(
            "N607 must expose exactly GPU indices 0-7"
        )
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
        raise Stage2AblationRunnerError(
            "dispatch stopped before physical row launch"
        )

    def release(self, gpu: int, pid: int) -> None:
        with self.locks[gpu]:
            self.owned[gpu].pop(int(pid), None)

    def launch_auxiliary(
        self,
        gpu: int,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout,
    ) -> subprocess.Popen:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self.locks[gpu]:
            self.owned[gpu][int(process.pid)] = process
        return process

    def terminate_owned(self, grace_seconds: float = 20.0) -> None:
        owned: list[subprocess.Popen] = []
        for gpu in range(GPU_COUNT):
            with self.locks[gpu]:
                owned.extend(self.owned[gpu].values())
        live = [process for process in owned if process.poll() is None]
        for process in live:
            try:
                os.killpg(int(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + max(0.0, float(grace_seconds))
        while time.time() < deadline:
            if all(process.poll() is not None for process in live):
                return
            time.sleep(0.25)
        for process in live:
            if process.poll() is not None:
                continue
            try:
                os.killpg(int(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def verify_release_checkout(
    plan: Mapping[str, Any],
    repo_root: Path,
) -> None:
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    expected_commit = str(plan.get("git_commit", "")).strip().lower()
    if actual_commit != expected_commit:
        raise Stage2AblationProtocolError(
            "checkout commit drift"
        )
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise Stage2AblationProtocolError(
            "release checkout is not clean"
        )


def _validate_row_execution_receipt(
    path: str | Path,
    physical: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        physical.get("mode") == "reuse_prediction"
        and hashlib.sha256(Path(path).read_bytes()).hexdigest()
        != physical.get("reuse_row_execution_receipt_sha256")
    ):
        raise Stage2AblationRunnerError(
            "reused row execution receipt hash drift"
        )
    receipt = _load_json(path)
    prediction = receipt.get("prediction")
    if (
        receipt.get("schema") != ROW_EXECUTION_SCHEMA
        or receipt.get("status")
        != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
        or receipt.get("row_id")
        != physical["physical_execution_id"]
        or receipt.get("receiver") in {"", None}
        or receipt.get("query_truth_opened") is not False
        or int(receipt.get("fit_query_rows_used", -1)) != 0
        or receipt.get("stage") != physical["stage"]
        or int(receipt.get("k_shot", -1)) != int(physical["k_shot"])
        or receipt.get("receiver") != physical["receiver"]
        or receipt.get("input_identity") != physical["input_identity"]
        or not isinstance(prediction, Mapping)
        or not isinstance(receipt.get("behavior"), Mapping)
        or not isinstance(receipt.get("quantization"), Mapping)
        or not isinstance(receipt.get("resource"), Mapping)
    ):
        raise Stage2AblationRunnerError(
            "physical row lacks a complete immutable prediction receipt"
        )
    try:
        verified = verify_prediction_artifact(
            prediction["path"],
            expected_artifact_sha256=prediction["artifact_sha256"],
            expected_seal_sha256=prediction["seal_sha256"],
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        PredictionArtifactError,
    ) as exc:
        raise Stage2AblationRunnerError(
            "physical row prediction artifact verification failed"
        ) from exc
    manifest = verified["manifest"]
    if (
        manifest.get("row_id") != physical["physical_execution_id"]
        or manifest.get("receiver") != physical["receiver"]
        or int(manifest.get("k_shot", -1)) != int(physical["k_shot"])
    ):
        raise Stage2AblationRunnerError(
            "physical row prediction binding drift"
        )
    return receipt


def _validate_score_completion(
    path: str | Path,
    logical: Mapping[str, Any],
    physical: Mapping[str, Any],
) -> dict[str, Any]:
    result = _load_json(path)
    if (
        result.get("schema") != SCORE_COMPLETION_SCHEMA
        or result.get("status") != "PASS"
        or result.get("logical_row_key")
        != logical["logical_row_key"]
        or result.get("ablation_id") != logical["ablation_id"]
        or result.get("physical_execution_id")
        != physical["physical_execution_id"]
        or result.get("effective_config_hash")
        != logical["effective_config_hash"]
        or result.get("alias_of") != logical["alias_of"]
        or result.get("score_output_path")
        != logical["score_output_path"]
        or result.get("score_output_sha256") != hashlib.sha256(
            Path(logical["score_output_path"]).read_bytes()
        ).hexdigest()
        or int(result.get("formal_scenario_count", -1)) != 3
        or result.get("performance_values_present") is not False
    ):
        raise Stage2AblationRunnerError(
            "logical row score completion receipt is incomplete"
        )
    return result


def _predict_command(
    *,
    python: Path,
    predictor_script: Path,
    request_path: str,
) -> list[str]:
    return [
        str(python),
        str(predictor_script),
        "--request",
        str(request_path),
    ]


def _validate_request_hash(
    path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    request = _load_json(path)
    if sha256_object(request) != str(expected_sha256):
        raise Stage2AblationProtocolError(
            "sealed child request content drift"
        )
    return request


def _score_command(
    *,
    python: Path,
    scorer_script: Path,
    request_path: str,
) -> list[str]:
    return [
        str(python),
        str(scorer_script),
        "--request",
        str(request_path),
    ]


def dry_run_commands(
    plan: Mapping[str, Any],
    *,
    python: Path,
    predictor_script: Path,
    scorer_script: Path,
) -> dict[str, Any]:
    validate_sealed_stage2_plan(plan)
    commands = []
    for physical in plan["physical_rows"]:
        predictor = None
        if physical["mode"] == "execute":
            predictor = _predict_command(
                python=python,
                predictor_script=predictor_script,
                request_path=physical["prediction_request_path"],
            )
        scorers = [
            _score_command(
                python=python,
                scorer_script=scorer_script,
                request_path=logical["score_request_path"],
            )
            for logical in physical["logical_rows"]
        ]
        commands.append(
            {
                "physical_execution_id": physical[
                    "physical_execution_id"
                ],
                "gpu": int(physical["worker"]["gpu"]),
                "slot": int(physical["worker"]["slot"]),
                "mode": physical["mode"],
                "predictor": predictor,
                "scorers": scorers,
            }
        )
    return {
        "dry_run": True,
        "logical_row_count": plan["logical_row_count"],
        "physical_execution_count": plan[
            "physical_execution_count"
        ],
        "reused_physical_count": plan["reused_physical_count"],
        "alias_logical_count": plan["alias_logical_count"],
        "slot_count": len(
            {
                (
                    int(physical["worker"]["gpu"]),
                    int(physical["worker"]["slot"]),
                )
                for physical in plan["physical_rows"]
            }
        ),
        "commands": commands,
    }


def run_release(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
) -> int:
    validate_sealed_stage2_plan(plan)
    repo_root = Path(args.repo_root).resolve()
    verify_release_checkout(plan, repo_root)
    if Path(sys.prefix).name != plan["python_environment_id"]:
        raise Stage2AblationProtocolError(
            "runner is not using the sealed CVS-RFFI environment"
        )
    python = Path(args.python).resolve()
    if python != Path(sys.executable).resolve():
        raise Stage2AblationProtocolError(
            "child Python differs from the reviewed runner interpreter"
        )
    expected_predictor = (
        repo_root / "code" / "scripts"
        / "run_full_ablation_stage2_row.py"
    ).resolve()
    expected_scorer = (
        repo_root / "code" / "scripts"
        / "score_full_ablation_stage2_row.py"
    ).resolve()
    if (
        Path(args.predictor_script).resolve() != expected_predictor
        or Path(args.scorer_script).resolve() != expected_scorer
    ):
        raise Stage2AblationProtocolError(
            "runner script path differs from the reviewed release"
        )
    run_root = Path(plan["run_root"])
    log_root = Path(plan["log_root"])
    if run_root.exists() or log_root.exists():
        raise FileExistsError(
            "refusing to overwrite an existing Phase2 run or log root"
        )
    run_root.mkdir(parents=True)
    (run_root / "physical").mkdir()
    (run_root / "logical").mkdir()
    log_root.mkdir(parents=True)
    (log_root / "status").mkdir()
    (log_root / "launch").mkdir()
    _exclusive_json(log_root / "sealed_plan.json", dict(plan))
    _exclusive_json(
        log_root / "runner_start.json",
        {
            "schema": "cvs.full_ablation.phase2.runner_start.v1",
            "run_id": plan["run_id"],
            "main_pid": os.getpid(),
            "cwd": str(repo_root),
            "python": str(python),
            "python_environment_id": plan["python_environment_id"],
            "gpu_count": GPU_COUNT,
            "slots_per_gpu": SLOTS_PER_GPU,
            "logical_row_count": plan["logical_row_count"],
            "physical_execution_count": plan[
                "physical_execution_count"
            ],
            "performance_values_visible_to_scheduler": False,
        },
    )

    queues: dict[tuple[int, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for physical in plan["physical_rows"]:
        worker = physical["worker"]
        queues[(int(worker["gpu"]), int(worker["slot"]))].append(
            physical
        )
    capacity = _Capacity(args.poll_seconds)
    stop_event = threading.Event()
    status_lock = threading.Lock()
    failure_lock = threading.Lock()
    statuses: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = defaultdict(list)
    thread_errors: list[dict[str, Any]] = []

    def run_physical(
        physical: Mapping[str, Any],
        gpu: int,
        slot: int,
    ) -> None:
        physical_id = str(physical["physical_execution_id"])
        log_path = Path(physical["log_path"])
        status_path = Path(physical["status_path"])
        if any(
            path.exists()
            for path in (
                log_path,
                status_path,
                log_root / "launch" / f"{physical_id}.json",
            )
        ):
            raise FileExistsError(
                f"physical row evidence collision: {physical_id}"
            )
        started = time.time()
        predictor_return_code: int | None = None
        scorer_return_codes: list[int] = []
        process_pid: int | None = None
        error: BaseException | None = None
        with log_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as log_handle:
            try:
                if physical["mode"] == "execute":
                    _validate_request_hash(
                        physical["prediction_request_path"],
                        physical["prediction_request_sha256"],
                    )
                    env = dict(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    env["PYTHONPATH"] = os.pathsep.join(
                        [str(repo_root / "code"), str(repo_root)]
                    )
                    command = _predict_command(
                        python=python,
                        predictor_script=expected_predictor,
                        request_path=physical[
                            "prediction_request_path"
                        ],
                    )
                    process = capacity.launch(
                        gpu,
                        command,
                        cwd=repo_root,
                        env=env,
                        stdout=log_handle,
                        stop_event=stop_event,
                    )
                    process_pid = int(process.pid)
                    _exclusive_json(
                        log_root
                        / "launch"
                        / f"{physical_id}.json",
                        {
                            "schema": (
                                "cvs.full_ablation.phase2."
                                "physical_launch.v1"
                            ),
                            "physical_execution_id": physical_id,
                            "gpu": gpu,
                            "slot": slot,
                            "pid": process_pid,
                            "cwd": str(repo_root),
                            "command": command,
                            "output_root": str(
                                run_root / "physical" / physical_id
                            ),
                        },
                    )
                    predictor_return_code = int(process.wait())
                    capacity.release(gpu, process_pid)
                    if predictor_return_code != 0:
                        raise Stage2AblationRunnerError(
                            "predictor returned nonzero"
                        )
                else:
                    _exclusive_json(
                        log_root
                        / "launch"
                        / f"{physical_id}.json",
                        {
                            "schema": (
                                "cvs.full_ablation.phase2."
                                "physical_launch.v1"
                            ),
                            "physical_execution_id": physical_id,
                            "gpu": gpu,
                            "slot": slot,
                            "pid": None,
                            "cwd": str(repo_root),
                            "command": [],
                            "reuse_row_execution_receipt": physical[
                                "row_execution_receipt"
                            ],
                        },
                    )
                    log_handle.write(
                        "REUSE_PREDICTION: existing complete immutable "
                        "prediction receipt selected\n"
                    )
                    log_handle.flush()
                _validate_row_execution_receipt(
                    physical["row_execution_receipt"],
                    physical,
                )
                for logical in physical["logical_rows"]:
                    _validate_request_hash(
                        logical["score_request_path"],
                        logical["score_request_sha256"],
                    )
                    command = _score_command(
                        python=python,
                        scorer_script=expected_scorer,
                        request_path=logical["score_request_path"],
                    )
                    log_handle.write(
                        "SCORE_START "
                        + str(logical["logical_row_key"])
                        + "\n"
                    )
                    log_handle.flush()
                    scorer = capacity.launch_auxiliary(
                        gpu,
                        command,
                        cwd=repo_root,
                        env={
                            **os.environ,
                            "PYTHONPATH": os.pathsep.join(
                                [
                                    str(repo_root / "code"),
                                    str(repo_root),
                                ]
                            ),
                        },
                        stdout=log_handle,
                    )
                    scorer_return_code = int(scorer.wait())
                    capacity.release(gpu, int(scorer.pid))
                    scorer_return_codes.append(
                        scorer_return_code
                    )
                    if scorer_return_code != 0:
                        raise Stage2AblationRunnerError(
                            "truth-side scorer returned nonzero"
                        )
                    _validate_score_completion(
                        logical["score_completion_path"],
                        logical,
                        physical,
                    )
            except BaseException as exc:
                error = exc
                raise
            finally:
                if process_pid is not None:
                    capacity.release(gpu, process_pid)
        status = {
            "schema": TERMINAL_ROW_SCHEMA,
            "run_id": plan["run_id"],
            "physical_execution_id": physical_id,
            "representative_logical_row_key": physical[
                "representative_logical_row_key"
            ],
            "mode": physical["mode"],
            "gpu": gpu,
            "slot": slot,
            "pid": process_pid,
            "predictor_return_code": predictor_return_code,
            "scorer_return_codes": scorer_return_codes,
            "logical_score_count": len(
                physical["logical_rows"]
            ),
            "expected_logical_score_count": len(
                physical["logical_rows"]
            ),
            "prediction_complete": True,
            "scores_complete": True,
            "status": "COMPLETE",
            "elapsed_seconds": time.time() - started,
        }
        _exclusive_json(status_path, status)
        with status_lock:
            statuses.append(status)

    def guarded_slot(gpu: int, slot: int) -> None:
        try:
            for physical in queues[(gpu, slot)]:
                if stop_event.is_set():
                    return
                physical_id = str(
                    physical["physical_execution_id"]
                )
                try:
                    run_physical(physical, gpu, slot)
                except BaseException as exc:
                    log_path = Path(physical["log_path"])
                    log_text = (
                        log_path.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )
                        if log_path.is_file()
                        else ""
                    )
                    fingerprint = normalize_exception_fingerprint(
                        log_text + "\n" + repr(exc)
                    )
                    prediction_exists = False
                    try:
                        _validate_row_execution_receipt(
                            physical["row_execution_receipt"],
                            physical,
                        )
                        prediction_exists = True
                    except Exception:
                        prediction_exists = False
                    p0 = is_p0_protocol_failure(log_text, exc)
                    failed = {
                        "schema": TERMINAL_ROW_SCHEMA,
                        "run_id": plan["run_id"],
                        "physical_execution_id": physical_id,
                        "representative_logical_row_key": physical[
                            "representative_logical_row_key"
                        ],
                        "mode": physical["mode"],
                        "gpu": gpu,
                        "slot": slot,
                        "status": "FAILED",
                        "prediction_complete": prediction_exists,
                        "scores_complete": False,
                        "zero_prediction": not prediction_exists,
                        "p0_protocol_violation": p0,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "exception_fingerprint": fingerprint,
                    }
                    failed_logical_count = 0
                    for logical in physical["logical_rows"]:
                        output_path = Path(
                            logical["score_output_path"]
                        )
                        if output_path.exists():
                            continue
                        failure_record = build_failed_row_record(
                            row_identity={
                                "logical_row_key": logical[
                                    "logical_row_key"
                                ],
                                "ablation_id": logical[
                                    "ablation_id"
                                ],
                                "physical_execution_id": (
                                    physical_id
                                ),
                                "effective_config_hash": logical[
                                    "effective_config_hash"
                                ],
                                "alias_of": logical["alias_of"],
                            },
                            stage=str(physical["stage"]),
                            receiver=str(physical["receiver"]),
                            k_shot=int(physical["k_shot"]),
                            failure_code=type(exc).__name__,
                            failure_fingerprint=fingerprint,
                            zero_prediction=not prediction_exists,
                        )
                        write_row_record_exclusive(
                            output_path,
                            failure_record,
                        )
                        failed_logical_count += 1
                    failed["failed_logical_record_count"] = (
                        failed_logical_count
                    )
                    status_path = Path(physical["status_path"])
                    if not status_path.exists():
                        _exclusive_json(status_path, failed)
                    with status_lock:
                        statuses.append(failed)
                    systemic = p0
                    if not prediction_exists:
                        with failure_lock:
                            failures[fingerprint].append(physical_id)
                            systemic = systemic or len(
                                set(failures[fingerprint])
                            ) >= 2
                    if systemic:
                        stop_event.set()
                        capacity.terminate_owned()
        except BaseException as exc:
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
        threading.Thread(
            target=guarded_slot,
            args=worker,
            daemon=False,
        )
        for worker in sorted(queues)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if stop_event.is_set():
        closed_ids = {
            str(status["physical_execution_id"])
            for status in statuses
        }
        for physical in plan["physical_rows"]:
            physical_id = str(physical["physical_execution_id"])
            if physical_id in closed_ids:
                continue
            fingerprint = hashlib.sha256(
                b"systemic_stop_before_dispatch"
            ).hexdigest()
            logical_count = 0
            for logical in physical["logical_rows"]:
                output_path = Path(logical["score_output_path"])
                if output_path.exists():
                    continue
                write_row_record_exclusive(
                    output_path,
                    build_failed_row_record(
                        row_identity={
                            "logical_row_key": logical[
                                "logical_row_key"
                            ],
                            "ablation_id": logical["ablation_id"],
                            "physical_execution_id": physical_id,
                            "effective_config_hash": logical[
                                "effective_config_hash"
                            ],
                            "alias_of": logical["alias_of"],
                        },
                        stage=str(physical["stage"]),
                        receiver=str(physical["receiver"]),
                        k_shot=int(physical["k_shot"]),
                        failure_code=(
                            "NOT_LAUNCHED_SYSTEMIC_STOP"
                        ),
                        failure_fingerprint=fingerprint,
                        zero_prediction=True,
                    ),
                )
                logical_count += 1
            status = {
                "schema": TERMINAL_ROW_SCHEMA,
                "run_id": plan["run_id"],
                "physical_execution_id": physical_id,
                "representative_logical_row_key": physical[
                    "representative_logical_row_key"
                ],
                "mode": physical["mode"],
                "gpu": int(physical["worker"]["gpu"]),
                "slot": int(physical["worker"]["slot"]),
                "status": "NOT_LAUNCHED_SYSTEMIC_STOP",
                "prediction_complete": False,
                "scores_complete": False,
                "zero_prediction": True,
                "p0_protocol_violation": False,
                "exception_fingerprint": fingerprint,
                "failed_logical_record_count": logical_count,
                "no_performance_result": True,
            }
            status_path = Path(physical["status_path"])
            if not status_path.exists():
                _exclusive_json(status_path, status)
            statuses.append(status)
    summary = {
        "schema": RUNNER_SUMMARY_SCHEMA,
        "run_id": plan["run_id"],
        "logical_row_count": plan["logical_row_count"],
        "physical_execution_count": plan[
            "physical_execution_count"
        ],
        "launched_physical_count": len(
            list((log_root / "launch").glob("*.json"))
        ),
        "completed_physical_count": sum(
            status["status"] == "COMPLETE"
            for status in statuses
        ),
        "failed_physical_count": sum(
            status["status"] == "FAILED"
            for status in statuses
        ),
        "not_launched_systemic_stop_count": sum(
            status["status"] == "NOT_LAUNCHED_SYSTEMIC_STOP"
            for status in statuses
        ),
        "completed_logical_score_count": sum(
            int(status.get("logical_score_count", 0))
            for status in statuses
            if status["status"] == "COMPLETE"
        ),
        "reused_physical_count": sum(
            status.get("mode") == "reuse_prediction"
            and status.get("status") == "COMPLETE"
            for status in statuses
        ),
        "systemic_stop": stop_event.is_set(),
        "performance_values_visible_to_scheduler": False,
        "failure_fingerprints": dict(failures),
        "thread_errors": thread_errors,
        "statuses": sorted(
            statuses,
            key=lambda item: item["physical_execution_id"],
        ),
    }
    _exclusive_json(log_root / "runner_summary.json", summary)
    if stop_event.is_set() or thread_errors:
        return 20
    if (
        summary["completed_physical_count"]
        != plan["physical_execution_count"]
        or summary["completed_logical_score_count"]
        != plan["logical_row_count"]
        or summary["failed_physical_count"] != 0
    ):
        return 10
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--predictor-script", required=True)
    parser.add_argument("--scorer-script", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = _load_json(args.plan)
    validate_sealed_stage2_plan(plan)
    if not args.execute:
        print(
            json.dumps(
                dry_run_commands(
                    plan,
                    python=Path(args.python),
                    predictor_script=Path(args.predictor_script),
                    scorer_script=Path(args.scorer_script),
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    return run_release(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
