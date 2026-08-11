#!/usr/bin/env python3
"""Prepare, smoke, and run the frozen D92-E0D Hard12-v2 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_e0d_hard12 import (  # noqa: E402
    CANONICAL_SELECTION_SHA256,
    CONTEXT_SHA256,
    D92E0DHard12V2Error,
    build_hard12v2_manifest,
)


PREDICTION_ENTRY = CODE_ROOT / "scripts" / "run_d92_e0d_prediction.py"
SCORING_ENTRY = CODE_ROOT / "scripts" / "score_d92_be_prediction.py"
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class D92E0DHard12V2RunnerError(RuntimeError):
    """Raised when the immutable Hard12-v2 runner contract drifts."""


D92E0DHard12RunnerError = D92E0DHard12V2RunnerError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def _append_event(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=True, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_manifest(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise D92E0DHard12V2RunnerError("matrix manifest must be a regular file")
    if _sha256_file(source) != str(expected_sha256).lower():
        raise D92E0DHard12V2RunnerError("matrix manifest SHA drift")
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if (
        payload.get("schema") != "cvs.phase2.d92_e0d_hard12v2.matrix.v1"
        or payload.get("status") != "FROZEN_DEVELOPMENT_MATRIX"
        or payload.get("protocol_schema") != "p2_min_v1"
        or payload.get("selection_sha256") != CANONICAL_SELECTION_SHA256
        or payload.get("context_sha256") != CONTEXT_SHA256
        or int(payload.get("shard_count", -1)) != 8
        or not isinstance(payload.get("jobs"), list)
    ):
        raise D92E0DHard12V2RunnerError("matrix manifest contract drift")
    return payload


def _prediction_command(
    job: Mapping[str, Any],
    *,
    ground_component_dir: str,
    ground_manifest_sha256: str,
    device: str,
    output_root: str | Path | None = None,
) -> list[str]:
    package = job["packages"]
    output = Path(output_root) if output_root is not None else Path(job["output_root"]) / "diag"
    command = [sys.executable, str(PREDICTION_ENTRY)]
    for state in ("before", "after"):
        for phase in ("enrollment", "apply"):
            item = package[f"{state}_{phase}"]
            prefix = f"--{state}-{phase}"
            if item.get("expected_seal_sha256") is None:
                raise D92E0DHard12V2RunnerError("package seal SHA was not materialized")
            command.extend(
                [
                    f"{prefix}-package-root",
                    str(item["package_root"]),
                    f"{prefix}-seal-path",
                    str(item["detached_seal_path"]),
                    f"{prefix}-seal-sha256",
                    str(item["expected_seal_sha256"]),
                ]
            )
    command.extend(
        [
            "--ground-component-dir",
            str(ground_component_dir),
            "--ground-manifest-sha256",
            str(ground_manifest_sha256),
            "--arm",
            str(job["arm_id"]),
            "--output-root",
            str(output),
            "--device",
            str(device),
        ]
    )
    return command


def _score_command(job: Mapping[str, Any]) -> list[str]:
    root = Path(job["output_root"])
    return [
        sys.executable,
        str(SCORING_ENTRY),
        "--before-prediction",
        str(root / "diag" / "before" / "prediction_artifact.npz"),
        "--after-prediction",
        str(root / "diag" / "after" / "prediction_artifact.npz"),
        "--truth-sidecar",
        str(job["truth_sidecar"]),
        "--candidate",
        str(job["candidate"]),
        "--output-path",
        str(root / "scorer" / "diag_cosine_score.json"),
    ]


def _child_env(cpu_threads: int) -> dict[str, str]:
    threads = int(cpu_threads)
    if threads <= 0:
        raise D92E0DHard12V2RunnerError("CPU thread count must be positive")
    result = dict(os.environ)
    for name in CPU_THREAD_ENV_VARS:
        result[name] = str(threads)
    result["CVSRFFI_CPU_THREADS"] = str(threads)
    result["CVSRFFI_CPU_INTEROP_THREADS"] = "1"
    return result


def _fingerprint(path: Path) -> str:
    if not path.is_file():
        return "missing_stderr"
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    message = next((line for line in reversed(lines) if line), "empty_stderr")
    message = re.sub(r"0x[0-9a-fA-F]+", "<hex>", message)
    message = re.sub(r"\b[0-9]+\b", "<n>", message)
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root)
    if output.exists():
        raise D92E0DHard12V2RunnerError("matrix output already exists")
    manifest = build_hard12v2_manifest(
        context_path=args.context_manifest,
        method_lock_path=args.method_lock,
        output_root=output,
        require_package_files=True,
    )
    output.mkdir(parents=True)
    manifest_path = output / "matrix_manifest.json"
    digest = _write_json_new(manifest_path, manifest)
    return {
        "status": "HARD12V2_MATRIX_PREPARED",
        "matrix_manifest": str(manifest_path),
        "matrix_manifest_sha256": digest,
        "job_count": manifest["job_count"],
    }


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.matrix_manifest, args.matrix_manifest_sha256)
    matches = [
        job
        for job in manifest["jobs"]
        if job["outer_key"] == "rx_7_7__seed_713104__k_1__new_20"
        and job["arm_id"] == "D92_FULL"
    ]
    if len(matches) != 1:
        raise D92E0DHard12V2RunnerError("smoke row identity drift")
    output = Path(args.output_root)
    if output.exists():
        raise D92E0DHard12V2RunnerError("smoke output already exists")
    output.mkdir(parents=True)
    prediction_root = output / "diag"
    command = _prediction_command(
        matches[0],
        ground_component_dir=manifest["ground_component_dir"],
        ground_manifest_sha256=manifest["ground_manifest_sha256"],
        device=args.device,
        output_root=prediction_root,
    )
    stdout_path = output / "prediction.stdout.log"
    stderr_path = output / "prediction.stderr.log"
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout:
        with stderr_path.open("x", encoding="utf-8", newline="\n") as stderr:
            completed = subprocess.run(
                command,
                cwd=CODE_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
                env=_child_env(args.cpu_threads),
            )
    if completed.returncode != 0:
        raise D92E0DHard12V2RunnerError("truth-free smoke prediction failed")
    for state in ("before", "after"):
        if not (prediction_root / state / "COMMIT.json").is_file():
            raise D92E0DHard12V2RunnerError("smoke prediction commit missing")
    receipt = {
        "schema": "cvs.phase2.d92_e0d_hard12v2.smoke_receipt.v1",
        "status": "D92_E0D_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "outer_key": matches[0]["outer_key"],
        "arm_id": "D92_FULL",
        "command": command,
        "query_truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
        "truth_open": False,
    }
    _write_json_new(output / "smoke_receipt.json", receipt)
    return receipt


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.matrix_manifest, args.matrix_manifest_sha256)
    if int(args.shard_count) != 8 or int(args.shard_index) not in range(8):
        raise D92E0DHard12V2RunnerError("shard identity drift")
    selected = [
        job
        for job in manifest["jobs"]
        if int(job["planned_shard_index"]) == int(args.shard_index)
    ]
    if not selected:
        raise D92E0DHard12V2RunnerError("selected shard has no jobs")
    output = Path(manifest["output_root"])
    events_path = output / "events" / f"shard_{args.shard_index}.jsonl"
    summary_path = output / "summaries" / f"shard_{args.shard_index}.json"
    if events_path.exists() or summary_path.exists():
        raise D92E0DHard12V2RunnerError("shard evidence already exists")
    completed_jobs: list[str] = []
    failures: list[dict[str, Any]] = []
    fingerprints: Counter[str] = Counter()
    child_env = _child_env(args.cpu_threads)
    systemic_stop = False
    for job in selected:
        job_root = Path(job["output_root"])
        if job_root.exists():
            failure = {"job_id": job["job_id"], "stage": "preflight", "error": "job output exists"}
            failures.append(failure)
            _append_event(events_path, {"timestamp": _now(), "event": "JOB_REFUSED_EXISTING_OUTPUT", **failure})
            continue
        job_root.mkdir(parents=True)
        prediction_command = _prediction_command(
            job,
            ground_component_dir=manifest["ground_component_dir"],
            ground_manifest_sha256=manifest["ground_manifest_sha256"],
            device=args.device,
        )
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_START", "job_id": job["job_id"], "command": prediction_command})
        prediction_stdout = job_root / "prediction.stdout.log"
        prediction_stderr = job_root / "prediction.stderr.log"
        with prediction_stdout.open("x", encoding="utf-8", newline="\n") as stdout:
            with prediction_stderr.open("x", encoding="utf-8", newline="\n") as stderr:
                prediction_result = subprocess.run(
                    prediction_command,
                    cwd=CODE_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                    env=child_env,
                )
        if prediction_result.returncode != 0:
            fingerprint = _fingerprint(prediction_stderr)
            fingerprints[fingerprint] += 1
            failure = {"job_id": job["job_id"], "stage": "pre_prediction", "returncode": prediction_result.returncode, "fingerprint": fingerprint}
            failures.append(failure)
            _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_FAILED", **failure})
            if fingerprints[fingerprint] >= 2:
                systemic_stop = True
                break
            continue
        for state in ("before", "after"):
            if not (job_root / "diag" / state / "COMMIT.json").is_file():
                raise D92E0DHard12V2RunnerError("prediction child returned without commit")
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_COMPLETE", "job_id": job["job_id"]})
        score_command = _score_command(job)
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_SCORE_START", "job_id": job["job_id"], "command": score_command})
        score_stdout = job_root / "score.stdout.log"
        score_stderr = job_root / "score.stderr.log"
        with score_stdout.open("x", encoding="utf-8", newline="\n") as stdout:
            with score_stderr.open("x", encoding="utf-8", newline="\n") as stderr:
                score_result = subprocess.run(
                    score_command,
                    cwd=CODE_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                    env=child_env,
                )
        score_path = job_root / "scorer" / "diag_cosine_score.json"
        if score_result.returncode != 0 or not score_path.is_file():
            failure = {"job_id": job["job_id"], "stage": "score", "returncode": score_result.returncode}
            failures.append(failure)
            _append_event(events_path, {"timestamp": _now(), "event": "JOB_SCORE_FAILED", **failure})
            continue
        before_prediction = job_root / "diag" / "before" / "prediction_artifact.npz"
        after_prediction = job_root / "diag" / "after" / "prediction_artifact.npz"
        receipt = {
            "schema": "cvs.phase2.d92_e0d_hard12v2.job_receipt.v1",
            "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
            "job_id": job["job_id"],
            "outer_key": job["outer_key"],
            "arm_id": job["arm_id"],
            "candidate": job["candidate"],
            "matrix_manifest_sha256": str(args.matrix_manifest_sha256).lower(),
            "method_lock_sha256": manifest.get("method_lock_sha256"),
            "selection_sha256": manifest["selection_sha256"],
            "context_sha256": manifest["context_sha256"],
            "prediction_command": prediction_command,
            "score_command": score_command,
            "before_prediction_sha256": _sha256_file(before_prediction),
            "after_prediction_sha256": _sha256_file(after_prediction),
            "score_sha256": _sha256_file(score_path),
            "truth_sidecar_exposed_to_predictor": False,
            "query_truth_joined_only_after_immutable_predictions": True,
            "query_truth_fed_back_to_predictor": False,
            "prediction_and_scorer_processes_isolated": True,
            "fresh_run_retry_authorized": False,
        }
        _write_json_new(job_root / "job_receipt.json", receipt)
        completed_jobs.append(str(job["job_id"]))
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_SCORE_COMPLETE", "job_id": job["job_id"]})
    status = (
        "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
        if systemic_stop
        else ("PASS" if not failures and len(completed_jobs) == len(selected) else "PARTIAL_FAILURE")
    )
    summary = {
        "schema": "cvs.phase2.d92_e0d_hard12v2.shard_summary.v1",
        "status": status,
        "shard_index": int(args.shard_index),
        "selected_job_count": len(selected),
        "completed_job_count": len(completed_jobs),
        "failed_job_count": len(failures),
        "completed_job_ids": completed_jobs,
        "failures": failures,
        "performance_result_allowed": status == "PASS",
        "fresh_run_retry_authorized": False,
    }
    _write_json_new(summary_path, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--context-manifest", required=True)
    prepare_parser.add_argument("--method-lock", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--matrix-manifest", required=True)
    smoke_parser.add_argument("--matrix-manifest-sha256", required=True)
    smoke_parser.add_argument("--output-root", required=True)
    smoke_parser.add_argument("--device", required=True)
    smoke_parser.add_argument("--cpu-threads", type=int, default=2)
    shard_parser = commands.add_parser("run-shard")
    shard_parser.add_argument("--matrix-manifest", required=True)
    shard_parser.add_argument("--matrix-manifest-sha256", required=True)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--shard-count", type=int, choices=(8,), default=8)
    shard_parser.add_argument("--device", required=True)
    shard_parser.add_argument("--cpu-threads", type=int, default=2)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "smoke":
        result = smoke(args)
    else:
        result = run_shard(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] in {
        "HARD12V2_MATRIX_PREPARED",
        "D92_E0D_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
