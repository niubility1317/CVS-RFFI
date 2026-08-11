#!/usr/bin/env python3
"""Prepare, smoke-test and execute the frozen D92 floor-boost Hard11 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_floorboost_hard11 import (  # noqa: E402
    ARM_ID,
    CANDIDATE_ID,
    CANONICAL_SELECTION_SHA256,
    D92FloorboostHard11Error,
    SHARD_COUNT,
    SMOKE_OUTER_KEY,
    build_hard11_manifest,
    validate_hard11_manifest,
    validate_method_lock,
)

try:  # Reuse the established prediction closure and truth-side commands.
    from scripts import run_d92_e0ocf_hard12v3 as _proven_runner  # noqa: E402
except ImportError:  # direct ``python scripts/foo.py`` execution
    import run_d92_e0ocf_hard12v3 as _proven_runner  # type: ignore[no-redef]  # noqa: E402


PREDICTION_ENTRY = _proven_runner.PREDICTION_ENTRY
SCORING_ENTRY = _proven_runner.SCORING_ENTRY
QUERY_ZERO_FIELDS = tuple(_proven_runner._QUERY_ZERO_FIELDS)
_prediction_command = _proven_runner._prediction_command
_score_command = _proven_runner._score_command
_prediction_closure_paths = _proven_runner._prediction_closure_paths
_prediction_closure_status = _proven_runner._prediction_closure_status
_child_env = _proven_runner._child_env
_fingerprint = _proven_runner._fingerprint
_normalized_fingerprint = _proven_runner._normalized_fingerprint


class D92FloorboostHard11RunnerError(RuntimeError):
    """Raised when the immutable Hard11 runner boundary would drift."""


D92FloorboostHard11ErrorAlias = D92FloorboostHard11RunnerError
D92FloorboostHard11Error = D92FloorboostHard11RunnerError


def _is_full_matrix(manifest: Mapping[str, Any]) -> bool:
    """Compatibility predicate retained for closure-oriented callers."""

    return int(manifest.get("job_count", -1)) == 11 and isinstance(manifest.get("jobs"), list) and len(manifest["jobs"]) == 11


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


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92FloorboostHard11RunnerError(f"shared smoke artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92FloorboostHard11RunnerError(f"shared smoke artifact is invalid: {path}") from error
    if not isinstance(payload, dict):
        raise D92FloorboostHard11RunnerError(f"shared smoke artifact must be an object: {path}")
    return payload


def _load_manifest(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file() or _sha256_file(source) != str(expected_sha256).lower():
        raise D92FloorboostHard11RunnerError("matrix manifest must be an immutable regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92FloorboostHard11RunnerError("matrix manifest JSON drift") from error
    if not isinstance(payload, dict):
        raise D92FloorboostHard11RunnerError("matrix manifest must be an object")
    if payload.get("smoke_outer_key") != SMOKE_OUTER_KEY:
        raise D92FloorboostHard11RunnerError("matrix manifest smoke_outer_key drift")
    try:
        lock_source = Path(str(payload["method_lock"])).resolve(strict=True)
        if lock_source.is_symlink() or not lock_source.is_file():
            raise OSError("method lock is not an immutable regular file")
        lock_sha = _sha256_file(lock_source)
        if lock_sha != payload.get("method_lock_sha256"):
            raise OSError("method lock SHA mismatch")
        lock = json.loads(lock_source.read_text(encoding="utf-8-sig"))
        validate_method_lock(lock)
        validate_hard11_manifest(
            payload,
            expected_method_lock_sha256=lock_sha,
            require_package_hashes=True,
        )
    except (KeyError, OSError, TypeError, ValueError, D92FloorboostHard11Error) as error:
        raise D92FloorboostHard11RunnerError(f"Hard11 manifest contract drift: {error}") from error
    return payload


def _shared_systemic_stop_path(output_root: Path) -> Path:
    return Path(output_root) / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json"


def _publish_shared_systemic_stop(
    output_root: Path,
    *,
    reason: str,
    fingerprint: str,
    distinct_outer_count: int,
) -> None:
    try:
        _write_json_new(
            _shared_systemic_stop_path(output_root),
            {
                "schema": "cvs.phase2.d92_floorboost_hard11.systemic_stop.v1",
                "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
                "timestamp": _now(),
                "reason": str(reason),
                "fingerprint": str(fingerprint),
                "distinct_outer_count": int(distinct_outer_count),
                "performance_result_allowed": False,
                "fresh_run_retry_authorized": False,
            },
        )
    except FileExistsError:
        pass


def _record_shared_pre_prediction_failure(
    output_root: Path, job: Mapping[str, Any], fingerprint: str
) -> bool:
    root = Path(output_root)
    normalized = str(fingerprint).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        normalized = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    outer_key = str(job["outer_key"])
    job_id = str(job["job_id"])
    fingerprint_root = root / "systemic_pre_prediction_failures" / normalized
    record = fingerprint_root / hashlib.sha256(outer_key.encode("utf-8")).hexdigest() / (
        f"{hashlib.sha256(job_id.encode('utf-8')).hexdigest()}.json"
    )
    try:
        _write_json_new(
            record,
            {
                "schema": "cvs.phase2.d92_floorboost_hard11.pre_prediction_failure.v1",
                "timestamp": _now(),
                "fingerprint": normalized,
                "outer_key": outer_key,
                "job_id": job_id,
                "arm_id": str(job["arm_id"]),
            },
        )
    except FileExistsError:
        pass
    distinct_outer_count = sum(1 for child in fingerprint_root.iterdir() if child.is_dir())
    if distinct_outer_count >= 2:
        _publish_shared_systemic_stop(
            root,
            reason="same_pre_prediction_fingerprint_on_two_distinct_outers",
            fingerprint=normalized,
            distinct_outer_count=distinct_outer_count,
        )
    return _shared_systemic_stop_path(root).is_file()


def _smoke_job(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        job
        for job in manifest["jobs"]
        if job.get("outer_key") == manifest.get("smoke_outer_key")
        and job.get("outer_role") == "liveness"
        and int(job.get("k_shot", -1)) == 1
        and job.get("arm_id") == ARM_ID
    ]
    if len(matches) != 1:
        raise D92FloorboostHard11RunnerError("K1 liveness smoke row identity drift")
    return matches[0]


def _validate_shared_smoke(
    manifest: Mapping[str, Any], *, manifest_sha256: str, device: str
) -> None:
    if int(manifest.get("job_count", -1)) != 11 or len(manifest.get("jobs", [])) != 11:
        raise D92FloorboostHard11RunnerError("Hard11 matrix identity drift")
    matrix_root = Path(str(manifest["output_root"])).resolve()
    smoke_root = matrix_root / "smoke"
    receipt = _read_json_object(smoke_root / "smoke_receipt.json")
    job = _smoke_job(manifest)
    expected_prediction_root = smoke_root / "diag"
    closure_paths = _prediction_closure_paths(expected_prediction_root)
    expected_files = {
        "before_prediction_sha256": closure_paths["before_prediction"],
        "after_prediction_sha256": closure_paths["after_prediction"],
        "before_commit_sha256": closure_paths["before_commit"],
        "after_commit_sha256": closure_paths["after_commit"],
        "before_fit_audit_sha256": closure_paths["before_fit_audit"],
        "after_fit_audit_sha256": closure_paths["after_fit_audit"],
        "fit_audit_sha256": closure_paths["after_fit_audit"],
    }
    identity_ok = (
        receipt.get("schema") == "cvs.phase2.d92_floorboost_hard11.smoke_receipt.v1"
        and receipt.get("status") == "D92_FLOORBOOST_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
        and str(receipt.get("matrix_manifest_sha256", "")).lower() == str(manifest_sha256).lower()
        and receipt.get("selection_sha256") == manifest.get("selection_sha256") == CANONICAL_SELECTION_SHA256
        and receipt.get("smoke_outer_key") == SMOKE_OUTER_KEY
        and receipt.get("outer_index") == job.get("outer_index")
        and receipt.get("outer_key") == job.get("outer_key")
        and receipt.get("job_id") == job.get("job_id")
        and receipt.get("arm_id") == ARM_ID
        and receipt.get("candidate") == CANDIDATE_ID
        and int(receipt.get("k_shot", -1)) == 1
        and receipt.get("outer_role") == "liveness"
        and receipt.get("truth_open") is False
        and all(receipt.get(field) is False for field in QUERY_ZERO_FIELDS)
    )
    if not identity_ok:
        raise D92FloorboostHard11RunnerError("shared smoke receipt identity/protocol drift")
    expected_command = _prediction_command(
        job,
        ground_component_dir=str(manifest["ground_component_dir"]),
        ground_manifest_sha256=str(manifest["ground_manifest_sha256"]),
        device=str(device),
        output_root=expected_prediction_root,
    )
    if receipt.get("command") != expected_command:
        raise D92FloorboostHard11RunnerError("shared smoke command identity drift")
    try:
        if Path(str(receipt.get("prediction_root"))).resolve() != expected_prediction_root:
            raise D92FloorboostHard11RunnerError("shared smoke prediction root drift")
    except (OSError, TypeError, ValueError) as error:
        raise D92FloorboostHard11RunnerError("shared smoke prediction root drift") from error
    closure = receipt.get("prediction_closure")
    if not isinstance(closure, Mapping):
        raise D92FloorboostHard11RunnerError("shared smoke prediction closure missing")
    for field, path in expected_files.items():
        if not path.is_file() or path.is_symlink():
            raise D92FloorboostHard11RunnerError(f"shared smoke prediction closure missing: {path}")
        digest = _sha256_file(path)
        if receipt.get(field) != digest or closure.get(field) != digest:
            raise D92FloorboostHard11RunnerError("shared smoke prediction hash drift")
    if receipt.get("fit_audit_protocol_closed") is not True or _prediction_closure_status(expected_prediction_root)[0] != "closed":
        raise D92FloorboostHard11RunnerError("shared smoke fit audit protocol closure drift")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root)
    if output.exists():
        raise D92FloorboostHard11RunnerError("matrix output already exists")
    manifest = build_hard11_manifest(
        context_path=args.context_manifest,
        method_lock_path=args.method_lock,
        output_root=output,
        require_package_files=True,
    )
    output.mkdir(parents=True)
    manifest_path = output / "matrix_manifest.json"
    digest = _write_json_new(manifest_path, manifest)
    return {
        "status": "FLOORBOOST_HARD11_MATRIX_PREPARED",
        "matrix_manifest": str(manifest_path),
        "matrix_manifest_sha256": digest,
        "job_count": manifest["job_count"],
        "scene_arm_count": manifest["scene_arm_count"],
    }


def truth_free_smoke(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.matrix_manifest, args.matrix_manifest_sha256)
    job = _smoke_job(manifest)
    output = Path(str(manifest["output_root"])).resolve() / "smoke"
    if Path(args.output_root).resolve() != output:
        raise D92FloorboostHard11RunnerError("Hard11 smoke output must be manifest output_root/smoke")
    if output.exists():
        raise D92FloorboostHard11RunnerError("smoke output already exists")
    output.mkdir(parents=True)
    prediction_root = output / "diag"
    command = _prediction_command(
        job,
        ground_component_dir=manifest["ground_component_dir"],
        ground_manifest_sha256=manifest["ground_manifest_sha256"],
        device=args.device,
        output_root=prediction_root,
    )
    stdout_path, stderr_path = output / "prediction.stdout.log", output / "prediction.stderr.log"
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open("x", encoding="utf-8", newline="\n") as stderr:
        completed = subprocess.run(
            command,
            cwd=CODE_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
            env=_child_env(args.cpu_threads),
        )
    closure_paths = _prediction_closure_paths(prediction_root)
    prediction_paths = {
        "before_prediction_sha256": closure_paths["before_prediction"],
        "after_prediction_sha256": closure_paths["after_prediction"],
        "before_commit_sha256": closure_paths["before_commit"],
        "after_commit_sha256": closure_paths["after_commit"],
        "before_fit_audit_sha256": closure_paths["before_fit_audit"],
        "after_fit_audit_sha256": closure_paths["after_fit_audit"],
        "fit_audit_sha256": closure_paths["after_fit_audit"],
    }
    matrix_root = Path(str(manifest["output_root"]))
    if completed.returncode != 0:
        _record_shared_pre_prediction_failure(matrix_root, job, _fingerprint(stderr_path))
        raise D92FloorboostHard11RunnerError("truth-free smoke prediction closure failed")
    closure_status, closure_reason = _prediction_closure_status(prediction_root)
    if closure_status == "protocol_p0":
        _publish_shared_systemic_stop(
            matrix_root,
            reason="query_audit_protocol_violation",
            fingerprint=_normalized_fingerprint(closure_reason),
            distinct_outer_count=1,
        )
        raise D92FloorboostHard11RunnerError("truth-free smoke query protocol closure failed")
    if closure_status != "closed":
        _record_shared_pre_prediction_failure(matrix_root, job, _normalized_fingerprint(closure_reason))
        raise D92FloorboostHard11RunnerError("truth-free smoke prediction closure failed")
    hashes = {field: _sha256_file(path) for field, path in prediction_paths.items()}
    receipt = {
        "schema": "cvs.phase2.d92_floorboost_hard11.smoke_receipt.v1",
        "status": "D92_FLOORBOOST_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "matrix_manifest_sha256": str(args.matrix_manifest_sha256).lower(),
        "selection_sha256": manifest["selection_sha256"],
        "smoke_outer_key": SMOKE_OUTER_KEY,
        "outer_index": job["outer_index"],
        "outer_key": job["outer_key"],
        "job_id": job["job_id"],
        "outer_role": job["outer_role"],
        "arm_id": ARM_ID,
        "candidate": CANDIDATE_ID,
        "k_shot": int(job["k_shot"]),
        "prediction_root": str(prediction_root),
        "command": command,
        **hashes,
        "prediction_closure": hashes,
        "fit_audit_protocol_closed": True,
        **{field: False for field in QUERY_ZERO_FIELDS},
        "truth_open": False,
    }
    _write_json_new(output / "smoke_receipt.json", receipt)
    return receipt


smoke = truth_free_smoke


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.matrix_manifest, args.matrix_manifest_sha256)
    _validate_shared_smoke(
        manifest,
        manifest_sha256=str(args.matrix_manifest_sha256).lower(),
        device=str(args.device),
    )
    if int(args.shard_count) != SHARD_COUNT or int(args.shard_index) not in range(SHARD_COUNT):
        raise D92FloorboostHard11RunnerError("shard identity drift")
    selected = [job for job in manifest["jobs"] if int(job["planned_shard_index"]) == int(args.shard_index)]
    if not selected:
        raise D92FloorboostHard11RunnerError("selected shard has no jobs")
    output = Path(manifest["output_root"])
    events_path = output / "events" / f"shard_{args.shard_index}.jsonl"
    summary_path = output / "summaries" / f"shard_{args.shard_index}.json"
    if events_path.exists() or summary_path.exists():
        raise D92FloorboostHard11RunnerError("shard evidence already exists")
    completed_jobs: list[str] = []
    failures: list[dict[str, Any]] = []
    systemic_stop = False
    for job in selected:
        if _shared_systemic_stop_path(output).is_file():
            systemic_stop = True
            break
        job_root = Path(job["output_root"])
        if job_root.exists():
            failure = {"job_id": job["job_id"], "stage": "preflight", "error": "job output exists"}
            failures.append(failure)
            _append_event(events_path, {"timestamp": _now(), "event": "JOB_REFUSED_EXISTING_OUTPUT", **failure})
            _publish_shared_systemic_stop(output, reason="immutable_job_output_overwrite_risk", fingerprint="job_output_exists", distinct_outer_count=1)
            systemic_stop = True
            break
        job_root.mkdir(parents=True)
        prediction_command = _prediction_command(
            job,
            ground_component_dir=manifest["ground_component_dir"],
            ground_manifest_sha256=manifest["ground_manifest_sha256"],
            device=args.device,
        )
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_START", "job_id": job["job_id"], "command": prediction_command})
        prediction_stdout, prediction_stderr = job_root / "prediction.stdout.log", job_root / "prediction.stderr.log"
        with prediction_stdout.open("x", encoding="utf-8", newline="\n") as stdout, prediction_stderr.open("x", encoding="utf-8", newline="\n") as stderr:
            prediction_result = subprocess.run(
                prediction_command,
                cwd=CODE_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
                env=_child_env(args.cpu_threads),
            )
        if prediction_result.returncode != 0:
            fingerprint = _fingerprint(prediction_stderr)
            failure = {"job_id": job["job_id"], "stage": "pre_prediction", "returncode": prediction_result.returncode, "fingerprint": fingerprint}
            failures.append(failure)
            _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_FAILED", **failure})
            if _record_shared_pre_prediction_failure(output, job, fingerprint):
                systemic_stop = True
                break
            continue
        closure_status, closure_reason = _prediction_closure_status(job_root / "diag")
        if closure_status == "technical_failure":
            fingerprint = _normalized_fingerprint(closure_reason)
            failure = {"job_id": job["job_id"], "stage": "pre_prediction", "returncode": prediction_result.returncode, "error": closure_reason, "fingerprint": fingerprint}
            failures.append(failure)
            _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_CLOSURE_FAILED", **failure})
            if _record_shared_pre_prediction_failure(output, job, fingerprint):
                systemic_stop = True
                break
            continue
        if closure_status == "protocol_p0":
            fingerprint = _normalized_fingerprint(closure_reason)
            failure = {"job_id": job["job_id"], "stage": "protocol_p0", "returncode": prediction_result.returncode, "error": closure_reason, "fingerprint": fingerprint}
            failures.append(failure)
            _append_event(events_path, {"timestamp": _now(), "event": "JOB_QUERY_PROTOCOL_P0", **failure})
            _publish_shared_systemic_stop(output, reason="query_audit_protocol_violation", fingerprint=fingerprint, distinct_outer_count=1)
            systemic_stop = True
            break
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_COMPLETE", "job_id": job["job_id"]})
        score_command = _score_command(job)
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_SCORE_START", "job_id": job["job_id"], "command": score_command})
        score_stdout, score_stderr = job_root / "score.stdout.log", job_root / "score.stderr.log"
        with score_stdout.open("x", encoding="utf-8", newline="\n") as stdout, score_stderr.open("x", encoding="utf-8", newline="\n") as stderr:
            score_result = subprocess.run(
                score_command,
                cwd=CODE_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
                env=_child_env(args.cpu_threads),
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
            "schema": "cvs.phase2.d92_floorboost_hard11.job_receipt.v1",
            "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
            "job_id": job["job_id"],
            "outer_key": job["outer_key"],
            "outer_role": job["outer_role"],
            "k_shot": job["k_shot"],
            "arm_id": ARM_ID,
            "candidate": CANDIDATE_ID,
            "role": "primary",
            "matrix_manifest_sha256": str(args.matrix_manifest_sha256).lower(),
            "method_lock_sha256": manifest["method_lock_sha256"],
            "selection_sha256": manifest["selection_sha256"],
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
    systemic_stop = systemic_stop or _shared_systemic_stop_path(output).is_file()
    status = "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE" if systemic_stop else ("PASS" if not failures and len(completed_jobs) == len(selected) else "PARTIAL_FAILURE")
    summary = {
        "schema": "cvs.phase2.d92_floorboost_hard11.shard_summary.v1",
        "status": status,
        "shard_index": int(args.shard_index),
        "selected_job_count": len(selected),
        "completed_job_count": len(completed_jobs),
        "failed_job_count": len(failures),
        "completed_job_ids": completed_jobs,
        "failures": failures,
        "performance_result_allowed": status == "PASS",
        "fresh_run_retry_authorized": False,
        "shared_systemic_stop_path": str(_shared_systemic_stop_path(output)),
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
    smoke_parser = commands.add_parser("truth-free-smoke", aliases=["smoke"])
    smoke_parser.add_argument("--matrix-manifest", required=True)
    smoke_parser.add_argument("--matrix-manifest-sha256", required=True)
    smoke_parser.add_argument("--output-root", required=True)
    smoke_parser.add_argument("--device", required=True)
    smoke_parser.add_argument("--cpu-threads", type=int, default=2)
    shard_parser = commands.add_parser("run-shard")
    shard_parser.add_argument("--matrix-manifest", required=True)
    shard_parser.add_argument("--matrix-manifest-sha256", required=True)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--shard-count", type=int, choices=(SHARD_COUNT,), default=SHARD_COUNT)
    shard_parser.add_argument("--device", required=True)
    shard_parser.add_argument("--cpu-threads", type=int, default=2)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command in {"truth-free-smoke", "smoke"}:
        result = truth_free_smoke(args)
    else:
        result = run_shard(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] in {
        "FLOORBOOST_HARD11_MATRIX_PREPARED",
        "D92_FLOORBOOST_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
