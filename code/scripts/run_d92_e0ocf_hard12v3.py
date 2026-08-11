#!/usr/bin/env python3
"""Prepare, truth-free smoke, and run the frozen D92-E0OCF Hard12-v3 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_e0ocf_hard12 import (  # noqa: E402
    ARM_ORDER,
    CANONICAL_SELECTION_SHA256,
    CONTEXT_SHA256,
    D92E0OCFHard12V3Error,
    PRIMARY_ARM,
    SMOKE_OUTER_KEY,
    build_hard12v3_manifest,
    validate_hard12v3_manifest,
    validate_method_lock,
)


PREDICTION_ENTRY = CODE_ROOT / "scripts" / "run_d92_e0d_prediction.py"
SCORING_ENTRY = CODE_ROOT / "scripts" / "score_d92_be_prediction.py"
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
)
_QUERY_ZERO_FIELDS = (
    "query_truth_access", "query_fit_access", "query_update_access", "query_selection_access",
    "query_role_oracle_access", "query_class_quota_access", "query_global_reassignment",
)
_SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
_PREDICTION_KEYS = {
    "query_tokens",
    "scenarios",
    "predicted_class_handles",
}
_COMMIT_SCHEMA = "cvs.phase2.diag_cosine_exploration_commit.v1"
_COMMIT_KEYS = {
    "schema",
    "members",
    "artifact_root_sha256",
    "execution_receipt_sha256",
    "prediction_artifact_sha256",
}
_COMMIT_MEMBER_NAMES = (
    "execution_receipt.json",
    "fit_audit.json",
    "prediction_artifact.npz",
    "resource_audit.json",
)
_COMMIT_MEMBER_KEYS = {"relative_path", "sha256", "size_bytes"}


class D92E0OCFHard12V3RunnerError(RuntimeError):
    """Raised when the immutable Hard12-v3 runner contract drifts."""


D92E0OCFHard12RunnerError = D92E0OCFHard12V3RunnerError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
    if source.is_symlink() or not source.is_file() or _sha256_file(source) != str(expected_sha256).lower():
        raise D92E0OCFHard12V3RunnerError("matrix manifest must be immutable regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92E0OCFHard12V3RunnerError("matrix manifest JSON drift") from error
    if not isinstance(payload, dict):
        raise D92E0OCFHard12V3RunnerError("matrix manifest must be an object")
    if payload.get("smoke_outer_key") != SMOKE_OUTER_KEY:
        raise D92E0OCFHard12V3RunnerError("matrix manifest smoke_outer_key drift")
    try:
        lock_source = Path(str(payload["method_lock"]))
        if lock_source.is_symlink():
            raise OSError("method lock symlink is forbidden")
        lock_source = lock_source.resolve(strict=True)
        if not lock_source.is_file():
            raise OSError("method lock is not a regular file")
        method_lock_sha256 = _sha256_file(lock_source)
        if method_lock_sha256 != payload.get("method_lock_sha256"):
            raise OSError("method lock SHA mismatch")
        method_lock = json.loads(lock_source.read_text(encoding="utf-8-sig"))
        validate_method_lock(method_lock)
    except (KeyError, OSError, TypeError, ValueError, D92E0OCFHard12V3Error) as error:
        raise D92E0OCFHard12V3RunnerError("method lock identity/content drift") from error
    try:
        validate_hard12v3_manifest(
            payload,
            expected_method_lock_sha256=method_lock_sha256,
            require_package_hashes=True,
        )
    except D92E0OCFHard12V3Error as error:
        raise D92E0OCFHard12V3RunnerError(
            f"matrix manifest contract drift: {error}"
        ) from error
    return payload


def _prediction_command(job: Mapping[str, Any], *, ground_component_dir: str, ground_manifest_sha256: str, device: str, output_root: str | Path | None = None) -> list[str]:
    package = job["packages"]
    output = Path(output_root) if output_root is not None else Path(job["output_root"]) / "diag"
    command = [sys.executable, str(PREDICTION_ENTRY)]
    for state in ("before", "after"):
        for phase in ("enrollment", "apply"):
            item = package[f"{state}_{phase}"]
            prefix = f"--{state}-{phase}"
            if item.get("expected_seal_sha256") is None:
                raise D92E0OCFHard12V3RunnerError("package seal SHA was not materialized")
            command.extend([f"{prefix}-package-root", str(item["package_root"]), f"{prefix}-seal-path", str(item["detached_seal_path"]), f"{prefix}-seal-sha256", str(item["expected_seal_sha256"])])
    command.extend(["--ground-component-dir", str(ground_component_dir), "--ground-manifest-sha256", str(ground_manifest_sha256), "--arm", str(job["arm_id"]), "--output-root", str(output), "--device", str(device)])
    return command


def _score_command(job: Mapping[str, Any]) -> list[str]:
    root = Path(job["output_root"])
    return [sys.executable, str(SCORING_ENTRY), "--before-prediction", str(root / "diag" / "before" / "prediction_artifact.npz"), "--after-prediction", str(root / "diag" / "after" / "prediction_artifact.npz"), "--truth-sidecar", str(job["truth_sidecar"]), "--candidate", str(job["candidate"]), "--output-path", str(root / "scorer" / "diag_cosine_score.json")]


def _child_env(cpu_threads: int) -> dict[str, str]:
    threads = int(cpu_threads)
    if threads <= 0:
        raise D92E0OCFHard12V3RunnerError("CPU thread count must be positive")
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
    text = None
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    text = text if text is not None else raw.decode("utf-8", errors="replace")
    message = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "empty_stderr")
    message = re.sub(r"0x[0-9a-fA-F]+", "<hex>", message)
    message = re.sub(r"\b[0-9]+\b", "<n>", message)
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _normalized_fingerprint(reason: str) -> str:
    return hashlib.sha256(str(reason).strip().lower().encode("utf-8")).hexdigest()


def _prediction_closure_paths(prediction_root: Path) -> dict[str, Path]:
    root = Path(prediction_root)
    return {
        "before_prediction": root / "before" / "prediction_artifact.npz",
        "after_prediction": root / "after" / "prediction_artifact.npz",
        "before_commit": root / "before" / "COMMIT.json",
        "after_commit": root / "after" / "COMMIT.json",
        "before_fit_audit": root / "before" / "fit_audit.json",
        "after_fit_audit": root / "after" / "fit_audit.json",
    }


def _fit_audit_status(path: Path) -> tuple[str, str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        return "technical_failure", "fit_audit_missing_or_empty"
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "technical_failure", "fit_audit_invalid_json"
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, Mapping) for row in rows
    ):
        return "technical_failure", "fit_audit_invalid_rows"
    for row in rows:
        for field in _QUERY_ZERO_FIELDS:
            if row.get(field) is not False:
                return "protocol_p0", f"fit_audit_{field}_not_false"
    if len(rows) != len(_SCENES):
        return "technical_failure", "fit_audit_invalid_rows"
    scenarios = [row.get("scenario") for row in rows]
    if (
        any(not isinstance(scenario, str) for scenario in scenarios)
        or len(set(scenarios)) != len(_SCENES)
        or set(scenarios) != set(_SCENES)
    ):
        return "technical_failure", "fit_audit_scenario_identity_drift"
    return "closed", "closed"


def _prediction_artifact_status(
    path: Path,
) -> tuple[str, str, tuple[tuple[str, str], ...] | None]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        return "technical_failure", "prediction_artifact_missing_or_empty", None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if (
                len(payload.files) != len(_PREDICTION_KEYS)
                or set(payload.files) != _PREDICTION_KEYS
            ):
                return "technical_failure", "prediction_artifact_key_drift", None
            query_tokens = np.asarray(payload["query_tokens"])
            scenarios = np.asarray(payload["scenarios"])
            predictions = np.asarray(payload["predicted_class_handles"])
    except (OSError, TypeError, ValueError, KeyError):
        return "technical_failure", "prediction_artifact_invalid_npz", None
    arrays = (query_tokens, scenarios, predictions)
    if (
        any(array.ndim != 1 or array.size == 0 for array in arrays)
        or len({int(array.size) for array in arrays}) != 1
    ):
        return "technical_failure", "prediction_artifact_shape_drift", None
    query_values = [str(value) for value in query_tokens.tolist()]
    scenario_values = [str(value) for value in scenarios.tolist()]
    if set(scenario_values) != set(_SCENES) or any(
        scenario_values.count(scene) <= 0 for scene in _SCENES
    ):
        return "technical_failure", "prediction_artifact_scenario_drift", None
    query_pairs = tuple(zip(scenario_values, query_values))
    if len(set(query_pairs)) != len(query_pairs):
        return "technical_failure", "prediction_artifact_query_pair_duplicate", None
    return "closed", "closed", query_pairs


def _commit_status(state_root: Path) -> tuple[str, str]:
    commit_path = state_root / "COMMIT.json"
    if (
        not commit_path.is_file()
        or commit_path.is_symlink()
        or commit_path.stat().st_size <= 0
    ):
        return "technical_failure", "commit_missing_or_empty"
    try:
        commit = json.loads(commit_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "technical_failure", "commit_invalid_json"
    if (
        not isinstance(commit, Mapping)
        or set(commit) != _COMMIT_KEYS
        or commit.get("schema") != _COMMIT_SCHEMA
    ):
        return "technical_failure", "commit_schema_or_key_drift"
    members = commit.get("members")
    if not isinstance(members, list) or len(members) != len(_COMMIT_MEMBER_NAMES):
        return "technical_failure", "commit_member_count_drift"
    if any(
        not isinstance(member, Mapping) or set(member) != _COMMIT_MEMBER_KEYS
        for member in members
    ):
        return "technical_failure", "commit_member_schema_drift"
    if tuple(member.get("relative_path") for member in members) != _COMMIT_MEMBER_NAMES:
        return "technical_failure", "commit_member_name_drift"
    for member in members:
        member_path = state_root / str(member["relative_path"])
        expected_sha256 = member.get("sha256")
        expected_size = member.get("size_bytes")
        if (
            not member_path.is_file()
            or member_path.is_symlink()
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or type(expected_size) is not int
            or expected_size <= 0
            or member_path.stat().st_size != expected_size
            or _sha256_file(member_path) != expected_sha256
        ):
            return "technical_failure", "commit_member_binding_drift"
    by_name = {str(member["relative_path"]): member for member in members}
    try:
        artifact_root_sha256 = hashlib.sha256(
            _canonical_json_bytes(members)
        ).hexdigest()
    except (TypeError, ValueError):
        return "technical_failure", "commit_artifact_root_invalid"
    if (
        commit.get("artifact_root_sha256") != artifact_root_sha256
        or commit.get("execution_receipt_sha256")
        != by_name["execution_receipt.json"]["sha256"]
        or commit.get("prediction_artifact_sha256")
        != by_name["prediction_artifact.npz"]["sha256"]
    ):
        return "technical_failure", "commit_root_or_receipt_binding_drift"
    return "closed", "closed"


def _prediction_closure_status(prediction_root: Path) -> tuple[str, str]:
    paths = _prediction_closure_paths(prediction_root)
    prediction_identity: dict[str, tuple[tuple[str, str], ...]] = {}
    for state in ("before", "after"):
        status, reason, query_pairs = _prediction_artifact_status(
            paths[f"{state}_prediction"]
        )
        if status != "closed" or query_pairs is None:
            return status, f"prediction_closure_{state}_{reason}"
        prediction_identity[state] = query_pairs
        status, reason = _commit_status(Path(prediction_root) / state)
        if status != "closed":
            return status, f"prediction_closure_{state}_{reason}"
        status, reason = _fit_audit_status(paths[f"{state}_fit_audit"])
        if status != "closed":
            return status, f"prediction_closure_{state}_{reason}"
    before_pairs = set(prediction_identity["before"])
    after_pairs = set(prediction_identity["after"])
    before_token_scenarios: dict[str, set[str]] = {}
    after_token_scenarios: dict[str, set[str]] = {}
    for scenario, token in before_pairs:
        before_token_scenarios.setdefault(token, set()).add(scenario)
    for scenario, token in after_pairs:
        after_token_scenarios.setdefault(token, set()).add(scenario)
    shared_tokens = set(before_token_scenarios) & set(after_token_scenarios)
    if not before_pairs.issubset(after_pairs) or any(
        before_token_scenarios[token] != after_token_scenarios[token]
        for token in shared_tokens
    ):
        return "technical_failure", "prediction_closure_cross_state_query_identity_drift"
    return "closed", "closed"


def _shared_systemic_stop_path(output_root: Path) -> Path:
    return Path(output_root) / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json"


def _publish_shared_systemic_stop(output_root: Path, *, reason: str, fingerprint: str, distinct_outer_count: int) -> None:
    try:
        _write_json_new(_shared_systemic_stop_path(output_root), {"schema": "cvs.phase2.d92_e0ocf_hard12v3.systemic_stop.v1", "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE", "timestamp": _now(), "reason": str(reason), "fingerprint": str(fingerprint), "distinct_outer_count": int(distinct_outer_count), "performance_result_allowed": False, "fresh_run_retry_authorized": False})
    except FileExistsError:
        pass


def _record_shared_pre_prediction_failure(output_root: Path, job: Mapping[str, Any], fingerprint: str) -> bool:
    root = Path(output_root)
    normalized = str(fingerprint).lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        normalized = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    outer_key = str(job["outer_key"])
    job_id = str(job["job_id"])
    fingerprint_root = root / "systemic_pre_prediction_failures" / normalized
    record = fingerprint_root / hashlib.sha256(outer_key.encode("utf-8")).hexdigest() / f"{hashlib.sha256(job_id.encode('utf-8')).hexdigest()}.json"
    try:
        _write_json_new(record, {"schema": "cvs.phase2.d92_e0ocf_hard12v3.pre_prediction_failure.v1", "timestamp": _now(), "fingerprint": normalized, "outer_key": outer_key, "job_id": job_id, "arm_id": str(job["arm_id"])})
    except FileExistsError:
        pass
    distinct_outer_count = sum(1 for child in fingerprint_root.iterdir() if child.is_dir())
    if distinct_outer_count >= 2:
        _publish_shared_systemic_stop(root, reason="same_pre_prediction_fingerprint_on_two_distinct_outers", fingerprint=normalized, distinct_outer_count=distinct_outer_count)
    return _shared_systemic_stop_path(root).is_file()


def _fit_audit_protocol_closed(path: Path) -> bool:
    return _fit_audit_status(path)[0] == "closed"


def _smoke_receipt_path(output_root: Path) -> Path | None:
    for candidate in (output_root / "smoke" / "smoke_receipt.json", output_root / "smoke_receipt.json"):
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return None


def _is_full_matrix(manifest: Mapping[str, Any]) -> bool:
    jobs = manifest.get("jobs")
    return int(manifest.get("job_count", -1)) == 60 and isinstance(jobs, list) and len(jobs) == 60


def _smoke_job(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    smoke_outer_key = manifest.get("smoke_outer_key")
    matches = [
        job for job in manifest["jobs"]
        if (smoke_outer_key is None or job.get("outer_key") == smoke_outer_key)
        and job.get("outer_role") == "liveness"
        and int(job.get("k_shot", -1)) == 1
        and job.get("arm_id") == "D92_FULL"
    ]
    if len(matches) != 1:
        raise D92E0OCFHard12V3RunnerError("K1 liveness smoke row identity drift")
    return matches[0]


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92E0OCFHard12V3RunnerError(f"shared smoke artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92E0OCFHard12V3RunnerError(f"shared smoke artifact is invalid: {path}") from error
    if not isinstance(payload, dict):
        raise D92E0OCFHard12V3RunnerError(f"shared smoke artifact must be an object: {path}")
    return payload


def _validate_shared_smoke(manifest: Mapping[str, Any], *, manifest_sha256: str, device: str) -> None:
    """Validate the immutable full-matrix smoke contract before shard dispatch."""

    if not _is_full_matrix(manifest):
        return
    matrix_root = Path(str(manifest["output_root"])).resolve()
    smoke_root = matrix_root / "smoke"
    receipt_path = smoke_root / "smoke_receipt.json"
    receipt = _read_json_object(receipt_path)
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
        receipt.get("schema") == "cvs.phase2.d92_e0ocf_hard12v3.smoke_receipt.v1"
        and receipt.get("status") == "D92_E0OCF_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
        and str(receipt.get("matrix_manifest_sha256", "")).lower() == str(manifest_sha256).lower()
        and receipt.get("selection_sha256") == manifest.get("selection_sha256") == CANONICAL_SELECTION_SHA256
        and receipt.get("smoke_outer_key") == manifest.get("smoke_outer_key", SMOKE_OUTER_KEY)
        and receipt.get("outer_index") == job.get("outer_index")
        and receipt.get("outer_key") == job.get("outer_key")
        and receipt.get("job_id") == job.get("job_id")
        and receipt.get("arm_id") == "D92_FULL"
        and int(receipt.get("k_shot", -1)) == 1
        and receipt.get("outer_role") == "liveness"
        and receipt.get("truth_open") is False
        and all(receipt.get(field) is False for field in _QUERY_ZERO_FIELDS)
    )
    if not identity_ok:
        raise D92E0OCFHard12V3RunnerError("shared smoke receipt identity/protocol drift")
    expected_command = _prediction_command(
        job,
        ground_component_dir=str(manifest["ground_component_dir"]),
        ground_manifest_sha256=str(manifest["ground_manifest_sha256"]),
        device=str(device),
        output_root=expected_prediction_root,
    )
    if receipt.get("command") != expected_command:
        raise D92E0OCFHard12V3RunnerError("shared smoke command identity drift")
    try:
        if Path(str(receipt.get("prediction_root"))).resolve() != expected_prediction_root:
            raise D92E0OCFHard12V3RunnerError("shared smoke prediction root drift")
    except (OSError, TypeError, ValueError) as error:
        raise D92E0OCFHard12V3RunnerError("shared smoke prediction root drift") from error
    closure = receipt.get("prediction_closure")
    if not isinstance(closure, Mapping):
        raise D92E0OCFHard12V3RunnerError("shared smoke prediction closure missing")
    for field, path in expected_files.items():
        if not path.is_file() or path.is_symlink():
            raise D92E0OCFHard12V3RunnerError(f"shared smoke prediction closure missing: {path}")
        digest = _sha256_file(path)
        if receipt.get(field) != digest or closure.get(field) != digest:
            raise D92E0OCFHard12V3RunnerError("shared smoke prediction hash drift")
    if (
        receipt.get("fit_audit_protocol_closed") is not True
        or _prediction_closure_status(expected_prediction_root)[0] != "closed"
    ):
        raise D92E0OCFHard12V3RunnerError("shared smoke fit audit protocol closure drift")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root)
    if output.exists():
        raise D92E0OCFHard12V3RunnerError("matrix output already exists")
    manifest = build_hard12v3_manifest(context_path=args.context_manifest, method_lock_path=args.method_lock, output_root=output, require_package_files=True)
    output.mkdir(parents=True)
    path = output / "matrix_manifest.json"
    digest = _write_json_new(path, manifest)
    return {"status": "HARD12V3_MATRIX_PREPARED", "matrix_manifest": str(path), "matrix_manifest_sha256": digest, "job_count": manifest["job_count"]}


def truth_free_smoke(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.matrix_manifest, args.matrix_manifest_sha256)
    job = _smoke_job(manifest)
    if _is_full_matrix(manifest):
        output = Path(str(manifest["output_root"])).resolve() / "smoke"
        if Path(args.output_root).resolve() != output.resolve():
            raise D92E0OCFHard12V3RunnerError("full-matrix smoke output must be manifest output_root/smoke")
    else:
        output = Path(args.output_root)
    if output.exists():
        raise D92E0OCFHard12V3RunnerError("smoke output already exists")
    output.mkdir(parents=True)
    prediction_root = output / "diag"
    command = _prediction_command(job, ground_component_dir=manifest["ground_component_dir"], ground_manifest_sha256=manifest["ground_manifest_sha256"], device=args.device, output_root=prediction_root)
    stdout_path, stderr_path = output / "prediction.stdout.log", output / "prediction.stderr.log"
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open("x", encoding="utf-8", newline="\n") as stderr:
        completed = subprocess.run(command, cwd=CODE_ROOT, stdout=stdout, stderr=stderr, text=True, check=False, env=_child_env(args.cpu_threads))
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
    if completed.returncode != 0:
        _record_shared_pre_prediction_failure(
            Path(str(manifest["output_root"])),
            job,
            _fingerprint(stderr_path),
        )
        raise D92E0OCFHard12V3RunnerError("truth-free smoke prediction closure failed")
    closure_status, closure_reason = _prediction_closure_status(prediction_root)
    if closure_status == "protocol_p0":
        _publish_shared_systemic_stop(
            Path(str(manifest["output_root"])),
            reason="query_audit_protocol_violation",
            fingerprint=_normalized_fingerprint(closure_reason),
            distinct_outer_count=1,
        )
        raise D92E0OCFHard12V3RunnerError("truth-free smoke query protocol closure failed")
    if closure_status != "closed":
        _record_shared_pre_prediction_failure(
            Path(str(manifest["output_root"])),
            job,
            _normalized_fingerprint(closure_reason),
        )
        raise D92E0OCFHard12V3RunnerError("truth-free smoke prediction closure failed")
    hashes = {field: _sha256_file(path) for field, path in prediction_paths.items()}
    receipt = {
        "schema": "cvs.phase2.d92_e0ocf_hard12v3.smoke_receipt.v1",
        "status": "D92_E0OCF_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "matrix_manifest_sha256": str(args.matrix_manifest_sha256).lower(),
        "selection_sha256": manifest["selection_sha256"],
        "smoke_outer_key": manifest.get("smoke_outer_key", SMOKE_OUTER_KEY),
        "outer_index": job["outer_index"],
        "outer_key": job["outer_key"],
        "job_id": job["job_id"],
        "outer_role": job["outer_role"],
        "arm_id": "D92_FULL",
        "k_shot": int(job["k_shot"]),
        "prediction_root": str(prediction_root),
        "command": command,
        **hashes,
        "prediction_closure": hashes,
        "fit_audit_protocol_closed": True,
        "query_truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
        "query_role_oracle_access": False,
        "query_class_quota_access": False,
        "query_global_reassignment": False,
        "truth_open": False,
    }
    _write_json_new(output / "smoke_receipt.json", receipt)
    return receipt


smoke = truth_free_smoke


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.matrix_manifest, args.matrix_manifest_sha256)
    _validate_shared_smoke(manifest, manifest_sha256=str(args.matrix_manifest_sha256).lower(), device=str(args.device))
    if int(args.shard_count) != 8 or int(args.shard_index) not in range(8):
        raise D92E0OCFHard12V3RunnerError("shard identity drift")
    selected = [job for job in manifest["jobs"] if int(job["planned_shard_index"]) == int(args.shard_index)]
    if not selected:
        raise D92E0OCFHard12V3RunnerError("selected shard has no jobs")
    output = Path(manifest["output_root"])
    events_path, summary_path = output / "events" / f"shard_{args.shard_index}.jsonl", output / "summaries" / f"shard_{args.shard_index}.json"
    if events_path.exists() or summary_path.exists():
        raise D92E0OCFHard12V3RunnerError("shard evidence already exists")
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
        prediction_command = _prediction_command(job, ground_component_dir=manifest["ground_component_dir"], ground_manifest_sha256=manifest["ground_manifest_sha256"], device=args.device)
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_START", "job_id": job["job_id"], "command": prediction_command})
        prediction_stdout, prediction_stderr = job_root / "prediction.stdout.log", job_root / "prediction.stderr.log"
        with prediction_stdout.open("x", encoding="utf-8", newline="\n") as stdout, prediction_stderr.open("x", encoding="utf-8", newline="\n") as stderr:
            prediction_result = subprocess.run(prediction_command, cwd=CODE_ROOT, stdout=stdout, stderr=stderr, text=True, check=False, env=_child_env(args.cpu_threads))
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
            failure = {
                "job_id": job["job_id"],
                "stage": "pre_prediction",
                "returncode": prediction_result.returncode,
                "error": closure_reason,
                "fingerprint": fingerprint,
            }
            failures.append(failure)
            _append_event(
                events_path,
                {"timestamp": _now(), "event": "JOB_PREDICTION_CLOSURE_FAILED", **failure},
            )
            if _record_shared_pre_prediction_failure(output, job, fingerprint):
                systemic_stop = True
                break
            continue
        if closure_status == "protocol_p0":
            fingerprint = _normalized_fingerprint(closure_reason)
            failure = {
                "job_id": job["job_id"],
                "stage": "protocol_p0",
                "returncode": prediction_result.returncode,
                "error": closure_reason,
                "fingerprint": fingerprint,
            }
            failures.append(failure)
            _append_event(
                events_path,
                {"timestamp": _now(), "event": "JOB_QUERY_PROTOCOL_P0", **failure},
            )
            _publish_shared_systemic_stop(
                output,
                reason="query_audit_protocol_violation",
                fingerprint=fingerprint,
                distinct_outer_count=1,
            )
            systemic_stop = True
            break
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_PREDICTION_COMPLETE", "job_id": job["job_id"]})
        score_command = _score_command(job)
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_SCORE_START", "job_id": job["job_id"], "command": score_command})
        score_stdout, score_stderr = job_root / "score.stdout.log", job_root / "score.stderr.log"
        with score_stdout.open("x", encoding="utf-8", newline="\n") as stdout, score_stderr.open("x", encoding="utf-8", newline="\n") as stderr:
            score_result = subprocess.run(score_command, cwd=CODE_ROOT, stdout=stdout, stderr=stderr, text=True, check=False, env=_child_env(args.cpu_threads))
        score_path = job_root / "scorer" / "diag_cosine_score.json"
        if score_result.returncode != 0 or not score_path.is_file():
            failure = {"job_id": job["job_id"], "stage": "score", "returncode": score_result.returncode}
            failures.append(failure)
            _append_event(events_path, {"timestamp": _now(), "event": "JOB_SCORE_FAILED", **failure})
            continue
        before_prediction, after_prediction = job_root / "diag" / "before" / "prediction_artifact.npz", job_root / "diag" / "after" / "prediction_artifact.npz"
        receipt = {"schema": "cvs.phase2.d92_e0ocf_hard12v3.job_receipt.v1", "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE", "job_id": job["job_id"], "outer_key": job["outer_key"], "outer_role": job["outer_role"], "k_shot": job["k_shot"], "arm_id": job["arm_id"], "candidate": job["candidate"], "role": job.get("role"), "matrix_manifest_sha256": str(args.matrix_manifest_sha256).lower(), "method_lock_sha256": manifest.get("method_lock_sha256"), "selection_sha256": manifest["selection_sha256"], "prediction_command": prediction_command, "score_command": score_command, "before_prediction_sha256": _sha256_file(before_prediction), "after_prediction_sha256": _sha256_file(after_prediction), "score_sha256": _sha256_file(score_path), "truth_sidecar_exposed_to_predictor": False, "query_truth_joined_only_after_immutable_predictions": True, "query_truth_fed_back_to_predictor": False, "prediction_and_scorer_processes_isolated": True, "fresh_run_retry_authorized": False}
        _write_json_new(job_root / "job_receipt.json", receipt)
        completed_jobs.append(str(job["job_id"]))
        _append_event(events_path, {"timestamp": _now(), "event": "JOB_SCORE_COMPLETE", "job_id": job["job_id"]})
    systemic_stop = systemic_stop or _shared_systemic_stop_path(output).is_file()
    status = "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE" if systemic_stop else ("PASS" if not failures and len(completed_jobs) == len(selected) else "PARTIAL_FAILURE")
    summary = {"schema": "cvs.phase2.d92_e0ocf_hard12v3.shard_summary.v1", "status": status, "shard_index": int(args.shard_index), "selected_job_count": len(selected), "completed_job_count": len(completed_jobs), "failed_job_count": len(failures), "completed_job_ids": completed_jobs, "failures": failures, "performance_result_allowed": status == "PASS", "fresh_run_retry_authorized": False, "shared_systemic_stop_path": str(_shared_systemic_stop_path(output))}
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
    shard_parser.add_argument("--shard-count", type=int, choices=(8,), default=8)
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
    return 0 if result["status"] in {"HARD12V3_MATRIX_PREPARED", "D92_E0OCF_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
