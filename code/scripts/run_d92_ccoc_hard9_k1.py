#!/usr/bin/env python3
"""Prepare, smoke-test, and execute the frozen CCOC Hard9+K1 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from cvsrffi.stage2_d92_ccoc_hard9_k1 import (  # noqa: E402
    ARM_ID,
    CANONICAL_SELECTION_SHA256,
    CANDIDATE_ID,
    FIT_GATE,
    JOB_RECEIPT_SCHEMA,
    MATRIX_SCHEMA,
    QUERY_ZERO_FIELDS,
    RESOURCE_GATE,
    SCENES,
    SHARD_COUNT,
    SHARD_SUMMARY_SCHEMA,
    SMOKE_OUTER_KEY,
    SYSTEMIC_FAILURE_SCHEMA,
    D92CCOCHard9K1Error,
    build_hard9_k1_manifest,
    validate_hard9_k1_manifest,
    validate_method_lock,
)

try:
    from scripts import run_d92_e0ocf_hard12v3 as _prediction_support  # noqa: E402
except ImportError:  # pragma: no cover - direct script execution on N607
    import run_d92_e0ocf_hard12v3 as _prediction_support  # type: ignore[no-redef]


PREDICTION_ENTRY = _prediction_support.PREDICTION_ENTRY
SCORING_ENTRY = _prediction_support.SCORING_ENTRY


class D92CCOCHard9K1RunnerError(RuntimeError):
    """Raised when a direct CCOC Hard9+K1 runner boundary drifts."""


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
    """Write one immutable JSON artifact; never repair or replace it later."""

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
    """Append only this shard's execution event stream."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=True, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D92CCOCHard9K1RunnerError(f"{label} is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92CCOCHard9K1RunnerError(f"{label} is invalid: {source}") from error
    if not isinstance(payload, dict):
        raise D92CCOCHard9K1RunnerError(f"{label} must be an object: {source}")
    return payload


def _finite(value: Any, label: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D92CCOCHard9K1RunnerError(f"{label} is not finite")
    result = float(value)
    if not math.isfinite(result) or (lower is not None and result < lower):
        raise D92CCOCHard9K1RunnerError(f"{label} is below bound")
    return result


def _integer(value: Any, label: str, *, lower: int = 0) -> int:
    result = _finite(value, label, lower=float(lower))
    if result != float(int(result)):
        raise D92CCOCHard9K1RunnerError(f"{label} is not an integer")
    return int(result)


def _resource_value(row: Mapping[str, Any], field: str) -> float:
    resource = row.get("after_registration_resource")
    if not isinstance(resource, Mapping):
        raise D92CCOCHard9K1RunnerError("fit audit resource receipt missing")
    return _finite(resource.get(field), field, lower=0.0)


def _query_access_is_zero(row: Mapping[str, Any]) -> bool:
    for field in QUERY_ZERO_FIELDS:
        for candidate in (field, f"d92_e0d_{field}", f"d92_e0d_ccoc_{field}"):
            if row.get(candidate) is not False:
                return False
    return True


def _require_ccoc_lifecycle(row: Mapping[str, Any], *, k_shot: int) -> None:
    active = int(k_shot) > 2
    prefix = "d92_e0d_ccoc_"
    if row.get("after_state_postprocess_mode") is not None:
        raise D92CCOCHard9K1RunnerError("fit audit postprocess drift")
    inventory = row.get("after_actual_component_inventory")
    if not isinstance(inventory, Mapping):
        raise D92CCOCHard9K1RunnerError("fit audit component inventory missing")
    total = _integer(row.get("after_total_component_fit_count"), "fit total")
    actual = _integer(inventory.get("actual_component_fit_count"), "fit actual")
    mode = str(row.get("after_registered_d_mode_effective", ""))
    if active:
        if (total, actual, mode) != (
            FIT_GATE["k_gt_2_total"],
            FIT_GATE["k_gt_2_actual"],
            "ccoc_full",
        ):
            raise D92CCOCHard9K1RunnerError("fit audit CCOC K>2 inventory drift")
        if (
            row.get(prefix + "active") is not True
            or row.get(prefix + "fallback_active") is not False
            or row.get(prefix + "fallback_reason") is not None
            or _integer(row.get(prefix + "candidate_attempt_fit_count"), "candidate fit")
            != 1
            or _integer(
                row.get(prefix + "fallback_reference_fit_count"),
                "fallback reference fit",
            )
            != 0
            or row.get(prefix + "candidate_statistic_receipt_available") is not True
            or row.get(prefix + "paired_e0_codec_state_equal") is not None
            or row.get(prefix + "g0_eligible") is not True
            or row.get(prefix + "g0_block_reason") is not None
            or _integer(row.get(prefix + "query_rows_used"), "query rows") != 0
        ):
            raise D92CCOCHard9K1RunnerError("fit audit CCOC active lifecycle drift")
        return
    if (total, actual, mode) != (
        FIT_GATE["k1_total"],
        FIT_GATE["k1_actual"],
        "d92_full_alias",
    ):
        raise D92CCOCHard9K1RunnerError("fit audit K1 alias inventory drift")
    if (
        row.get(prefix + "active") is not False
        or row.get(prefix + "fallback_active") is not False
        or row.get(prefix + "fallback_reason") != FIT_GATE["k1_alias"]
        or _integer(row.get(prefix + "candidate_attempt_fit_count"), "candidate fit")
        != 0
        or _integer(
            row.get(prefix + "fallback_reference_fit_count"),
            "fallback reference fit",
        )
        != 0
        or row.get(prefix + "candidate_statistic_receipt_available") is not False
        or row.get(prefix + "paired_e0_codec_state_equal") is not None
        or row.get(prefix + "g0_eligible") is not False
        or row.get(prefix + "g0_block_reason") != FIT_GATE["k1_alias"]
        or _integer(row.get(prefix + "query_rows_used"), "query rows") != 0
    ):
        raise D92CCOCHard9K1RunnerError("fit audit K1 alias lifecycle drift")


def _validate_fit_audit(
    path: str | Path,
    *,
    k_shot: int,
    reference_resources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Close CCOC support/resource gates without opening query truth.

    `reference_resources` is a frozen E0 resource-only projection. It carries
    no query labels or scores and is used only after prediction closure.
    """

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D92CCOCHard9K1RunnerError(f"fit audit is missing: {source}")
    try:
        rows = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92CCOCHard9K1RunnerError(f"fit audit is invalid: {source}") from error
    if (
        not isinstance(rows, list)
        or len(rows) != len(SCENES)
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise D92CCOCHard9K1RunnerError("fit audit scene closure drift")
    by_scene = {str(row.get("scenario")): row for row in rows}
    if len(by_scene) != len(SCENES) or set(by_scene) != set(SCENES):
        raise D92CCOCHard9K1RunnerError("fit audit scene identity drift")
    if set(reference_resources) != set(SCENES):
        raise D92CCOCHard9K1RunnerError("reference resource scene identity drift")

    candidate_peaks: list[float] = []
    candidate_walls: list[float] = []
    candidate_ratios: list[float] = []
    for scene in SCENES:
        row = by_scene[scene]
        reference = reference_resources[scene]
        if row.get("arm_id") != ARM_ID or row.get("candidate_id") != CANDIDATE_ID:
            raise D92CCOCHard9K1RunnerError("fit audit arm/candidate identity drift")
        if not _query_access_is_zero(row):
            raise D92CCOCHard9K1RunnerError("fit audit query access is not zero")
        _require_ccoc_lifecycle(row, k_shot=k_shot)
        class_count = _integer(row.get("registered_class_count"), "registered class count", lower=1)
        candidate_macs = _integer(row.get("query_macs"), "candidate query MACs")
        candidate_state = _integer(row.get("after_state_bytes"), "candidate state bytes", lower=1)
        reference_macs = _integer(reference.get("query_macs"), "E0 query MACs")
        reference_state = _integer(reference.get("state_bytes"), "E0 state bytes", lower=1)
        if candidate_macs != class_count * 288 or candidate_macs != reference_macs:
            raise D92CCOCHard9K1RunnerError("fit audit query MAC drift")
        if candidate_state != reference_state:
            raise D92CCOCHard9K1RunnerError("fit audit state drift")
        candidate_wall = _resource_value(row, "registration_wall_time_ns")
        candidate_peak = _resource_value(
            row,
            "registration_incremental_peak_working_set_bytes",
        )
        reference_wall = _finite(
            reference.get("registration_wall_time_ns"),
            "E0 registration wall",
            lower=0.0,
        )
        _finite(
            reference.get("registration_incremental_peak_working_set_bytes"),
            "E0 registration peak",
            lower=0.0,
        )
        if reference_wall <= 0.0:
            raise D92CCOCHard9K1RunnerError("fit audit ratio reference wall is zero")
        candidate_ratio = candidate_wall / reference_wall
        if candidate_wall > float(RESOURCE_GATE["registration_wall_p90_max_ns"]):
            raise D92CCOCHard9K1RunnerError("fit audit wall hard limit drift")
        if candidate_ratio > float(RESOURCE_GATE["registration_wall_ratio_max"]):
            raise D92CCOCHard9K1RunnerError("fit audit ratio hard limit drift")
        # This is deliberately an absolute candidate cap: E0 peak values do
        # not offset or relax the CCOC hard gate.
        if candidate_peak > float(RESOURCE_GATE["candidate_peak_hard_max_bytes"]):
            raise D92CCOCHard9K1RunnerError("fit audit peak hard limit drift")
        candidate_walls.append(candidate_wall)
        candidate_peaks.append(candidate_peak)
        candidate_ratios.append(candidate_ratio)

    p90_index = max(0, math.ceil(0.90 * len(candidate_walls)) - 1)
    return {
        "scene_count": len(SCENES),
        "candidate_wall_p90_ns": sorted(candidate_walls)[p90_index],
        "candidate_ratio_p90": sorted(candidate_ratios)[p90_index],
        "candidate_peak_max_bytes": max(candidate_peaks),
        "candidate_peak_hard_pass": max(candidate_peaks)
        <= float(RESOURCE_GATE["candidate_peak_hard_max_bytes"]),
        "candidate_peak_target_pass": max(candidate_peaks)
        <= float(RESOURCE_GATE["candidate_peak_target_max_bytes"]),
        "candidate_wall_target_pass": sorted(candidate_walls)[p90_index]
        <= float(RESOURCE_GATE["registration_wall_p90_target_max_ns"]),
        "candidate_ratio_target_pass": sorted(candidate_ratios)[p90_index]
        <= float(RESOURCE_GATE["registration_wall_ratio_target_max"]),
    }


def _verify_manifest_artifacts(manifest: Mapping[str, Any]) -> None:
    """Hash-check the four sealed packages and opaque truth sidecar."""

    for job in manifest.get("jobs", []):
        if not isinstance(job, Mapping):
            raise D92CCOCHard9K1RunnerError("manifest job is invalid")
        truth = Path(str(job.get("truth_sidecar")))
        expected_truth = str(job.get("truth_sidecar_sha256", "")).lower()
        if (
            not truth.is_file()
            or truth.is_symlink()
            or len(expected_truth) != 64
            or _sha256_file(truth) != expected_truth
        ):
            raise D92CCOCHard9K1RunnerError("truth sidecar SHA drift")
        packages = job.get("packages")
        if not isinstance(packages, Mapping):
            raise D92CCOCHard9K1RunnerError("package manifest is missing")
        for name, item in packages.items():
            if not isinstance(item, Mapping):
                raise D92CCOCHard9K1RunnerError(f"package entry is invalid: {name}")
            package_root = Path(str(item.get("package_root")))
            seal = Path(str(item.get("detached_seal_path")))
            expected_seal = str(item.get("expected_seal_sha256", "")).lower()
            if (
                not package_root.is_dir()
                or package_root.is_symlink()
                or not seal.is_file()
                or seal.is_symlink()
                or len(expected_seal) != 64
                or _sha256_file(seal) != expected_seal
            ):
                raise D92CCOCHard9K1RunnerError(f"package seal SHA drift: {name}")


def _load_manifest(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    expected = str(expected_sha256).lower()
    if (
        source.is_symlink()
        or not source.is_file()
        or len(expected) != 64
        or _sha256_file(source) != expected
    ):
        raise D92CCOCHard9K1RunnerError("matrix manifest must be immutable regular file")
    payload = _read_json_object(source, label="matrix manifest")
    try:
        lock_path = Path(str(payload["method_lock"])).resolve(strict=True)
        if lock_path.is_symlink() or not lock_path.is_file():
            raise OSError("method lock is not a regular file")
        lock_sha = _sha256_file(lock_path)
        lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
        validate_method_lock(lock)
        validate_hard9_k1_manifest(
            payload,
            expected_method_lock_sha256=lock_sha,
            require_package_hashes=True,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        D92CCOCHard9K1Error,
    ) as error:
        raise D92CCOCHard9K1RunnerError(
            f"Hard9+K1 manifest contract drift: {error}"
        ) from error
    return payload


def _reference_resources_for_outer(
    manifest: Mapping[str, Any],
    outer_key: str,
) -> dict[str, dict[str, Any]]:
    """Return the sealed E0 resource-only baseline for one outer row."""

    lock_path = Path(str(manifest["method_lock"]))
    lock = _read_json_object(lock_path, label="method lock")
    validate_method_lock(lock)
    baseline = lock.get("historical_baseline")
    if not isinstance(baseline, Mapping):
        raise D92CCOCHard9K1RunnerError("historical baseline identity drift")
    rows = baseline.get("e0_resource_rows")
    if not isinstance(rows, Mapping) or outer_key not in rows:
        raise D92CCOCHard9K1RunnerError("E0 resource baseline missing")
    resource = rows[outer_key]
    if not isinstance(resource, Mapping):
        raise D92CCOCHard9K1RunnerError("E0 resource baseline malformed")
    required = {
        "registration_wall_time_ns",
        "registration_incremental_peak_working_set_bytes",
        "query_macs",
        "state_bytes",
    }
    if set(resource) != required:
        raise D92CCOCHard9K1RunnerError("E0 resource baseline field drift")
    return {scene: dict(resource) for scene in SCENES}


def _prediction_command(
    job: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    device: str,
    output_root: str | Path | None = None,
) -> list[str]:
    return _prediction_support._prediction_command(
        job,
        ground_component_dir=str(manifest["ground_component_dir"]),
        ground_manifest_sha256=str(manifest["ground_manifest_sha256"]),
        device=str(device),
        output_root=output_root,
    )


def _score_command(job: Mapping[str, Any]) -> list[str]:
    return _prediction_support._score_command(job)


def _child_env(cpu_threads: int) -> dict[str, str]:
    return _prediction_support._child_env(int(cpu_threads))


def _prediction_closure_paths(root: Path) -> dict[str, Path]:
    return _prediction_support._prediction_closure_paths(root)


def _prediction_closure_status(root: Path) -> tuple[str, str]:
    return _prediction_support._prediction_closure_status(root)


def _fingerprint(path: Path) -> str:
    return _prediction_support._fingerprint(path)


def _normalized_fingerprint(reason: str) -> str:
    return _prediction_support._normalized_fingerprint(reason)


def _shared_stop_path(output_root: Path) -> Path:
    return output_root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json"


def _record_pre_prediction_failure(
    output_root: str | Path,
    job: Mapping[str, Any],
    fingerprint: str,
) -> bool:
    """Record only pre-prediction failures and stop after two outer rows."""

    root = Path(output_root)
    normalized = str(fingerprint).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        normalized = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    outer_key = str(job["outer_key"])
    job_id = str(job["job_id"])
    fingerprint_root = root / "systemic_pre_prediction_failures" / normalized
    record_path = (
        fingerprint_root
        / hashlib.sha256(outer_key.encode("utf-8")).hexdigest()
        / f"{hashlib.sha256(job_id.encode('utf-8')).hexdigest()}.json"
    )
    record = {
        "schema": SYSTEMIC_FAILURE_SCHEMA,
        "status": "PRE_PREDICTION_FAILURE_RECORDED",
        "timestamp": _now(),
        "fingerprint": normalized,
        "outer_key": outer_key,
        "job_id": job_id,
        "arm_id": str(job["arm_id"]),
        "performance_result_allowed": False,
    }
    try:
        _write_json_new(record_path, record)
    except FileExistsError:
        pass
    distinct_outer_count = (
        sum(1 for child in fingerprint_root.iterdir() if child.is_dir())
        if fingerprint_root.is_dir()
        else 0
    )
    if distinct_outer_count >= 2:
        try:
            _write_json_new(
                _shared_stop_path(root),
                {
                    "schema": SYSTEMIC_FAILURE_SCHEMA,
                    "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
                    "timestamp": _now(),
                    "reason": "same_pre_prediction_fingerprint_on_two_distinct_outers",
                    "fingerprint": normalized,
                    "distinct_outer_count": distinct_outer_count,
                    "performance_result_allowed": False,
                    "fresh_run_retry_authorized": False,
                },
            )
        except FileExistsError:
            pass
    return _shared_stop_path(root).is_file()


def _smoke_job(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        job
        for job in manifest["jobs"]
        if job.get("outer_key") == manifest.get("smoke_outer_key")
        and job.get("outer_role") == "performance"
        and int(job.get("k_shot", -1)) > 2
        and job.get("arm_id") == ARM_ID
    ]
    if len(matches) != 1:
        raise D92CCOCHard9K1RunnerError("K>2 CCOC smoke row identity drift")
    return matches[0]


def _validate_shared_smoke(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
    device: str,
) -> None:
    """Require the exact, truth-free K>2 smoke closure before shards run."""

    if int(manifest.get("job_count", -1)) != 10 or len(manifest.get("jobs", [])) != 10:
        raise D92CCOCHard9K1RunnerError("Hard9+K1 matrix identity drift")
    smoke_root = Path(str(manifest["output_root"])).resolve() / "smoke"
    receipt = _read_json_object(smoke_root / "smoke_receipt.json", label="smoke receipt")
    job = _smoke_job(manifest)
    prediction_root = smoke_root / "diag"
    paths = _prediction_closure_paths(prediction_root)
    hashes = {
        "before_prediction_sha256": _sha256_file(paths["before_prediction"]),
        "after_prediction_sha256": _sha256_file(paths["after_prediction"]),
        "before_commit_sha256": _sha256_file(paths["before_commit"]),
        "after_commit_sha256": _sha256_file(paths["after_commit"]),
        "before_fit_audit_sha256": _sha256_file(paths["before_fit_audit"]),
        "after_fit_audit_sha256": _sha256_file(paths["after_fit_audit"]),
    }
    identity = (
        receipt.get("schema") == "cvs.phase2.d92_ccoc_hard9_k1.smoke_receipt.v1"
        and receipt.get("status")
        == "D92_CCOC_HARD9_K1_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
        and str(receipt.get("matrix_manifest_sha256", "")).lower()
        == str(manifest_sha256).lower()
        and receipt.get("selection_sha256") == CANONICAL_SELECTION_SHA256
        and receipt.get("smoke_outer_key") == SMOKE_OUTER_KEY
        and receipt.get("job_id") == job.get("job_id")
        and receipt.get("outer_role") == "performance"
        and receipt.get("arm_id") == ARM_ID
        and receipt.get("candidate") == CANDIDATE_ID
        and int(receipt.get("k_shot", -1)) > 2
        and receipt.get("truth_open") is False
        and receipt.get("query_truth_joined_only_after_immutable_predictions")
        is True
        and receipt.get("prediction_and_scorer_processes_isolated") is True
        and all(receipt.get(field) is False for field in QUERY_ZERO_FIELDS)
        and receipt.get("prediction_closure") == hashes
        and all(receipt.get(field) == value for field, value in hashes.items())
    )
    if not identity:
        raise D92CCOCHard9K1RunnerError("smoke receipt identity/protocol drift")
    expected_command = _prediction_command(
        job,
        manifest=manifest,
        device=device,
        output_root=prediction_root,
    )
    if receipt.get("command") != expected_command:
        raise D92CCOCHard9K1RunnerError("smoke command identity drift")
    if _prediction_closure_status(prediction_root) != ("closed", "closed"):
        raise D92CCOCHard9K1RunnerError("smoke prediction closure drift")
    resource_gate = _validate_fit_audit(
        paths["after_fit_audit"],
        k_shot=int(job["k_shot"]),
        reference_resources=_reference_resources_for_outer(
            manifest,
            str(job["outer_key"]),
        ),
    )
    if receipt.get("fit_audit_resource_gate") != resource_gate:
        raise D92CCOCHard9K1RunnerError("smoke resource gate receipt drift")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_hard9_k1_manifest(args.config, require_package_files=True)
    output_root = Path(str(manifest["output_root"]))
    if output_root.exists() or output_root.is_symlink():
        raise D92CCOCHard9K1RunnerError("matrix output already exists")
    output_root.mkdir(parents=True)
    manifest_path = output_root / "matrix_manifest.json"
    digest = _write_json_new(manifest_path, manifest)
    return {
        "schema": MATRIX_SCHEMA,
        "status": "CCOC_HARD9_K1_MATRIX_PREPARED",
        "matrix_manifest": str(manifest_path),
        "matrix_manifest_sha256": digest,
        "job_count": 10,
        "scene_arm_count": 30,
    }


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.matrix_manifest, args.matrix_manifest_sha256)
    _verify_manifest_artifacts(manifest)
    job = _smoke_job(manifest)
    output_root = Path(str(manifest["output_root"])).resolve() / "smoke"
    if output_root.exists() or output_root.is_symlink():
        raise D92CCOCHard9K1RunnerError("smoke output identity/immutability drift")
    output_root.mkdir(parents=True)
    prediction_root = output_root / "diag"
    command = _prediction_command(
        job,
        manifest=manifest,
        device=str(args.device),
        output_root=prediction_root,
    )
    with (output_root / "prediction.stdout.log").open("x", encoding="utf-8") as stdout, (
        output_root / "prediction.stderr.log"
    ).open("x", encoding="utf-8") as stderr:
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
        _record_pre_prediction_failure(
            Path(str(manifest["output_root"])),
            job,
            _fingerprint(output_root / "prediction.stderr.log"),
        )
        raise D92CCOCHard9K1RunnerError("truth-free smoke prediction failed")
    status, reason = _prediction_closure_status(prediction_root)
    if status != "closed":
        _record_pre_prediction_failure(
            Path(str(manifest["output_root"])),
            job,
            _normalized_fingerprint(reason),
        )
        raise D92CCOCHard9K1RunnerError("truth-free smoke prediction closure failed")
    paths = _prediction_closure_paths(prediction_root)
    resource_gate = _validate_fit_audit(
        paths["after_fit_audit"],
        k_shot=int(job["k_shot"]),
        reference_resources=_reference_resources_for_outer(
            manifest,
            str(job["outer_key"]),
        ),
    )
    hashes = {
        "before_prediction_sha256": _sha256_file(paths["before_prediction"]),
        "after_prediction_sha256": _sha256_file(paths["after_prediction"]),
        "before_commit_sha256": _sha256_file(paths["before_commit"]),
        "after_commit_sha256": _sha256_file(paths["after_commit"]),
        "before_fit_audit_sha256": _sha256_file(paths["before_fit_audit"]),
        "after_fit_audit_sha256": _sha256_file(paths["after_fit_audit"]),
    }
    receipt: dict[str, Any] = {
        "schema": "cvs.phase2.d92_ccoc_hard9_k1.smoke_receipt.v1",
        "status": "D92_CCOC_HARD9_K1_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "matrix_manifest_sha256": str(args.matrix_manifest_sha256).lower(),
        "selection_sha256": manifest["selection_sha256"],
        "smoke_outer_key": SMOKE_OUTER_KEY,
        "job_id": job["job_id"],
        "outer_role": job["outer_role"],
        "arm_id": ARM_ID,
        "candidate": CANDIDATE_ID,
        "k_shot": int(job["k_shot"]),
        "truth_sidecar_sha256": job["truth_sidecar_sha256"],
        "prediction_root": str(prediction_root),
        "command": command,
        **hashes,
        "prediction_closure": hashes,
        "fit_audit_resource_gate": resource_gate,
        "truth_open": False,
        "query_truth_joined_only_after_immutable_predictions": True,
        "prediction_and_scorer_processes_isolated": True,
        **{field: False for field in QUERY_ZERO_FIELDS},
    }
    _write_json_new(output_root / "smoke_receipt.json", receipt)
    return receipt


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.matrix_manifest, args.matrix_manifest_sha256)
    _verify_manifest_artifacts(manifest)
    _validate_shared_smoke(
        manifest,
        manifest_sha256=str(args.matrix_manifest_sha256).lower(),
        device=str(args.device),
    )
    if int(args.shard_count) != SHARD_COUNT or int(args.shard_index) not in range(
        SHARD_COUNT
    ):
        raise D92CCOCHard9K1RunnerError("shard identity drift")
    shard_index = int(args.shard_index)
    selected = [
        job
        for job in manifest["jobs"]
        if int(job["planned_shard_index"]) == shard_index
    ]
    output_root = Path(str(manifest["output_root"]))
    events_path = output_root / "events" / f"shard_{shard_index}.jsonl"
    summary_path = output_root / "summaries" / f"shard_{shard_index}.json"
    if not selected or events_path.exists() or summary_path.exists():
        raise D92CCOCHard9K1RunnerError("shard selection/evidence identity drift")

    completed_jobs: list[str] = []
    failures: list[dict[str, Any]] = []
    for job in selected:
        if _shared_stop_path(output_root).is_file():
            break
        job_root = Path(str(job["output_root"]))
        if job_root.exists() or job_root.is_symlink():
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "pre_prediction",
                    "error": "immutable_job_output_overwrite_risk",
                }
            )
            _record_pre_prediction_failure(
                output_root,
                job,
                "immutable_job_output_overwrite_risk",
            )
            break
        job_root.mkdir(parents=True)
        command = _prediction_command(job, manifest=manifest, device=str(args.device))
        _append_event(
            events_path,
            {
                "timestamp": _now(),
                "event": "JOB_PREDICTION_START",
                "job_id": job["job_id"],
                "command": command,
            },
        )
        with (job_root / "prediction.stdout.log").open("x", encoding="utf-8") as stdout, (
            job_root / "prediction.stderr.log"
        ).open("x", encoding="utf-8") as stderr:
            prediction_result = subprocess.run(
                command,
                cwd=CODE_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
                env=_child_env(args.cpu_threads),
            )
        prediction_root = job_root / "diag"
        closure_status, closure_reason = (
            _prediction_closure_status(prediction_root)
            if prediction_result.returncode == 0
            else ("technical_failure", "prediction_returncode")
        )
        if prediction_result.returncode != 0 or closure_status != "closed":
            fingerprint = (
                _fingerprint(job_root / "prediction.stderr.log")
                if prediction_result.returncode != 0
                else _normalized_fingerprint(closure_reason)
            )
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "pre_prediction",
                    "fingerprint": fingerprint,
                }
            )
            if _record_pre_prediction_failure(output_root, job, fingerprint):
                break
            continue
        try:
            fit_resource_gate = _validate_fit_audit(
                prediction_root / "after" / "fit_audit.json",
                k_shot=int(job["k_shot"]),
                reference_resources=_reference_resources_for_outer(
                    manifest,
                    str(job["outer_key"]),
                ),
            )
        except D92CCOCHard9K1RunnerError as error:
            # A closed prediction exists, so this is a rejected job rather than
            # a pre-prediction systemic-ledger input.
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "post_prediction_protocol_validation",
                    "error": str(error),
                }
            )
            continue
        score_command = _score_command(job)
        with (job_root / "score.stdout.log").open("x", encoding="utf-8") as stdout, (
            job_root / "score.stderr.log"
        ).open("x", encoding="utf-8") as stderr:
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
        if score_result.returncode != 0 or not score_path.is_file() or score_path.is_symlink():
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "score",
                    "returncode": score_result.returncode,
                }
            )
            continue
        paths = _prediction_closure_paths(prediction_root)
        receipt = {
            "schema": JOB_RECEIPT_SCHEMA,
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
            "prediction_command": command,
            "score_command": score_command,
            "before_prediction_sha256": _sha256_file(paths["before_prediction"]),
            "after_prediction_sha256": _sha256_file(paths["after_prediction"]),
            "score_sha256": _sha256_file(score_path),
            "truth_sidecar_sha256": job["truth_sidecar_sha256"],
            "truth_sidecar_exposed_to_predictor": False,
            "query_truth_joined_only_after_immutable_predictions": True,
            "query_truth_fed_back_to_predictor": False,
            "prediction_and_scorer_processes_isolated": True,
            "fit_audit_resource_gate": fit_resource_gate,
            "fresh_run_retry_authorized": False,
        }
        _write_json_new(job_root / "job_receipt.json", receipt)
        completed_jobs.append(str(job["job_id"]))
    stopped = _shared_stop_path(output_root).is_file()
    status = (
        "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
        if stopped
        else (
            "PASS"
            if not failures and len(completed_jobs) == len(selected)
            else "PARTIAL_FAILURE"
        )
    )
    summary = {
        "schema": SHARD_SUMMARY_SCHEMA,
        "status": status,
        "shard_index": shard_index,
        "selected_job_count": len(selected),
        "completed_job_count": len(completed_jobs),
        "failed_job_count": len(failures),
        "completed_job_ids": completed_jobs,
        "failures": failures,
        "performance_result_allowed": status == "PASS",
        "fresh_run_retry_authorized": False,
        "shared_systemic_stop_path": str(_shared_stop_path(output_root)),
    }
    _write_json_new(summary_path, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--matrix-manifest", required=True)
    smoke_parser.add_argument("--matrix-manifest-sha256", required=True)
    smoke_parser.add_argument("--device", required=True)
    smoke_parser.add_argument("--cpu-threads", type=int, default=2)
    shard_parser = commands.add_parser("run-shard")
    shard_parser.add_argument("--matrix-manifest", required=True)
    shard_parser.add_argument("--matrix-manifest-sha256", required=True)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument(
        "--shard-count",
        type=int,
        choices=(SHARD_COUNT,),
        default=SHARD_COUNT,
    )
    shard_parser.add_argument("--device", required=True)
    shard_parser.add_argument("--cpu-threads", type=int, default=2)
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        value = (
            prepare(args)
            if args.command == "prepare"
            else smoke(args)
            if args.command == "smoke"
            else run_shard(args)
        )
    except (D92CCOCHard9K1RunnerError, D92CCOCHard9K1Error, ValueError) as error:
        print(f"D92 CCOC Hard9+K1 failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return 0 if value["status"] in {
        "CCOC_HARD9_K1_MATRIX_PREPARED",
        "D92_CCOC_HARD9_K1_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
