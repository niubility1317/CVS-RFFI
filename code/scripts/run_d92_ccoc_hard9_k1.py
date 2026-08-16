#!/usr/bin/env python3
"""Prepare, smoke-test, and execute the frozen CCOC Hard9+K1 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


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
_SCORE_ARTIFACT_SCHEMA = "cvs.phase2.diag_cosine_dev_pair_score.v1"
_SCORE_BINDING_SCHEMA = "cvs.phase2.d92_ccoc_hard9_k1.score_binding.v1"
_ACTIVE_PROCESS_SCHEMA = "cvs.phase2.d92_ccoc_hard9_k1.active_process.v1"
_ACTIVE_PROCESS_ACTION_SCHEMA = "cvs.phase2.d92_ccoc_hard9_k1.stop_action.v1"


_PREDICTION_CLOSURE_SHA_FIELDS = (
    "before_prediction_sha256",
    "after_prediction_sha256",
    "before_commit_sha256",
    "after_commit_sha256",
    "before_fit_audit_sha256",
    "after_fit_audit_sha256",
    "before_resource_audit_sha256",
    "after_resource_audit_sha256",
    "before_execution_receipt_sha256",
    "after_execution_receipt_sha256",
)


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


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
    """Hash-check predictor packages while leaving truth opaque until scoring."""

    for job in manifest.get("jobs", []):
        if not isinstance(job, Mapping):
            raise D92CCOCHard9K1RunnerError("manifest job is invalid")
        truth = job.get("truth_sidecar")
        expected_truth = str(job.get("truth_sidecar_sha256", "")).lower()
        if (
            not isinstance(truth, str)
            or not truth
            or "\x00" in truth
            or not _is_sha256(expected_truth)
            or expected_truth == "0" * 64
        ):
            raise D92CCOCHard9K1RunnerError("truth sidecar metadata drift")
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


def _verify_runtime_source_files(
    runtime_source: Mapping[str, Any],
    *,
    code_root: str | Path = CODE_ROOT,
    git_runner: Callable[[Path, tuple[str, ...]], str] | None = None,
) -> dict[str, Any]:
    """Fail closed if any locked scientific execution source has drifted."""

    if not isinstance(runtime_source, Mapping) or set(runtime_source) != {
        "scientific_entry_commit",
        "files",
    }:
        raise D92CCOCHard9K1RunnerError("runtime source lock schema drift")
    commit = str(runtime_source.get("scientific_entry_commit", "")).lower()
    files = runtime_source.get("files")
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(files, Mapping)
        or not files
    ):
        raise D92CCOCHard9K1RunnerError("runtime source lock identity drift")
    root = Path(code_root).resolve(strict=True)
    repo_root = root.parent
    repo_prefix = root.relative_to(repo_root).as_posix()
    verification_mode = "sha256_only"
    if git_runner is not None:
        verification_mode = "sha256_plus_git"
        try:
            git_runner(repo_root, ("cat-file", "-e", f"{commit}^{{commit}}"))
        except D92CCOCHard9K1RunnerError as error:
            raise D92CCOCHard9K1RunnerError(
                "runtime source frozen commit is unavailable"
            ) from error
    for relative_path, record in files.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith(("/", "\\"))
            or "\\" in relative_path
            or ".." in Path(relative_path).parts
            or not isinstance(record, Mapping)
            or set(record) != {"git_blob", "sha256"}
            or not _is_sha256(record.get("sha256"))
            or not isinstance(record.get("git_blob"), str)
            or len(str(record["git_blob"])) != 40
            or any(
                character not in "0123456789abcdef"
                for character in str(record["git_blob"]).lower()
            )
        ):
            raise D92CCOCHard9K1RunnerError("runtime source record drift")
        source = (root / relative_path).resolve()
        if (
            root not in source.parents
            or not source.is_file()
            or source.is_symlink()
            or _sha256_file(source) != str(record["sha256"]).lower()
        ):
            raise D92CCOCHard9K1RunnerError(
                f"runtime source SHA drift: {relative_path}"
            )
        if git_runner is not None:
            repository_path = f"{repo_prefix}/{relative_path}"
            frozen_blob = git_runner(
                repo_root,
                ("rev-parse", f"{commit}:{repository_path}"),
            )
            if str(frozen_blob).strip().lower() != str(record["git_blob"]).lower():
                raise D92CCOCHard9K1RunnerError(
                    f"runtime source frozen blob drift: {relative_path}"
                )
            head_blob = git_runner(
                repo_root,
                ("rev-parse", f"HEAD:{repository_path}"),
            )
            if str(head_blob).strip().lower() != str(record["git_blob"]).lower():
                raise D92CCOCHard9K1RunnerError(
                    f"runtime source HEAD blob drift: {relative_path}"
                )
    return {
        "scientific_entry_commit": commit,
        "repository_root": str(repo_root),
        "file_count": len(files),
        "verification_mode": verification_mode,
    }


def _verify_runtime_source_lock(
    lock: Mapping[str, Any],
    *,
    code_root: str | Path = CODE_ROOT,
    git_runner: Callable[[Path, tuple[str, ...]], str] | None = None,
) -> dict[str, Any]:
    if not isinstance(lock, Mapping):
        raise D92CCOCHard9K1RunnerError("runtime source method-lock drift")
    return _verify_runtime_source_files(
        lock.get("runtime_source", {}),
        code_root=code_root,
        git_runner=git_runner,
    )


def _verify_truth_sidecar_snapshot(
    path: str | Path,
    *,
    expected_sha256: str,
) -> str:
    """Return a verified, read-only truth-sidecar SHA at one score boundary."""

    source = Path(path)
    expected = str(expected_sha256).lower()
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        or not _is_sha256(expected)
        or _sha256_file(source) != expected
    ):
        raise D92CCOCHard9K1RunnerError("truth sidecar snapshot/hash drift")
    return expected


def _prediction_closure_hashes(
    before_prediction_path: str | Path,
    after_prediction_path: str | Path,
) -> dict[str, str]:
    """Hash every sealed prediction-closure member for an outer receipt."""

    before_prediction = Path(before_prediction_path)
    after_prediction = Path(after_prediction_path)
    if (
        before_prediction.name != "prediction_artifact.npz"
        or after_prediction.name != "prediction_artifact.npz"
        or before_prediction.parent.name != "before"
        or after_prediction.parent.name != "after"
        or before_prediction.parent.parent != after_prediction.parent.parent
    ):
        raise D92CCOCHard9K1RunnerError("prediction closure hash path drift")
    evidence_paths = {
        "before_prediction_sha256": before_prediction,
        "after_prediction_sha256": after_prediction,
        "before_commit_sha256": before_prediction.with_name("COMMIT.json"),
        "after_commit_sha256": after_prediction.with_name("COMMIT.json"),
        "before_fit_audit_sha256": before_prediction.with_name("fit_audit.json"),
        "after_fit_audit_sha256": after_prediction.with_name("fit_audit.json"),
        "before_resource_audit_sha256": before_prediction.with_name(
            "resource_audit.json"
        ),
        "after_resource_audit_sha256": after_prediction.with_name(
            "resource_audit.json"
        ),
        "before_execution_receipt_sha256": before_prediction.with_name(
            "execution_receipt.json"
        ),
        "after_execution_receipt_sha256": after_prediction.with_name(
            "execution_receipt.json"
        ),
    }
    if tuple(evidence_paths) != _PREDICTION_CLOSURE_SHA_FIELDS:
        raise D92CCOCHard9K1RunnerError("prediction closure hash schema drift")
    for field, path in evidence_paths.items():
        if not path.is_file() or path.is_symlink():
            raise D92CCOCHard9K1RunnerError(
                f"prediction closure hash artifact drift: {field}"
            )
    return {field: _sha256_file(path) for field, path in evidence_paths.items()}


def _validate_score_artifact(
    path: str | Path,
    *,
    job: Mapping[str, Any],
    matrix_manifest_sha256: str,
    method_lock_sha256: str,
    truth_sidecar_sha256: str,
    before_prediction_path: str | Path,
    after_prediction_path: str | Path,
    score_binding_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind parsed independent-score inputs to one immutable matrix job."""

    score_path = Path(path)
    if not score_path.is_file() or score_path.is_symlink():
        raise D92CCOCHard9K1RunnerError("score artifact is missing")
    score = _read_json_object(score_path, label="score artifact")
    before_path = Path(before_prediction_path)
    after_path = Path(after_prediction_path)
    closure_hashes = _prediction_closure_hashes(before_path, after_path)
    expected_truth = str(truth_sidecar_sha256).lower()
    expected_before = closure_hashes["before_prediction_sha256"]
    expected_after = closure_hashes["after_prediction_sha256"]
    identity = {
        "job_id": str(job.get("job_id", "")),
        "outer_key": str(job.get("outer_key", "")),
        "arm_id": str(job.get("arm_id", "")),
        "candidate": str(job.get("candidate", "")),
        "matrix_manifest_sha256": str(matrix_manifest_sha256).lower(),
        "method_lock_sha256": str(method_lock_sha256).lower(),
    }
    if (
        not all(identity.values())
        or not _is_sha256(identity["matrix_manifest_sha256"])
        or not _is_sha256(identity["method_lock_sha256"])
        or score.get("schema") != _SCORE_ARTIFACT_SCHEMA
        or score.get("candidate") != identity["candidate"]
        or score.get("truth_sidecar_sha256") != expected_truth
        or score.get("before_prediction_sha256") != expected_before
        or score.get("after_prediction_sha256") != expected_after
    ):
        raise D92CCOCHard9K1RunnerError("score artifact input/identity drift")
    if score_binding_path is not None:
        binding = _read_json_object(score_binding_path, label="score binding")
        if (
            binding.get("schema") != _SCORE_BINDING_SCHEMA
            or binding.get("job_id") != identity["job_id"]
            or binding.get("outer_key") != identity["outer_key"]
            or binding.get("arm_id") != identity["arm_id"]
            or binding.get("candidate") != identity["candidate"]
            or binding.get("matrix_manifest_sha256")
            != identity["matrix_manifest_sha256"]
            or binding.get("method_lock_sha256") != identity["method_lock_sha256"]
            or binding.get("truth_sidecar_sha256") != expected_truth
            or binding.get("before_prediction_sha256") != expected_before
            or binding.get("after_prediction_sha256") != expected_after
        ):
            raise D92CCOCHard9K1RunnerError("score binding identity drift")
        if any(
            str(binding.get(field, "")).lower() != expected
            for field, expected in closure_hashes.items()
        ):
            raise D92CCOCHard9K1RunnerError("score binding closure hash drift")
    return {
        **identity,
        "score_artifact_sha256": _sha256_file(score_path),
        "truth_sidecar_sha256": expected_truth,
        **closure_hashes,
    }


def _write_score_binding(
    job_root: Path,
    *,
    job: Mapping[str, Any],
    matrix_manifest_sha256: str,
    method_lock_sha256: str,
    paths: Mapping[str, Path],
    score_command: list[str],
) -> tuple[Path, str]:
    """Seal actual scoring inputs at the boundary immediately before scoring."""

    closure_hashes = _prediction_closure_hashes(
        paths["before_prediction"],
        paths["after_prediction"],
    )
    truth_sha256 = _verify_truth_sidecar_snapshot(
        job["truth_sidecar"],
        expected_sha256=str(job["truth_sidecar_sha256"]),
    )
    binding_path = job_root / "score_binding.json"
    digest = _write_json_new(
        binding_path,
        {
            "schema": _SCORE_BINDING_SCHEMA,
            "timestamp": _now(),
            "job_id": str(job["job_id"]),
            "outer_key": str(job["outer_key"]),
            "outer_role": str(job["outer_role"]),
            "arm_id": str(job["arm_id"]),
            "candidate": str(job["candidate"]),
            "matrix_manifest_sha256": str(matrix_manifest_sha256).lower(),
            "method_lock_sha256": str(method_lock_sha256).lower(),
            "truth_sidecar": str(job["truth_sidecar"]),
            "truth_sidecar_sha256": truth_sha256,
            **closure_hashes,
            "score_command": list(score_command),
            "performance_result_allowed": False,
        },
    )
    return binding_path, digest


def _write_job_receipt(
    job_root: str | Path,
    receipt: Mapping[str, Any],
    *,
    closure_hashes: Mapping[str, str],
) -> str:
    """Write the final immutable receipt with the score-validated closure."""

    root = Path(job_root)
    if root.is_symlink() or not root.is_dir():
        raise D92CCOCHard9K1RunnerError("job receipt root drift")
    if set(closure_hashes) != set(_PREDICTION_CLOSURE_SHA_FIELDS) or any(
        not _is_sha256(closure_hashes.get(field))
        for field in _PREDICTION_CLOSURE_SHA_FIELDS
    ):
        raise D92CCOCHard9K1RunnerError("job receipt closure hash drift")
    normalized_hashes = {
        field: str(closure_hashes[field]).lower()
        for field in _PREDICTION_CLOSURE_SHA_FIELDS
    }
    if any(
        field in receipt
        for field in (*_PREDICTION_CLOSURE_SHA_FIELDS, "prediction_closure")
    ):
        raise D92CCOCHard9K1RunnerError("job receipt closure overwrite risk")
    evidence = receipt.get("score_evidence")
    if not isinstance(evidence, Mapping) or any(
        str(evidence.get(field, "")).lower() != expected
        for field, expected in normalized_hashes.items()
    ):
        raise D92CCOCHard9K1RunnerError("job receipt score evidence drift")
    return _write_json_new(
        root / "job_receipt.json",
        {
            **dict(receipt),
            **normalized_hashes,
            "prediction_closure": normalized_hashes,
        },
    )


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
        _verify_runtime_source_lock(lock)
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


def _load_verified_e0_resource_records(
    job: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Load the same-outer sealed E0 fit audit and index it by scene key.

    The frozen record seals both the whole historical fit-audit bytes and the
    resource-only projection used by each candidate scene.  It intentionally
    never position-zips rows or estimates missing scene measurements.
    """

    outer_key = str(job.get("outer_key", ""))
    try:
        k_shot = _integer(job.get("k_shot"), "E0 resource K", lower=1)
        new_class_count = _integer(
            job.get("new_class_count"), "E0 resource new-class count", lower=1
        )
    except D92CCOCHard9K1RunnerError as error:
        raise D92CCOCHard9K1RunnerError("E0 resource job identity drift") from error
    resource = job.get("e0_resource")
    if not isinstance(resource, Mapping) or set(resource) != {"fit_audit", "scenes"}:
        raise D92CCOCHard9K1RunnerError("E0 resource manifest record drift")
    sealed = resource.get("fit_audit")
    locked_scenes = resource.get("scenes")
    if (
        not isinstance(sealed, Mapping)
        or set(sealed) != {"path", "sha256"}
        or not isinstance(locked_scenes, Mapping)
        or set(locked_scenes) != set(SCENES)
        or not _is_sha256(sealed.get("sha256"))
    ):
        raise D92CCOCHard9K1RunnerError("E0 resource sealed record drift")
    source = Path(str(sealed.get("path", "")))
    if (
        not source.is_file()
        or source.is_symlink()
        or _sha256_file(source) != str(sealed["sha256"]).lower()
    ):
        raise D92CCOCHard9K1RunnerError("E0 resource fit-audit SHA drift")
    try:
        rows = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92CCOCHard9K1RunnerError("E0 resource fit-audit invalid") from error
    if (
        not isinstance(rows, list)
        or len(rows) != len(SCENES)
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise D92CCOCHard9K1RunnerError("E0 resource scene closure drift")
    by_scene = {str(row.get("scenario", "")): row for row in rows}
    if len(by_scene) != len(SCENES) or set(by_scene) != set(SCENES):
        raise D92CCOCHard9K1RunnerError("E0 resource scene identity drift")

    required_fields = {
        "registration_wall_time_ns",
        "registration_incremental_peak_working_set_bytes",
        "query_macs",
        "state_bytes",
    }
    actual: dict[str, dict[str, int]] = {}
    for scene in SCENES:
        row = by_scene[scene]
        if (
            row.get("arm_id") != "E0_FULL_ONLY"
            or row.get("candidate_id") != "d92_e0d_e0_full_only"
            or _integer(row.get("k_shot"), "E0 resource K", lower=1) != k_shot
            or _integer(
                row.get("registered_class_count"),
                "E0 resource registered-class count",
                lower=1,
            )
            != 6 + new_class_count
        ):
            raise D92CCOCHard9K1RunnerError("E0 resource fit-audit identity drift")
        actual_scene = {
            "registration_wall_time_ns": _integer(
                _resource_value(row, "registration_wall_time_ns"),
                "E0 registration wall",
            ),
            "registration_incremental_peak_working_set_bytes": _integer(
                _resource_value(
                    row,
                    "registration_incremental_peak_working_set_bytes",
                ),
                "E0 registration peak",
            ),
            "query_macs": _integer(row.get("query_macs"), "E0 query MACs"),
            "state_bytes": _integer(
                row.get("after_state_bytes"), "E0 state bytes", lower=1
            ),
        }
        locked = locked_scenes[scene]
        if not isinstance(locked, Mapping) or set(locked) != required_fields:
            raise D92CCOCHard9K1RunnerError("E0 resource locked scene field drift")
        expected_scene = {
            field: _integer(locked.get(field), f"E0 locked {field}", lower=0)
            for field in required_fields
        }
        if actual_scene != expected_scene:
            raise D92CCOCHard9K1RunnerError("E0 resource sealed scene value drift")
        actual[scene] = actual_scene
    if not outer_key:
        raise D92CCOCHard9K1RunnerError("E0 resource outer identity drift")
    return actual


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


def _prediction_failure_stage(prediction_root: str | Path) -> str:
    """Classify failed prediction attempts without treating partial output as pre-run."""

    root = Path(prediction_root)
    closure = _prediction_closure_paths(root)
    artifacts = list(closure.values())
    for state in ("before", "after"):
        artifacts.extend(
            (
                root / state / "execution_receipt.json",
                root / state / "resource_audit.json",
            )
        )
    return (
        "post_prediction"
        if any(path.exists() or path.is_symlink() for path in artifacts)
        else "pre_prediction"
    )


def _fingerprint(path: Path) -> str:
    return _prediction_support._fingerprint(path)


def _normalized_fingerprint(reason: str) -> str:
    return _prediction_support._normalized_fingerprint(reason)


def _shared_stop_path(output_root: Path) -> Path:
    return output_root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json"


def _stop_score_lock_path(output_root: Path) -> Path:
    return output_root / "coordination" / "stop_score_dispatch.lock"


@contextmanager
def _stop_score_dispatch_lock(
    output_root: str | Path,
    *,
    coordinator_id: str,
    purpose: str,
    timeout_seconds: float = 5.0,
):
    """Serialize stop publication and scorer Popen/receipt registration.

    A stale lock deliberately fails closed.  Releasing a lock owned by another
    coordinator is forbidden, so a crash cannot be repaired by a later shard.
    """

    root = Path(output_root).resolve()
    owner_id = str(coordinator_id)
    if not owner_id or purpose not in {
        "prediction_dispatch",
        "score_dispatch",
        "stop_publish",
        "stop_action",
    }:
        raise D92CCOCHard9K1RunnerError("stop/score coordination identity drift")
    timeout = _finite(timeout_seconds, "stop/score coordination timeout", lower=0.0)
    path = _stop_score_lock_path(root)
    owner = {
        "schema": _ACTIVE_PROCESS_ACTION_SCHEMA,
        "status": "STOP_SCORE_DISPATCH_LOCK",
        "timestamp": _now(),
        "coordinator_id": owner_id,
        "purpose": purpose,
        "run_root": str(root),
        "pid": os.getpid(),
        "performance_result_allowed": False,
    }
    deadline = time.monotonic() + timeout
    while True:
        try:
            _write_json_new(path, owner)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise D92CCOCHard9K1RunnerError(
                    "stop/score coordination lock is unavailable"
                )
            time.sleep(0.01)
    try:
        yield
    finally:
        existing = _read_json_object(path, label="stop/score coordination lock")
        if existing != owner:
            raise D92CCOCHard9K1RunnerError(
                "stop/score coordination lock ownership drift"
            )
        try:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
            path.unlink()
        except OSError as error:
            raise D92CCOCHard9K1RunnerError(
                "stop/score coordination lock release failed"
            ) from error


def _dispatch_score_under_stop_barrier(
    output_root: str | Path,
    *,
    coordinator_id: str,
    start: Callable[[], Any],
    before_start: Callable[[], None] | None = None,
) -> tuple[bool, Any | None]:
    """Atomically recheck stop and register a score child before dispatch."""

    root = Path(output_root).resolve()
    with _stop_score_dispatch_lock(
        root,
        coordinator_id=coordinator_id,
        purpose="score_dispatch",
    ):
        if _shared_stop_path(root).is_file():
            return False, None
        if before_start is not None:
            before_start()
        if _shared_stop_path(root).is_file():
            return False, None
        return True, start()


def _dispatch_prediction_under_stop_barrier(
    output_root: str | Path,
    *,
    coordinator_id: str,
    start: Callable[[], Any],
    before_start: Callable[[], None] | None = None,
) -> tuple[bool, Any | None]:
    """Atomically recheck stop and register a prediction child before dispatch."""

    root = Path(output_root).resolve()
    with _stop_score_dispatch_lock(
        root,
        coordinator_id=coordinator_id,
        purpose="prediction_dispatch",
    ):
        if _shared_stop_path(root).is_file():
            return False, None
        if before_start is not None:
            before_start()
        if _shared_stop_path(root).is_file():
            return False, None
        return True, start()


def _start_score_unless_stopped(
    output_root: str | Path,
    *,
    start: Callable[[], Any],
) -> tuple[bool, Any | None]:
    """Compatibility boundary for the direct pre-existing stop test."""

    return _dispatch_score_under_stop_barrier(
        output_root,
        coordinator_id=f"compat-score-dispatch-{os.getpid()}",
        start=start,
    )


def _active_process_path(
    output_root: str | Path,
    *,
    shard_index: int,
    job_id: str,
    stage: str,
) -> Path:
    token = hashlib.sha256(f"{job_id}\x00{stage}".encode("utf-8")).hexdigest()
    return (
        Path(output_root)
        / "active_processes"
        / f"shard_{int(shard_index)}"
        / f"{token}.json"
    )


def _write_active_process_receipt(
    output_root: str | Path,
    *,
    job: Mapping[str, Any],
    shard_index: int,
    stage: str,
    pid: int,
    parent_pid: int,
    cwd: str | Path,
    cmdline: tuple[str, ...] | list[str],
) -> Path:
    """Record one shard-owned child before it can be stopped by a coordinator."""

    root = Path(output_root).resolve()
    job_id = str(job.get("job_id", ""))
    outer_key = str(job.get("outer_key", ""))
    arm_id = str(job.get("arm_id", ""))
    candidate = str(job.get("candidate", ""))
    command = [str(value) for value in cmdline]
    if (
        not job_id
        or not outer_key
        or arm_id != ARM_ID
        or candidate != CANDIDATE_ID
        or stage not in {"prediction", "score"}
        or int(shard_index) not in range(SHARD_COUNT)
        or int(pid) <= 0
        or int(parent_pid) <= 0
        or not command
    ):
        raise D92CCOCHard9K1RunnerError("active-process receipt identity drift")
    path = _active_process_path(
        root,
        shard_index=int(shard_index),
        job_id=job_id,
        stage=stage,
    )
    _write_json_new(
        path,
        {
            "schema": _ACTIVE_PROCESS_SCHEMA,
            "status": "ACTIVE_CHILD_RECORDED",
            "timestamp": _now(),
            "run_root": str(root),
            "job_id": job_id,
            "outer_key": outer_key,
            "arm_id": arm_id,
            "candidate": candidate,
            "shard_index": int(shard_index),
            "stage": stage,
            "pid": int(pid),
            "parent_pid": int(parent_pid),
            "cwd": str(Path(cwd).resolve()),
            "cmdline": command,
            "performance_result_allowed": False,
        },
    )
    return path


def _start_shard_child(
    command: list[str],
    *,
    output_root: str | Path,
    job: Mapping[str, Any],
    shard_index: int,
    stage: str,
    stdout: Any,
    stderr: Any,
    env: Mapping[str, str],
    popen: Callable[..., Any] = subprocess.Popen,
    after_popen_before_receipt: Callable[[], Any] | None = None,
) -> Any:
    """Start one owned child and persist its exclusive stop receipt first."""

    child = popen(
        command,
        cwd=CODE_ROOT,
        stdout=stdout,
        stderr=stderr,
        text=True,
        env=dict(env),
    )
    try:
        if after_popen_before_receipt is not None:
            after_popen_before_receipt()
        _write_active_process_receipt(
            output_root,
            job=job,
            shard_index=shard_index,
            stage=stage,
            pid=int(child.pid),
            parent_pid=os.getpid(),
            cwd=CODE_ROOT,
            cmdline=command,
        )
    except BaseException:
        try:
            child.terminate()
            child.wait()
        except (AttributeError, OSError, subprocess.SubprocessError):
            pass
        raise
    return child


def _wait_shard_child(child: Any) -> int:
    return int(child.wait())


def _run_shard_child(
    command: list[str],
    *,
    output_root: str | Path,
    job: Mapping[str, Any],
    shard_index: int,
    stage: str,
    stdout: Any,
    stderr: Any,
    env: Mapping[str, str],
    popen: Callable[..., Any] = subprocess.Popen,
) -> int:
    """Start and wait for one owned child outside the score-dispatch barrier."""

    return _wait_shard_child(
        _start_shard_child(
            command,
            output_root=output_root,
            job=job,
            shard_index=shard_index,
            stage=stage,
            stdout=stdout,
            stderr=stderr,
            env=env,
            popen=popen,
        )
    )


def _read_active_process_receipt(path: Path, *, output_root: Path) -> dict[str, Any]:
    receipt = _read_json_object(path, label="active-process receipt")
    required = {
        "schema",
        "status",
        "timestamp",
        "run_root",
        "job_id",
        "outer_key",
        "arm_id",
        "candidate",
        "shard_index",
        "stage",
        "pid",
        "parent_pid",
        "cwd",
        "cmdline",
        "performance_result_allowed",
    }
    if (
        set(receipt) != required
        or receipt.get("schema") != _ACTIVE_PROCESS_SCHEMA
        or receipt.get("status") != "ACTIVE_CHILD_RECORDED"
        or Path(str(receipt.get("run_root", ""))).resolve() != output_root.resolve()
        or receipt.get("arm_id") != ARM_ID
        or receipt.get("candidate") != CANDIDATE_ID
        or int(receipt.get("shard_index", -1)) not in range(SHARD_COUNT)
        or receipt.get("stage") not in {"prediction", "score"}
        or _integer(receipt.get("pid"), "active-process PID", lower=1) <= 0
        or _integer(receipt.get("parent_pid"), "active-process parent PID", lower=1)
        <= 0
        or not isinstance(receipt.get("cmdline"), list)
        or not receipt["cmdline"]
        or any(not isinstance(item, str) or not item for item in receipt["cmdline"])
        or receipt.get("performance_result_allowed") is not False
    ):
        raise D92CCOCHard9K1RunnerError("active-process receipt contract drift")
    return receipt


def _iter_active_process_receipts(output_root: Path) -> list[Path]:
    root = output_root / "active_processes"
    if not root.is_dir() or root.is_symlink():
        return []
    paths: list[Path] = []
    for shard_root in sorted(root.iterdir(), key=lambda item: item.name):
        if not shard_root.is_dir() or shard_root.is_symlink():
            continue
        for path in sorted(shard_root.iterdir(), key=lambda item: item.name):
            if path.suffix == ".json" and path.is_file() and not path.is_symlink():
                paths.append(path)
    return paths


def _inspect_posix_process(pid: int) -> dict[str, Any] | None:
    if os.name != "posix":
        return None
    root = Path("/proc") / str(int(pid))
    try:
        status = (root / "status").read_text(encoding="utf-8")
        parent = next(
            int(line.split("\t", 1)[1])
            for line in status.splitlines()
            if line.startswith("PPid:\t")
        )
        return {
            "pid": int(pid),
            "parent_pid": parent,
            "cwd": os.readlink(root / "cwd"),
            "cmdline": [
                item.decode("utf-8", errors="surrogateescape")
                for item in (root / "cmdline").read_bytes().split(b"\x00")
                if item
            ],
        }
    except (OSError, StopIteration, ValueError):
        return None


def _receipt_matches_process(
    receipt: Mapping[str, Any],
    process: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(process, Mapping):
        return False
    try:
        return (
            int(process.get("pid", -1)) == int(receipt["pid"])
            and int(process.get("parent_pid", -1)) == int(receipt["parent_pid"])
            and Path(str(process.get("cwd", ""))).resolve()
            == Path(str(receipt["cwd"])).resolve()
            and list(process.get("cmdline", [])) == list(receipt["cmdline"])
        )
    except (TypeError, ValueError):
        return False


def _send_posix_signal(pid: int, signal_name: str) -> None:
    if os.name != "posix":
        raise D92CCOCHard9K1RunnerError("coordinator stop is unsupported on this host")
    selected = {"SIGTERM": signal.SIGTERM, "SIGKILL": signal.SIGKILL}.get(signal_name)
    if selected is None:
        raise D92CCOCHard9K1RunnerError("coordinator signal identity drift")
    os.kill(int(pid), selected)


def _acquire_stop_coordinator(output_root: Path, coordinator_id: str) -> None:
    path = output_root / "coordination" / "stop_coordinator.json"
    payload = {
        "schema": _ACTIVE_PROCESS_ACTION_SCHEMA,
        "status": "UNIQUE_STOP_COORDINATOR",
        "timestamp": _now(),
        "coordinator_id": coordinator_id,
        "run_root": str(output_root.resolve()),
        "performance_result_allowed": False,
    }
    try:
        _write_json_new(path, payload)
    except FileExistsError:
        existing = _read_json_object(path, label="stop coordinator")
        if (
            set(existing)
            != {
                "schema",
                "status",
                "timestamp",
                "coordinator_id",
                "run_root",
                "performance_result_allowed",
            }
            or existing.get("schema") != _ACTIVE_PROCESS_ACTION_SCHEMA
            or existing.get("status") != "UNIQUE_STOP_COORDINATOR"
            or existing.get("coordinator_id") != coordinator_id
            or existing.get("run_root") != str(output_root.resolve())
            or existing.get("performance_result_allowed") is not False
        ):
            raise D92CCOCHard9K1RunnerError("stop coordinator ownership drift")


def _stop_verified_active_processes_locked(
    root: Path,
    *,
    coordinator_id: str,
    process_inspector: Callable[[int], Mapping[str, Any] | None] | None,
    signal_sender: Callable[[int, str], None] | None,
    sleep_seconds: float,
) -> dict[str, Any]:
    """Stop only verified run-owned children while the barrier is held."""

    _acquire_stop_coordinator(root, str(coordinator_id))
    inspect = process_inspector or _inspect_posix_process
    send = signal_sender or _send_posix_signal
    delay = _finite(sleep_seconds, "stop grace seconds", lower=0.0)
    receipts = [
        _read_active_process_receipt(path, output_root=root)
        for path in _iter_active_process_receipts(root)
    ]
    verified = 0
    skipped = 0
    graceful = 0
    escalated = 0
    for receipt in receipts:
        pid = int(receipt["pid"])
        if not _receipt_matches_process(receipt, inspect(pid)):
            skipped += 1
            continue
        verified += 1
        send(pid, "SIGTERM")
        graceful += 1
        if delay > 0.0:
            time.sleep(delay)
        if _receipt_matches_process(receipt, inspect(pid)):
            send(pid, "SIGKILL")
            escalated += 1
    result = {
        "schema": _ACTIVE_PROCESS_ACTION_SCHEMA,
        "status": "VERIFIED_STOP_COORDINATOR_COMPLETE",
        "timestamp": _now(),
        "run_root": str(root),
        "coordinator_id": str(coordinator_id),
        "active_receipt_count": len(receipts),
        "verified_process_count": verified,
        "skipped_unverified_process_count": skipped,
        "graceful_termination_attempt_count": graceful,
        "escalated_termination_attempt_count": escalated,
        "performance_result_allowed": False,
    }
    try:
        _write_json_new(root / "coordination" / "stop_action.json", result)
    except FileExistsError as error:
        raise D92CCOCHard9K1RunnerError(
            "stop coordinator action already recorded"
        ) from error
    return result


def stop_verified_active_processes(
    output_root: str | Path,
    *,
    coordinator_id: str,
    process_inspector: Callable[[int], Mapping[str, Any] | None] | None = None,
    signal_sender: Callable[[int, str], None] | None = None,
    sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    """Stop only verified run-owned children after a systemic stop receipt exists."""

    root = Path(output_root).resolve()
    if not coordinator_id or not _shared_stop_path(root).is_file():
        raise D92CCOCHard9K1RunnerError("systemic stop/coordinator identity drift")
    with _stop_score_dispatch_lock(
        root,
        coordinator_id=str(coordinator_id),
        purpose="stop_action",
    ):
        return _stop_verified_active_processes_locked(
            root,
            coordinator_id=str(coordinator_id),
            process_inspector=process_inspector,
            signal_sender=signal_sender,
            sleep_seconds=sleep_seconds,
        )


def _publish_systemic_stop_and_terminate(
    output_root: str | Path,
    *,
    coordinator_id: str,
    fingerprint: str,
    distinct_outer_count: int,
    process_inspector: Callable[[int], Mapping[str, Any] | None] | None,
    signal_sender: Callable[[int, str], None] | None,
    sleep_seconds: float,
) -> bool:
    """Atomically publish the stop and terminate only receipt-verified children."""

    root = Path(output_root).resolve()
    with _stop_score_dispatch_lock(
        root,
        coordinator_id=str(coordinator_id),
        purpose="stop_publish",
    ):
        created = False
        try:
            _write_json_new(
                _shared_stop_path(root),
                {
                    "schema": SYSTEMIC_FAILURE_SCHEMA,
                    "status": "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE",
                    "timestamp": _now(),
                    "reason": "same_pre_prediction_fingerprint_on_two_distinct_outers",
                    "fingerprint": str(fingerprint),
                    "distinct_outer_count": int(distinct_outer_count),
                    "performance_result_allowed": False,
                    "fresh_run_retry_authorized": False,
                },
            )
            created = True
        except FileExistsError:
            pass
        if created:
            _stop_verified_active_processes_locked(
                root,
                coordinator_id=str(coordinator_id),
                process_inspector=process_inspector,
                signal_sender=signal_sender,
                sleep_seconds=sleep_seconds,
            )
        return _shared_stop_path(root).is_file()


def _record_pre_prediction_failure(
    output_root: str | Path,
    job: Mapping[str, Any],
    fingerprint: str,
    *,
    coordinator_id: str | None = None,
    process_inspector: Callable[[int], Mapping[str, Any] | None] | None = None,
    signal_sender: Callable[[int, str], None] | None = None,
    sleep_seconds: float = 1.0,
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
        return _publish_systemic_stop_and_terminate(
            root,
            coordinator_id=(
                str(coordinator_id)
                if coordinator_id
                else f"automatic-systemic-stop-{os.getpid()}"
            ),
            fingerprint=normalized,
            distinct_outer_count=distinct_outer_count,
            process_inspector=process_inspector,
            signal_sender=signal_sender,
            sleep_seconds=sleep_seconds,
        )
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
        reference_resources=_load_verified_e0_resource_records(job),
    )
    if receipt.get("fit_audit_resource_gate") != resource_gate:
        raise D92CCOCHard9K1RunnerError("smoke resource gate receipt drift")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_hard9_k1_manifest(args.config, require_package_files=True)
    lock = _read_json_object(manifest["method_lock"], label="method lock")
    validate_method_lock(lock)
    runtime_source_receipt = _verify_runtime_source_lock(lock)
    for job in manifest["jobs"]:
        _load_verified_e0_resource_records(job)
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
        "runtime_source_verification_mode": runtime_source_receipt[
            "verification_mode"
        ],
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
        stage = _prediction_failure_stage(prediction_root)
        if stage == "pre_prediction":
            _record_pre_prediction_failure(
                Path(str(manifest["output_root"])),
                job,
                _fingerprint(output_root / "prediction.stderr.log"),
            )
        raise D92CCOCHard9K1RunnerError(
            f"truth-free smoke {stage.replace('_', '-')} prediction failed"
        )
    status, reason = _prediction_closure_status(prediction_root)
    if status != "closed":
        stage = _prediction_failure_stage(prediction_root)
        if stage == "pre_prediction":
            _record_pre_prediction_failure(
                Path(str(manifest["output_root"])),
                job,
                _normalized_fingerprint(reason),
            )
        raise D92CCOCHard9K1RunnerError(
            f"truth-free smoke {stage.replace('_', '-')} prediction closure failed"
        )
    paths = _prediction_closure_paths(prediction_root)
    resource_gate = _validate_fit_audit(
        paths["after_fit_audit"],
        k_shot=int(job["k_shot"]),
        reference_resources=_load_verified_e0_resource_records(job),
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
    coordinator_id = f"shard-{shard_index}-pid-{os.getpid()}"
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
                coordinator_id=coordinator_id,
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
            prediction_started, prediction_child = (
                _dispatch_prediction_under_stop_barrier(
                    output_root,
                    coordinator_id=coordinator_id,
                    start=lambda: _start_shard_child(
                        command,
                        output_root=output_root,
                        job=job,
                        shard_index=shard_index,
                        stage="prediction",
                        stdout=stdout,
                        stderr=stderr,
                        env=_child_env(args.cpu_threads),
                    ),
                )
            )
        if not prediction_started:
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "systemic_stop_before_prediction",
                }
            )
            break
        if prediction_child is None:  # pragma: no cover - guarded by prediction_started
            raise D92CCOCHard9K1RunnerError("prediction start value drift")
        prediction_returncode = _wait_shard_child(prediction_child)
        prediction_root = job_root / "diag"
        closure_status, closure_reason = (
            _prediction_closure_status(prediction_root)
            if prediction_returncode == 0
            else ("technical_failure", "prediction_returncode")
        )
        if prediction_returncode != 0 or closure_status != "closed":
            fingerprint = (
                _fingerprint(job_root / "prediction.stderr.log")
                if prediction_returncode != 0
                else _normalized_fingerprint(closure_reason)
            )
            failure_stage = _prediction_failure_stage(prediction_root)
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": (
                        "pre_prediction"
                        if failure_stage == "pre_prediction"
                        else "post_prediction_technical_failure"
                    ),
                    "fingerprint": fingerprint,
                }
            )
            if failure_stage == "pre_prediction" and _record_pre_prediction_failure(
                output_root,
                job,
                fingerprint,
                coordinator_id=coordinator_id,
            ):
                break
            continue
        try:
            fit_resource_gate = _validate_fit_audit(
                prediction_root / "after" / "fit_audit.json",
                k_shot=int(job["k_shot"]),
                reference_resources=_load_verified_e0_resource_records(job),
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
        paths = _prediction_closure_paths(prediction_root)
        try:
            with (job_root / "score.stdout.log").open("x", encoding="utf-8") as stdout, (
                job_root / "score.stderr.log"
            ).open("x", encoding="utf-8") as stderr:

                def _start_score() -> tuple[Path, str, Any]:
                    binding_path, binding_sha256 = _write_score_binding(
                        job_root,
                        job=job,
                        matrix_manifest_sha256=str(args.matrix_manifest_sha256),
                        method_lock_sha256=str(manifest["method_lock_sha256"]),
                        paths=paths,
                        score_command=score_command,
                    )
                    return (
                        binding_path,
                        binding_sha256,
                        _start_shard_child(
                            score_command,
                            output_root=output_root,
                            job=job,
                            shard_index=shard_index,
                            stage="score",
                            stdout=stdout,
                            stderr=stderr,
                            env=_child_env(args.cpu_threads),
                        ),
                    )

                score_started, score_value = _dispatch_score_under_stop_barrier(
                    output_root,
                    coordinator_id=coordinator_id,
                    start=_start_score,
                )
        except D92CCOCHard9K1RunnerError as error:
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "post_prediction_score_input_validation",
                    "error": str(error),
                }
            )
            continue
        if not score_started:
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "post_prediction_systemic_stop_before_score",
                }
            )
            break
        if score_value is None:  # pragma: no cover - guarded by score_started
            raise D92CCOCHard9K1RunnerError("score start value drift")
        score_binding_path, score_binding_sha256, score_child = score_value
        score_returncode = _wait_shard_child(score_child)
        try:
            truth_sidecar_sha256_after_score = _verify_truth_sidecar_snapshot(
                job["truth_sidecar"],
                expected_sha256=str(job["truth_sidecar_sha256"]),
            )
        except D92CCOCHard9K1RunnerError as error:
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "post_prediction_score_truth_validation",
                    "error": str(error),
                }
            )
            continue
        score_path = job_root / "scorer" / "diag_cosine_score.json"
        if score_returncode != 0 or not score_path.is_file() or score_path.is_symlink():
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "score",
                    "returncode": score_returncode,
                }
            )
            continue
        try:
            score_evidence = _validate_score_artifact(
                score_path,
                job=job,
                matrix_manifest_sha256=str(args.matrix_manifest_sha256),
                method_lock_sha256=str(manifest["method_lock_sha256"]),
                truth_sidecar_sha256=truth_sidecar_sha256_after_score,
                before_prediction_path=paths["before_prediction"],
                after_prediction_path=paths["after_prediction"],
                score_binding_path=score_binding_path,
            )
        except D92CCOCHard9K1RunnerError as error:
            failures.append(
                {
                    "job_id": job["job_id"],
                    "stage": "post_prediction_score_artifact_validation",
                    "error": str(error),
                }
            )
            continue
        closure_hashes = {
            field: score_evidence[field] for field in _PREDICTION_CLOSURE_SHA_FIELDS
        }
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
            "score_sha256": _sha256_file(score_path),
            "truth_sidecar_sha256": truth_sidecar_sha256_after_score,
            "truth_sidecar_sha256_before_score": score_evidence[
                "truth_sidecar_sha256"
            ],
            "truth_sidecar_sha256_after_score": truth_sidecar_sha256_after_score,
            "score_binding": str(score_binding_path),
            "score_binding_sha256": score_binding_sha256,
            "score_evidence": score_evidence,
            "truth_sidecar_exposed_to_predictor": False,
            "query_truth_joined_only_after_immutable_predictions": True,
            "query_truth_fed_back_to_predictor": False,
            "prediction_and_scorer_processes_isolated": True,
            "fit_audit_resource_gate": fit_resource_gate,
            "fresh_run_retry_authorized": False,
        }
        _write_job_receipt(job_root, receipt, closure_hashes=closure_hashes)
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


def coordinator_stop(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the one bounded, receipt-verified systemic-stop action."""

    return stop_verified_active_processes(
        args.output_root,
        coordinator_id=str(args.coordinator_id),
        sleep_seconds=float(args.grace_seconds),
    )


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
    stop_parser = commands.add_parser("coordinator-stop")
    stop_parser.add_argument("--output-root", required=True)
    stop_parser.add_argument("--coordinator-id", required=True)
    stop_parser.add_argument("--grace-seconds", type=float, default=1.0)
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        if args.command == "prepare":
            value = prepare(args)
        elif args.command == "smoke":
            value = smoke(args)
        elif args.command == "run-shard":
            value = run_shard(args)
        else:
            value = coordinator_stop(args)
    except (D92CCOCHard9K1RunnerError, D92CCOCHard9K1Error, ValueError) as error:
        print(f"D92 CCOC Hard9+K1 failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return 0 if value["status"] in {
        "CCOC_HARD9_K1_MATRIX_PREPARED",
        "D92_CCOC_HARD9_K1_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "PASS",
        "VERIFIED_STOP_COORDINATOR_COMPLETE",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
