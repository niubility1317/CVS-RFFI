"""Truth-free, SHA-pinned D92 package binding for D107 Target125.

The builder deliberately consumes only the completed D92 matrix locator and
sealed package references.  It never opens a package, a query label, a query
role, or a score.  Opening the sealed received-IQ packages is owned by the
predictor after this immutable plan/context pair has been published.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from .stage2_d107_matrix_protocol import (
    ARMS,
    CANDIDATE_ID,
    OUTER_JOB_COUNT,
    PROTOCOL_SCHEMA,
    RECEIVERS,
    SCENE_ROW_COUNT,
    SCENES,
    SEEDS,
    SURFACE_COUNT,
    TARGET125_SLICES,
    canonical_bytes,
    canonical_sha256,
    freeze_d107_target125_matrix,
)


D92_MATRIX_SCHEMA = "cvs.phase2.somph_diag_125_stability.v1"
D92_CANDIDATE = "d92_registration_balanced_covariance"
PLAN_SCHEMA = "cvs.phase2.d107.scmkrr.target125.input_plan.v1"
CONTEXT_SCHEMA = "cvs.phase2.d107.scmkrr.target125.input_context.v1"
PREPARE_RECEIPT_SCHEMA = "cvs.phase2.d107.scmkrr.target125.prepare_receipt.v1"
D107_METHOD_LOCK_SCHEMA = "cvs.phase2.d107.scmkrr_method_lock.v1"

_PACKAGE_NAMES = (
    "before_enrollment",
    "before_apply",
    "after_enrollment",
    "after_apply",
)


class D107Target125InputError(ValueError):
    """Raised when a D92/D107 locator cannot close the Target125 input plane."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D107Target125InputError(f"{name} must be a lowercase SHA256")
    return value


def _regular_path(value: Any, name: str, *, directory: bool = False) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or path.is_symlink() or not path.exists():
        raise D107Target125InputError(
            f"{name} must be an existing absolute non-symlink path"
        )
    path = path.resolve(strict=True)
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise D107Target125InputError(f"{name} has the wrong file type")
    return path


def _read_json(path: Path, name: str, expected_sha256: str) -> dict[str, Any]:
    source = _regular_path(path, name)
    if _sha256_file(source) != _sha(expected_sha256, f"expected {name} SHA256"):
        raise D107Target125InputError(f"{name} SHA mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D107Target125InputError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise D107Target125InputError(f"{name} must contain an object")
    return value


def _asset(path: Path, expected_sha256: str, name: str) -> dict[str, str]:
    source = _regular_path(path, name)
    expected = _sha(expected_sha256, f"expected {name} SHA256")
    if _sha256_file(source) != expected:
        raise D107Target125InputError(f"{name} SHA mismatch")
    return {"path": str(source), "sha256": expected}


def _assert_d92_protocol(manifest: Mapping[str, Any]) -> None:
    """Reject an Oracle/clean-access D92 matrix before any package is selected."""

    if (
        manifest.get("schema") != D92_MATRIX_SCHEMA
        or manifest.get("candidate") != D92_CANDIDATE
        or manifest.get("job_count") != OUTER_JOB_COUNT
        or manifest.get("receivers") != list(RECEIVERS)
    ):
        raise D107Target125InputError("D92 matrix identity/grid drift")
    candidate_text = " ".join(
        str(manifest.get(name, ""))
        for name in ("candidate", "claim_scope", "result_label", "protocol_note")
    ).lower()
    if "oracle" in candidate_text:
        raise D107Target125InputError("Oracle-labelled D92 matrix is forbidden")
    contract = manifest.get("phase2_contract")
    if not isinstance(contract, Mapping):
        raise D107Target125InputError("D92 phase2 contract is missing")
    for name in (
        "clean_sample_access",
        "clean_derived_signal_access",
        "phase2_clean_cache_reachable",
        "phase2_clean_control_flow_reachable",
        "phase2_clean_dataset_reachable",
        "phase2_query_batch_global_assignment",
        "phase2_query_class_quota_access",
        "phase2_query_role_oracle_access",
        "phase2_query_true_batch_class_count_access",
    ):
        if contract.get(name) is not False:
            raise D107Target125InputError(f"D92 contract grants forbidden {name}")


def _package_ref(job_root: Path, phase: str, profile: str) -> dict[str, str]:
    if phase not in {"before", "after"} or profile not in {"enrollment", "apply"}:
        raise D107Target125InputError("internal D92 package selector drift")
    package_leaf = "enrollment_only" if profile == "enrollment" else "apply_only_staging"
    seal_root = (
        job_root / "offline" / "seals"
        if profile == "enrollment"
        else job_root / "apply_seals"
    )
    seal_leaf = (
        f"{phase}_enrollment.seal.json"
        if profile == "enrollment"
        else f"{phase}_apply.seal.json"
    )
    root = _regular_path(
        job_root / "offline" / "predictor" / phase / package_leaf,
        f"D92 {phase} {profile} package root",
        directory=True,
    )
    seal = _regular_path(
        seal_root / seal_leaf,
        f"D92 {phase} {profile} detached seal",
    )
    return {
        "package_root": str(root),
        "detached_seal_path": str(seal),
        "expected_seal_sha256": _sha256_file(seal),
    }


def _required_job_value(job: Mapping[str, Any], name: str) -> Any:
    if name not in job:
        raise D107Target125InputError(f"D92 job misses {name}")
    return job[name]


def _selected_rows(
    manifest: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    _assert_d92_protocol(manifest)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != OUTER_JOB_COUNT:
        raise D107Target125InputError("D92 matrix must contain all 125 jobs")
    source_by_key: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, Mapping):
            raise D107Target125InputError("D92 job must be an object")
        receiver = _required_job_value(job, "receiver")
        seed = _required_job_value(job, "seed")
        k_shot = _required_job_value(job, "k_shot")
        new_count = _required_job_value(job, "new_class_count")
        if (
            receiver not in RECEIVERS
            or type(seed) is not int
            or seed not in SEEDS
            or (k_shot, new_count) not in TARGET125_SLICES
            or job.get("candidate") != D92_CANDIDATE
            or job.get("scenarios") != list(SCENES)
        ):
            raise D107Target125InputError("D92 job identity/scenario drift")
        key = (str(receiver), int(seed), int(k_shot), int(new_count))
        if key in source_by_key:
            raise D107Target125InputError("D92 matrix has duplicate job keys")
        source_by_key[key] = job

    expected_source_keys = {
        (receiver, seed, k_shot, new_count)
        for receiver in RECEIVERS
        for seed in SEEDS
        for k_shot, new_count in TARGET125_SLICES
    }
    if set(source_by_key) != expected_source_keys:
        raise D107Target125InputError("D92 matrix does not close the full 125 grid")

    matrix = freeze_d107_target125_matrix()
    rows: list[dict[str, Any]] = []
    for outer in matrix.outer_rows:
        source_pool_k = 10 if (outer.k_shot, outer.new_count) == (5, 20) else outer.k_shot
        source = source_by_key.get(
            (outer.receiver, outer.seed, source_pool_k, outer.new_count)
        )
        if source is None:
            raise D107Target125InputError("D92 source pool has a missing Target125 job")
        source_job_id = _required_job_value(source, "job_id")
        if type(source_job_id) is not str or not source_job_id:
            raise D107Target125InputError("D92 source job_id drift")
        job_root = _regular_path(
            output_root / "jobs" / source_job_id,
            "D92 job output root",
            directory=True,
        )
        if source.get("output_root") != str(job_root):
            raise D107Target125InputError("D92 job output-root binding drift")
        packages = {
            "before_enrollment": _package_ref(job_root, "before", "enrollment"),
            "before_apply": _package_ref(job_root, "before", "apply"),
            "after_enrollment": _package_ref(job_root, "after", "enrollment"),
            "after_apply": _package_ref(job_root, "after", "apply"),
        }
        if tuple(packages) != _PACKAGE_NAMES:
            raise D107Target125InputError("D92 package ordering drift")
        rows.append(
            {
                "outer_id": outer.outer_id,
                "source_d92_job_id": source_job_id,
                "receiver": outer.receiver,
                "seed": outer.seed,
                "k_shot": outer.k_shot,
                "active_k": outer.k_shot,
                "new_count": outer.new_count,
                "source_pool_k": source_pool_k,
                "k5_prefix_from_matched_k10": (
                    outer.k_shot == 5 and outer.new_count == 20
                ),
                "packages": packages,
            }
        )
    if len(rows) != OUTER_JOB_COUNT:
        raise D107Target125InputError("D107 Target125 row count drift")
    return rows


def _d107_method_lock(path: Path, expected_sha256: str) -> dict[str, str]:
    source = _regular_path(path, "D107 method lock")
    expected = _sha(expected_sha256, "expected D107 method-lock SHA256")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise D107Target125InputError("D107 method-lock SHA mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D107Target125InputError("D107 method lock is not valid UTF-8 JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("schema") != D107_METHOD_LOCK_SCHEMA
        or document.get("candidate_id") != CANDIDATE_ID
        or document.get("protocol_schema") != PROTOCOL_SCHEMA
        or document.get("feature_view") != "signed_z_id_l2"
        or document.get("arms") != list(ARMS)
    ):
        raise D107Target125InputError("D107 method-lock identity drift")
    return {"path": str(source), "sha256": expected}


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    raw = canonical_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def prepare_d107_target125_inputs(
    *,
    d92_matrix_manifest_path: Path,
    expected_d92_matrix_manifest_sha256: str,
    d92_output_root: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    d107_method_lock_path: Path,
    expected_d107_method_lock_sha256: str,
    rdce_asset_dir: Path,
    expected_rdce_wire_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish immutable, truth-free D107 Target125 plan/context/receipt files."""

    matrix_path = _regular_path(d92_matrix_manifest_path, "D92 matrix manifest")
    matrix_sha = _sha(
        expected_d92_matrix_manifest_sha256, "expected D92 matrix-manifest SHA256"
    )
    output_root = _regular_path(d92_output_root, "D92 output root", directory=True)
    manifest = _read_json(matrix_path, "D92 matrix manifest", matrix_sha)
    checkpoint = _asset(checkpoint_path, expected_checkpoint_sha256, "checkpoint")
    method_lock = _d107_method_lock(
        d107_method_lock_path, expected_d107_method_lock_sha256
    )
    d92_sealed_runtime_sha = _sha(
        manifest.get("sealed_runtime_sha256"), "D92 sealed runtime SHA256"
    )
    if manifest.get("phase1_checkpoint_sha256") != checkpoint["sha256"]:
        raise D107Target125InputError("D92/checkpoint SHA binding drift")
    rdce_directory = _regular_path(rdce_asset_dir, "RDCE asset directory", directory=True)
    rdce_wire_sha = _sha(expected_rdce_wire_sha256, "expected RDCE wire SHA256")
    rows = _selected_rows(manifest, output_root)
    matrix = freeze_d107_target125_matrix()
    identity: dict[str, Any] = {
        "matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "d92_matrix_manifest": {"path": str(matrix_path), "sha256": matrix_sha},
        "d92_output_root": str(output_root),
        "d92_sealed_runtime_sha256": d92_sealed_runtime_sha,
        "checkpoint": checkpoint,
        "d107_method_lock": method_lock,
        "rdce_asset": {
            "directory": str(rdce_directory),
            "wire_sha256": rdce_wire_sha,
        },
    }
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "matrix_protocol": matrix.receipt_payload(),
        "identity": identity,
        "rows": rows,
    }
    plan["plan_receipt_sha256"] = canonical_sha256(plan)
    context: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "identity": identity,
        "rows": rows,
    }
    context["context_receipt_sha256"] = canonical_sha256(context)

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable prepare output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D107Target125InputError("unsafe prepare output parent")
    destination.mkdir()
    plan_path = destination / "target125_plan.json"
    context_path = destination / "target125_context.json"
    plan_file_sha = _write_json_new(plan_path, plan)
    context_file_sha = _write_json_new(context_path, context)
    receipt: dict[str, Any] = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "status": "TARGET125_D92_PACKAGES_LOCATED",
        "matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "context_receipt_sha256": context["context_receipt_sha256"],
        "plan_file_sha256": plan_file_sha,
        "context_file_sha256": context_file_sha,
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": SURFACE_COUNT // 2,
        "surface_count": SURFACE_COUNT,
        "query_truth_access": False,
        "query_role_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
    }
    receipt["prepare_receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = destination / "prepare_receipt.json"
    receipt_file_sha = _write_json_new(receipt_path, receipt)
    return {
        **receipt,
        "plan_manifest": str(plan_path),
        "context_manifest": str(context_path),
        "prepare_receipt": str(receipt_path),
        "prepare_receipt_file_sha256": receipt_file_sha,
    }


__all__ = [
    "CONTEXT_SCHEMA",
    "D107_METHOD_LOCK_SCHEMA",
    "D107Target125InputError",
    "D92_CANDIDATE",
    "D92_MATRIX_SCHEMA",
    "PLAN_SCHEMA",
    "PREPARE_RECEIPT_SCHEMA",
    "prepare_d107_target125_inputs",
]
