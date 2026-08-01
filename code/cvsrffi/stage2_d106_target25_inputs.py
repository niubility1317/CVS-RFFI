"""Truth-free adapter from the completed D92 matrix into D106 Target25.

The adapter consumes the SHA-pinned D92 ``matrix_manifest.json`` and the
immutable D92 output root.  It selects the pre-declared seed-713102 Target25
rows and binds only the four already sealed SOMP-H package locations needed by
the D106 runner.  It does not consume a split locator, truth, scores, formal
policy documents, signed envelopes, or external authority metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from .stage2_d106_matrix_protocol import (
    MATCHED_ARM_PAIR_COUNT,
    OUTER_JOB_COUNT,
    PROTOCOL_SCHEMA,
    RECEIVERS,
    SCENARIO_ROW_COUNT,
    STATE_SURFACE_COUNT,
    TARGET25_SEED,
    TARGET25_SLICES,
    canonical_sha256,
    freeze_d106_matrix_protocol,
)


D92_MATRIX_SCHEMA = "cvs.phase2.somph_diag_125_stability.v1"
D92_CANDIDATE = "d92_registration_balanced_covariance"
PLAN_SCHEMA = "cvs.phase2.d106.target25_input_plan.v2"
CONTEXT_SCHEMA = "cvs.phase2.d106.target25_input_context.v2"
PREPARE_RECEIPT_SCHEMA = "cvs.phase2.d106.target25_input_receipt.v2"
KCR_ROUTE_LOCK_SCHEMA = "cvs.phase2.d106.k_conditioned_route_lock.v1"

# Retained as import aliases for callers which only need a schema symbol.
D106_INDEX_SCHEMA = D92_MATRIX_SCHEMA
D106_SPLIT_LOCATOR_SCHEMA = "REMOVED_NO_EXTERNAL_SPLIT_LOCATOR"

_MATRIX_FIELDS = {
    "candidate",
    "claim_scope",
    "confirmation_seeds",
    "development_seed_excluded",
    "formal_launch_authority",
    "job_count",
    "jobs",
    "locked_shard_count",
    "method_lock",
    "method_lock_sha256",
    "output_root",
    "phase1_checkpoint",
    "phase1_checkpoint_sha256",
    "phase2_contract",
    "planned_shard_job_counts",
    "protocol_note",
    "receivers",
    "row_pair_count",
    "row_pipeline",
    "scenario_pair_count",
    "scenario_state_metric_count",
    "schema",
    "sealed_runtime",
    "sealed_runtime_sha256",
    "slices",
    "stage2_balance",
    "status",
}
_JOB_FIELDS = {
    "authority_bundle",
    "authority_commit_path",
    "authority_commit_sha256",
    "cache_manifest",
    "candidate",
    "index",
    "job_id",
    "k_shot",
    "new_class_count",
    "output_root",
    "planned_shard_index",
    "receiver",
    "row_pair",
    "scenarios",
    "seed",
    "seed_role",
    "support_nesting",
}
_PACKAGE_NAMES = (
    "before_enrollment",
    "before_apply",
    "after_enrollment",
    "after_apply",
)
_KCR_ROUTE_LOCK_FIELDS = {
    "schema",
    "candidate_id",
    "route_by_k",
    "query_truth_access",
    "query_role_access",
    "query_fit_access",
    "query_update_access",
    "query_selection",
}


class D106Target25InputError(ValueError):
    """Raised when the raw D92 package surface cannot close Target25."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise D106Target25InputError(f"{name} must be a lowercase SHA256")
    return text


def _regular_path(value: Any, name: str, *, directory: bool = False) -> Path:
    source = Path(str(value))
    if not source.is_absolute() or source.is_symlink() or not source.exists():
        raise D106Target25InputError(
            f"{name} must be an existing absolute non-symlink path"
        )
    source = source.resolve(strict=True)
    if (directory and not source.is_dir()) or (not directory and not source.is_file()):
        raise D106Target25InputError(f"{name} has the wrong file type")
    return source


def _read_json(path: Path, name: str, expected_sha256: str) -> dict[str, Any]:
    source = _regular_path(path, name)
    if _sha256_file(source) != _sha(expected_sha256, f"expected {name} SHA256"):
        raise D106Target25InputError(f"{name} SHA mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D106Target25InputError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise D106Target25InputError(f"{name} must contain an object")
    return value


def _asset(path: Path, expected_sha256: str, name: str) -> dict[str, str]:
    source = _regular_path(path, name)
    expected = _sha(expected_sha256, f"expected {name} SHA256")
    if _sha256_file(source) != expected:
        raise D106Target25InputError(f"{name} SHA mismatch")
    return {"path": str(source), "sha256": expected}


def _kcr_route_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    source = _regular_path(path, "KCR route lock")
    expected = _sha(expected_sha256, "expected KCR route-lock SHA256")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise D106Target25InputError("KCR route-lock SHA mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Target25InputError("KCR route lock is not valid UTF-8 JSON") from error
    if (
        not isinstance(document, dict)
        or raw not in {_canonical_bytes(document), _canonical_bytes(document) + b"\n"}
        or set(document) != _KCR_ROUTE_LOCK_FIELDS
        or document.get("schema") != KCR_ROUTE_LOCK_SCHEMA
        or document.get("candidate_id") != "D106-KCR/r1"
        or document.get("route_by_k") != {"1": "M_DA", "5": "M0", "10": "M_HEAD"}
        or any(
            document.get(name) is not False
            for name in (
                "query_truth_access",
                "query_role_access",
                "query_fit_access",
                "query_update_access",
                "query_selection",
            )
        )
    ):
        raise D106Target25InputError("KCR route-lock canonical schema/route drift")
    return {
        "path": str(source),
        "sha256": expected,
        "candidate_id": document["candidate_id"],
        "route_by_k": document["route_by_k"],
    }


def _package_ref(job_root: Path, state: str, profile: str) -> dict[str, str]:
    if state not in {"before", "after"} or profile not in {"enrollment", "apply"}:
        raise D106Target25InputError("internal D92 package selector drift")
    package_leaf = "enrollment_only" if profile == "enrollment" else "apply_only_staging"
    seal_root = (
        job_root / "offline" / "seals"
        if profile == "enrollment"
        else job_root / "apply_seals"
    )
    seal_leaf = (
        f"{state}_enrollment.seal.json"
        if profile == "enrollment"
        else f"{state}_apply.seal.json"
    )
    root = _regular_path(
        job_root / "offline" / "predictor" / state / package_leaf,
        f"D92 {state} {profile} package root",
        directory=True,
    )
    seal = _regular_path(
        seal_root / seal_leaf,
        f"D92 {state} {profile} detached seal",
    )
    return {
        "package_root": str(root),
        "detached_seal_path": str(seal),
        "expected_seal_sha256": _sha256_file(seal),
    }


def _target25_rows(
    manifest: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    if set(manifest) != _MATRIX_FIELDS:
        raise D106Target25InputError("D92 matrix manifest field closure drift")
    if (
        manifest.get("schema") != D92_MATRIX_SCHEMA
        or manifest.get("candidate") != D92_CANDIDATE
        or manifest.get("job_count") != 125
        or manifest.get("row_pair_count") != 125
        or manifest.get("receivers") != list(RECEIVERS)
        or manifest.get("formal_launch_authority") is not False
        or manifest.get("claim_scope") != "development_only_not_formal_confirmation"
        or manifest.get("output_root") != str(output_root)
    ):
        raise D106Target25InputError("D92 matrix manifest identity drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 125:
        raise D106Target25InputError("D92 matrix manifest must contain 125 jobs")
    selected: list[Mapping[str, Any]] = []
    for job in jobs:
        if not isinstance(job, Mapping) or set(job) != _JOB_FIELDS:
            raise D106Target25InputError("D92 matrix job field closure drift")
        if job.get("seed") == TARGET25_SEED:
            selected.append(job)
    expected_keys = [
        (receiver, k_shot, new_count)
        for receiver in RECEIVERS
        for k_shot, new_count in TARGET25_SLICES
    ]
    actual_keys = [
        (job.get("receiver"), job.get("k_shot"), job.get("new_class_count"))
        for job in selected
    ]
    if len(selected) != OUTER_JOB_COUNT or actual_keys != expected_keys:
        raise D106Target25InputError("D92 Target25 seed row order/coverage drift")

    source_by_key = {
        (str(job["receiver"]), int(job["k_shot"]), int(job["new_class_count"])): job
        for job in selected
    }
    matrix = freeze_d106_matrix_protocol()
    rows: list[dict[str, Any]] = []
    for frozen in matrix.jobs:
        # D92 built K5 and K10 as independent sealed pools.  The frozen D106
        # K5/new20 comparison instead needs a genuinely matched prefix, so it
        # consumes the same K10 package as K10/new20 and materializes rank<5.
        source_pool_k = (
            10 if (frozen.k_shot, frozen.new_count) == (5, 20) else frozen.k_shot
        )
        source = source_by_key[(frozen.receiver, source_pool_k, frozen.new_count)]
        if (
            source.get("candidate") != D92_CANDIDATE
            or source.get("k_shot") != source_pool_k
            or source.get("new_class_count") != frozen.new_count
            or source.get("scenarios")
            != ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
        ):
            raise D106Target25InputError("D92 Target25 row candidate/scenario drift")
        source_job_id = str(source.get("job_id"))
        job_root = _regular_path(
            output_root / "jobs" / source_job_id,
            "D92 Target25 job output root",
            directory=True,
        )
        if source.get("output_root") != str(job_root):
            raise D106Target25InputError("D92 Target25 job output-root drift")
        packages = {
            "before_enrollment": _package_ref(job_root, "before", "enrollment"),
            "before_apply": _package_ref(job_root, "before", "apply"),
            "after_enrollment": _package_ref(job_root, "after", "enrollment"),
            "after_apply": _package_ref(job_root, "after", "apply"),
        }
        if tuple(packages) != _PACKAGE_NAMES:
            raise D106Target25InputError("D92 package order drift")
        rows.append(
            {
                "job_id": frozen.job_id,
                "source_d92_job_id": source_job_id,
                "receiver": frozen.receiver,
                "seed": frozen.seed,
                "k_shot": frozen.k_shot,
                "source_pool_k": source_pool_k,
                "new_count": frozen.new_count,
                "packages": packages,
            }
        )
    return rows


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    raw = _canonical_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def prepare_d106_target25_inputs(
    *,
    d92_matrix_manifest_path: Path,
    expected_d92_matrix_manifest_sha256: str,
    d92_output_root: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    rdce_wire_path: Path,
    expected_rdce_wire_sha256: str,
    rdce_lock_path: Path,
    expected_rdce_lock_sha256: str,
    rcmr_lock_path: Path,
    expected_rcmr_lock_sha256: str,
    kcr_route_lock_path: Path,
    expected_kcr_route_lock_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish an immutable locator-only D106 Target25 plan/context/receipt."""

    manifest_path = _regular_path(d92_matrix_manifest_path, "D92 matrix manifest")
    manifest_sha = _sha(
        expected_d92_matrix_manifest_sha256,
        "expected D92 matrix manifest SHA256",
    )
    output_root = _regular_path(d92_output_root, "D92 output root", directory=True)
    manifest = _read_json(manifest_path, "D92 matrix manifest", manifest_sha)
    rows = _target25_rows(manifest, output_root)
    assets = {
        "checkpoint": _asset(checkpoint_path, expected_checkpoint_sha256, "checkpoint"),
        "rdce_wire": _asset(rdce_wire_path, expected_rdce_wire_sha256, "RDCE wire"),
        "rdce_lock": _asset(rdce_lock_path, expected_rdce_lock_sha256, "RDCE lock"),
        "rcmr_lock": _asset(rcmr_lock_path, expected_rcmr_lock_sha256, "RCMR lock"),
        "kcr_route_lock": _kcr_route_lock(
            kcr_route_lock_path, expected_kcr_route_lock_sha256
        ),
    }
    matrix = freeze_d106_matrix_protocol()
    identity = {
        "matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "d92_matrix_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "d92_output_root": str(output_root),
        "assets": assets,
    }
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "claim_scope": "development_screen",
        "formal_launch_authority": False,
        "identity": identity,
        "matrix_protocol": matrix.receipt_payload(),
        "rows": rows,
    }
    plan["plan_receipt_sha256"] = canonical_sha256(plan)
    context: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "claim_scope": "development_screen",
        "formal_launch_authority": False,
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "identity": identity,
        "rows": rows,
    }
    context["context_receipt_sha256"] = canonical_sha256(context)

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable prepare output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D106Target25InputError("unsafe prepare output parent")
    destination.mkdir()
    plan_path = destination / "target25_plan.json"
    context_path = destination / "target25_context.json"
    plan_file_sha = _write_json_new(plan_path, plan)
    context_file_sha = _write_json_new(context_path, context)
    receipt: dict[str, Any] = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "status": "TARGET25_D92_PACKAGES_LOCATED",
        "promotable": False,
        "claim_scope": "development_screen",
        "formal_launch_authority": False,
        "matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "d92_matrix_manifest_sha256": manifest_sha,
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "context_receipt_sha256": context["context_receipt_sha256"],
        "plan_file_sha256": plan_file_sha,
        "context_file_sha256": context_file_sha,
        "outer_job_count": OUTER_JOB_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "matched_arm_pair_count": MATCHED_ARM_PAIR_COUNT,
        "state_surface_count": STATE_SURFACE_COUNT,
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
    "D106_INDEX_SCHEMA",
    "D106_SPLIT_LOCATOR_SCHEMA",
    "D106Target25InputError",
    "D92_CANDIDATE",
    "D92_MATRIX_SCHEMA",
    "KCR_ROUTE_LOCK_SCHEMA",
    "PLAN_SCHEMA",
    "PREPARE_RECEIPT_SCHEMA",
    "prepare_d106_target25_inputs",
]
