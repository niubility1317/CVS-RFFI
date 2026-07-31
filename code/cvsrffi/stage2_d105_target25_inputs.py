"""Fail-closed D92 -> D105 Target25 input preparation.

The external matrix index is a locator-only control document.  It may name
sealed authority/package artifacts, but it cannot submit physical IDs, roots,
class registries, or ``VALIDATED_ONCE`` claims.  Those facts are reconstructed
from the canonical D92 authority and package readers before one immutable
Target25 plan/context pair is published.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from .somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
from .somph_predictor_bundle import preflight_somph_predictor_bundle_with_authority
from .somph_offline_target_package import (
    load_verified_lineage_context_from_authority_commit,
)
from .stage2_d105_phase1_bundle import (
    BUNDLE_WIRE_NAME,
    D105_CANDIDATE_RUNTIME_FILES,
    load_d105_candidate_method_lock,
    load_d105_candidate_runtime_manifest,
    load_d105_phase1_asset,
)
from .stage2_d105_query_evaluation import build_d105_prediction_context
from .stage2_d105_target25_launcher import (
    CONTEXT_MANIFEST_SCHEMA,
    seal_d105_target25_context_manifest,
)
from .stage2_d105_target25_runner import (
    DEVELOPMENT_CLAIM_SCOPE,
    FORMAL_CLAIM_SCOPE,
    LEO_SCENARIOS,
    TARGET25_SEED,
    TARGET25_SLICES,
    D105Target25Plan,
    D105Target25ScenarioPlan,
    D105Target25StatePlan,
    canonical_sha256,
    freeze_d105_target25_plan,
    write_d105_target25_plan_manifest,
)
from .stage2_diag_cosine_exploration import _validate_matched_packages
from .stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


INDEX_SCHEMA = "cvs.phase2.d105.target25_d92_matrix_index.v1"
PREPARE_RECEIPT_SCHEMA = "cvs.phase2.d105.target25_prepare_receipt.v1"
RUNTIME_CLOSURE_SCHEMA = "cvs.phase2.d105.target25_runtime_entrypoint_closure.v1"

_PACKAGE_REF_FIELDS = {
    "package_root",
    "detached_seal_path",
    "expected_seal_sha256",
    "formal_policy_path",
    "formal_policy_authorization_path",
    "signed_policy_authorization_envelope_path",
    "expected_signed_policy_authorization_envelope_sha256",
}
_AUTHORITY_FIELDS = {
    "receiver",
    "authority_bundle_root",
    "expected_authority_commit_sha256",
    "cache_set_manifest_path",
}
_ROW_FIELDS = {
    "receiver",
    "k_shot",
    "new_count",
    "before_enrollment",
    "before_apply",
    "after_enrollment",
    "after_apply",
    "qknn_lock",
}
_INDEX_FIELDS = {
    "schema",
    "seed",
    "claim_scope",
    "formal_launch_authority",
    "authorities",
    "phase1_bundle_dir",
    "checkpoint_path",
    "candidate_runtime_manifest_path",
    "candidate_method_lock_path",
    "feature_batch_size",
    "score_chunk_size",
    "rows",
}
_PREPARE_RECEIPT_FIELDS = {
    "schema",
    "status",
    "claim_scope",
    "formal_launch_authority",
    "promotable",
    "matrix_index_sha256",
    "plan_manifest_sha256",
    "context_manifest_sha256",
    "plan_receipt_sha256",
    "authority_envelope_root_sha256",
    "data_feature_runtime_sha256",
    "data_materialization_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "outer_row_count",
    "prepare_receipt_sha256",
}


class D105Target25InputError(ValueError):
    """Raised when a D92 locator or derived Target25 fact fails closure."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise D105Target25InputError(f"{name} must be a lowercase SHA256")
    return text


def _regular_path(value: Any, name: str, *, directory: bool = False) -> Path:
    raw = Path(str(value))
    if not raw.is_absolute() or raw.is_symlink() or not raw.exists():
        raise D105Target25InputError(
            f"{name} must be an existing absolute non-symlink path"
        )
    resolved = raw.resolve(strict=True)
    if (directory and not resolved.is_dir()) or (
        not directory and not resolved.is_file()
    ):
        raise D105Target25InputError(f"{name} has the wrong file type")
    return resolved


def _immutable_json(path: Path, name: str) -> dict[str, Any]:
    source = _regular_path(path, name)
    if source.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Target25InputError(f"{name} must be read-only")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D105Target25InputError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise D105Target25InputError(f"{name} must contain an object")
    return value


def _package_ref(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _PACKAGE_REF_FIELDS:
        raise D105Target25InputError(f"{name} package reference closure drift")
    root = _regular_path(value["package_root"], f"{name}.package_root", directory=True)
    seal = _regular_path(
        value["detached_seal_path"], f"{name}.detached_seal_path"
    )
    expected = _sha(value["expected_seal_sha256"], f"{name}.expected_seal_sha256")
    if _sha256_file(seal) != expected:
        raise D105Target25InputError(f"{name} detached seal SHA mismatch")
    policy = _regular_path(value["formal_policy_path"], f"{name}.formal_policy_path")
    authorization = _regular_path(
        value["formal_policy_authorization_path"],
        f"{name}.formal_policy_authorization_path",
    )
    envelope = _regular_path(
        value["signed_policy_authorization_envelope_path"],
        f"{name}.signed_policy_authorization_envelope_path",
    )
    expected_envelope = _sha(
        value["expected_signed_policy_authorization_envelope_sha256"],
        f"{name}.expected_signed_policy_authorization_envelope_sha256",
    )
    if _sha256_file(envelope) != expected_envelope:
        raise D105Target25InputError(f"{name} signed policy envelope SHA mismatch")
    return {
        "package_root": str(root),
        "detached_seal_path": str(seal),
        "expected_seal_sha256": expected,
        "formal_policy_path": str(policy),
        "formal_policy_authorization_path": str(authorization),
        "signed_policy_authorization_envelope_path": str(envelope),
        "expected_signed_policy_authorization_envelope_sha256": expected_envelope,
    }


def _class_handles(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    rows = manifest.get("registered_classes")
    if not isinstance(rows, list):
        raise D105Target25InputError("D92 package class registry missing")
    result = tuple(
        str(item.get("class_handle", "")) if isinstance(item, Mapping) else ""
        for item in rows
    )
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise D105Target25InputError("D92 package class registry drift")
    return result


def _validate_p2_min_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "phase2_query_post_reception_view_fit_access": False,
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise D105Target25InputError("D92 package is not p2_min_v1 compatible")


def _tokens(
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    *,
    scenario: str,
    field: str,
) -> tuple[str, ...]:
    arrays = payloads.get(scenario)
    if not isinstance(arrays, Mapping) or field not in arrays:
        raise D105Target25InputError(f"D92 package {field} missing for {scenario}")
    values = tuple(np.asarray(arrays[field]).astype(str).tolist())
    if not values or any(not item for item in values) or len(set(values)) != len(values):
        raise D105Target25InputError(f"D92 package {field} token closure drift")
    return values


def _load_package(
    ref: Mapping[str, str],
    name: str,
    *,
    authority: Mapping[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    try:
        authority_manifest, authority_seal, authority_audit = (
            preflight_somph_predictor_bundle_with_authority(
                ref["package_root"],
                detached_seal_path=ref["detached_seal_path"],
                expected_seal_sha256=ref["expected_seal_sha256"],
                formal_policy_path=ref["formal_policy_path"],
                formal_policy_authorization_path=(
                    ref["formal_policy_authorization_path"]
                ),
                signed_policy_authorization_envelope_path=(
                    ref["signed_policy_authorization_envelope_path"]
                ),
                expected_signed_policy_authorization_envelope_sha256=(
                    ref["expected_signed_policy_authorization_envelope_sha256"]
                ),
            )
        )
        payloads, manifest, audit = load_verified_somph_predictor_bundle(
            ref["package_root"],
            detached_seal_path=ref["detached_seal_path"],
            expected_seal_sha256=ref["expected_seal_sha256"],
        )
    except Exception as error:
        raise D105Target25InputError(f"{name} canonical package verification failed") from error
    authorization = _immutable_json(
        Path(ref["formal_policy_authorization_path"]),
        f"{name} formal policy authorization",
    )
    if (
        authority_manifest != manifest
        or authority_seal.get("package_root_sha256")
        != manifest.get("package_root_sha256")
        or authority_audit.get("signed_path_free_runtime_authorization_verified")
        is not True
        or authority_audit.get("authority_commit_sha256")
        != authority.get("commit_sha256")
        or authority_audit.get("package_detached_seal_sha256")
        != ref["expected_seal_sha256"]
        or authority_audit.get("formal_policy_authorization_sha256")
        != canonical_sha256(authorization)
        or authorization.get("authority_commit_sha256")
        != authority.get("commit_sha256")
        or authorization.get("dataset_authority_root_sha256")
        != authority.get("dataset_authority_root_sha256")
        or authorization.get("package_root_sha256")
        != manifest.get("package_root_sha256")
        or authorization.get("package_detached_seal_sha256")
        != ref["expected_seal_sha256"]
        or authorization.get("receiver") != manifest.get("receiver")
        or authorization.get("seed") != manifest.get("seed")
        or authorization.get("stage") != manifest.get("stage")
        or authorization.get("registration_state")
        != manifest.get("registration_state")
        or authorization.get("k_shot") != manifest.get("k_shot")
        or audit.get("diagnostic_only") is not True
        or audit.get("formal_launch_authority") is not False
    ):
        raise D105Target25InputError(
            f"{name} signed package/authority boundary drift"
        )
    _validate_p2_min_manifest(manifest)
    return payloads, manifest


def _authority_receipts(
    values: Any, *, claim_scope: str, formal_launch_authority: bool
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list) or len(values) != 5:
        raise D105Target25InputError("D92 authority index must contain five receivers")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping) or set(value) != _AUTHORITY_FIELDS:
            raise D105Target25InputError("D92 authority locator closure drift")
        receiver = str(value["receiver"])
        if not receiver or receiver in result:
            raise D105Target25InputError("D92 authority receiver closure drift")
        root = _regular_path(
            value["authority_bundle_root"], "authority_bundle_root", directory=True
        )
        cache = _regular_path(
            value["cache_set_manifest_path"], "cache_set_manifest_path"
        )
        commit_sha = _sha(
            value["expected_authority_commit_sha256"],
            "expected_authority_commit_sha256",
        )
        try:
            context = load_verified_lineage_context_from_authority_commit(
                cache_set_manifest_path=cache,
                authority_bundle_root=root,
                expected_authority_commit_sha256=commit_sha,
            )
        except Exception as error:
            raise D105Target25InputError(
                f"D92 signed authority verification failed for {receiver}"
            ) from error
        commit = context.get("authority_commit")
        lock = context.get("authority_lock")
        attestation = context.get("authority_attestation")
        if (
            not isinstance(commit, Mapping)
            or not isinstance(lock, Mapping)
            or not isinstance(attestation, Mapping)
            or lock.get("receiver") != receiver
            or lock.get("seed") != TARGET25_SEED
            or context.get("external_authority_lock_verified") is not True
        ):
            raise D105Target25InputError("D92 authority receiver/seed binding drift")
        upstream_formal = context.get("formal_launch_authority")
        if claim_scope == FORMAL_CLAIM_SCOPE:
            if (
                formal_launch_authority is not True
                or upstream_formal is not True
                or commit.get("formal_launch_authority") is not True
            ):
                raise D105Target25InputError(
                    "formal prepare requires formal_launch_authority=true"
                )
        elif (
            claim_scope != DEVELOPMENT_CLAIM_SCOPE
            or formal_launch_authority is not False
            or upstream_formal is not False
            or commit.get("formal_launch_authority") is not False
        ):
            raise D105Target25InputError(
                "development_screen requires signed diagnostic authority and formal=false"
            )
        envelope_sha = _sha(
            commit.get("signed_authority_envelope_sha256"),
            "signed_authority_envelope_sha256",
        )
        result[receiver] = {
            "commit_sha256": commit_sha,
            "envelope_sha256": envelope_sha,
            "dataset_authority_root_sha256": _sha(
                attestation.get("dataset_authority_root_sha256"),
                "dataset_authority_root_sha256",
            ),
        }
    return result


def _state_plan(
    *,
    stage: str,
    scenario: str,
    receiver: str,
    authority: Mapping[str, Any],
    support: Sequence[str],
    query: Sequence[str],
    registered: Sequence[str],
    old: Sequence[str],
    new: Sequence[str],
    active_k: int,
    qknn_lock_digest: str,
    phase1: Mapping[str, Any],
    data_runtime_sha: str,
    data_lock_sha: str,
    candidate_runtime_sha: str,
    candidate_lock_sha: str,
    package_roots: Mapping[str, str],
) -> D105Target25StatePlan:
    capsule_id = canonical_sha256(
        {
            "schema": "cvs.phase2.d105.d92_authority_capsule_adapter.v1",
            "protocol_schema": "p2_min_v1",
            "receiver": receiver,
            "dataset_authority_root_sha256": authority[
                "dataset_authority_root_sha256"
            ],
            "authority_commit_sha256": authority["commit_sha256"],
        }
    )
    split_id = canonical_sha256(
        {
            "schema": "cvs.phase2.d105.d92_split_adapter.v1",
            "protocol_schema": "p2_min_v1",
            "capsule_id": capsule_id,
            "stage": stage,
            "scenario": scenario,
            "support_physical_ids": sorted(support),
            "query_physical_ids": sorted(query),
            "registered_classes": list(registered),
        }
    )
    support_root = canonical_sha256(sorted(support))
    query_root = canonical_sha256(sorted(query))
    _prediction_context_payload, prediction_context_sha256 = (
        build_d105_prediction_context(
            registration_state=(
                "BEFORE_REGISTRATION" if stage == "S_B" else "AFTER_REGISTRATION"
            ),
            stage=stage,
            scenario=scenario,
            receiver=receiver,
            seed=TARGET25_SEED,
            active_k=active_k,
            registered_classes=registered,
            capsule_id=capsule_id,
            split_id=split_id,
            split_validator_receipt_sha256=str(authority["commit_sha256"]),
            support_physical_root_sha256=support_root,
            query_physical_root_sha256=query_root,
            package_root_sha256=package_roots,
            phase1_bundle_manifest_sha256=str(phase1["manifest_sha256"]),
            validated_bundle_id_sha256=str(
                phase1["validated_bundle_id_sha256"]
            ),
            bundle_content_root_sha256=str(
                phase1["expected_content_root_sha256"]
            ),
            bundle_validator_receipt_sha256=str(
                phase1["validator_receipt_sha256"]
            ),
            checkpoint_sha256=str(phase1["checkpoint_sha256"]),
            data_feature_runtime_sha256=data_runtime_sha,
            data_materialization_lock_sha256=data_lock_sha,
            d105_candidate_runtime_manifest_sha256=candidate_runtime_sha,
            d105_candidate_method_lock_sha256=candidate_lock_sha,
            qknn_lock_digest=qknn_lock_digest,
        )
    )
    return D105Target25StatePlan(
        stage=stage,
        capsule_id=capsule_id,
        split_id=split_id,
        authority_receipt_sha256=str(authority["commit_sha256"]),
        authority_envelope_sha256=str(authority["envelope_sha256"]),
        data_feature_runtime_sha256=data_runtime_sha,
        data_materialization_lock_sha256=data_lock_sha,
        d105_candidate_runtime_manifest_sha256=candidate_runtime_sha,
        d105_candidate_method_lock_sha256=candidate_lock_sha,
        support_physical_ids=tuple(support),
        query_physical_ids=tuple(query),
        registered_classes=tuple(registered),
        old_classes=tuple(old),
        new_classes=tuple(new),
        prediction_context_sha256=prediction_context_sha256,
    )


def _phase1_authority(
    *,
    bundle_dir: Path,
    checkpoint_path: Path,
    candidate_runtime_path: Path,
    candidate_lock_path: Path,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    try:
        asset = load_d105_phase1_asset(
            bundle_dir, require_formal_phase2_eligible=True
        )
        runtime = load_d105_candidate_runtime_manifest(
            candidate_runtime_path,
            expected_checkpoint_sha256=asset.bundle.checkpoint_sha256,
        )
        lock = load_d105_candidate_method_lock(
            candidate_lock_path,
            expected_checkpoint_sha256=asset.bundle.checkpoint_sha256,
            expected_runtime_sha256=runtime[
                "d105_candidate_runtime_manifest_sha256"
            ],
        )
    except Exception as error:
        raise D105Target25InputError("formal D105 Phase1 asset verification failed") from error
    checkpoint_sha = _sha256_file(checkpoint_path)
    if checkpoint_sha != asset.bundle.checkpoint_sha256:
        raise D105Target25InputError("D105 checkpoint/Phase1 asset SHA drift")
    manifest = dict(asset.manifest)
    bundle_wire = bundle_dir / BUNDLE_WIRE_NAME
    authority = {
        "bundle_dir": str(bundle_dir),
        "manifest_sha256": asset.manifest_sha256,
        "bundle_wire_sha256": _sha256_file(bundle_wire),
        "validated_bundle_id_sha256": str(asset.validated_bundle_id_sha256),
        "validator_receipt_sha256": str(asset.validator_receipt_sha256),
        "expected_content_root_sha256": asset.bundle.content_root_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "candidate_runtime_manifest_path": str(candidate_runtime_path),
        "candidate_method_lock_path": str(candidate_lock_path),
        "d105_candidate_runtime_manifest_sha256": runtime[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": lock[
            "d105_candidate_method_lock_sha256"
        ],
    }
    if manifest.get("bundle_wire_sha256") != authority["bundle_wire_sha256"]:
        raise D105Target25InputError("D105 Phase1 bundle wire SHA drift")
    return (
        authority,
        authority["d105_candidate_runtime_manifest_sha256"],
        authority["d105_candidate_method_lock_sha256"],
        dict(lock["lock"]),
    )


def _validate_qknn_lock(
    qknn: Phase1ZIDStudentTLock,
    *,
    active_k: int,
    candidate_lock_document: Mapping[str, Any],
) -> None:
    if qknn.active_k != active_k:
        raise D105Target25InputError("D105 qKNN lock K-shot drift")
    expected_qknn = candidate_lock_document["student_t_qknn"]
    if any(
        getattr(qknn, field) != expected_qknn[field]
        for field in (
            "student_nu",
            "kernel_effective_dim",
            "kernel_volume_gamma",
            "shared_h0",
            "scale_prior_strength",
            "scale_min_ratio",
            "scale_max_ratio",
            "temperature",
        )
    ):
        raise D105Target25InputError("D105 qKNN parameters/candidate lock drift")


def prepare_d105_target25_inputs(
    *,
    matrix_index_path: Path,
    expected_matrix_index_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Derive and publish one immutable Target25 plan/context pair."""

    index_path = _regular_path(matrix_index_path, "D92 matrix index")
    expected_index_sha = _sha(
        expected_matrix_index_sha256, "expected_matrix_index_sha256"
    )
    if _sha256_file(index_path) != expected_index_sha:
        raise D105Target25InputError("external D92 matrix index SHA mismatch")
    index = _immutable_json(index_path, "D92 matrix index")
    if set(index) != _INDEX_FIELDS or index.get("schema") != INDEX_SCHEMA:
        raise D105Target25InputError("D92 matrix index schema closure drift")
    if index.get("seed") != TARGET25_SEED:
        raise D105Target25InputError("D92 matrix index seed drift")
    claim_scope = str(index.get("claim_scope"))
    formal = index.get("formal_launch_authority")
    if claim_scope not in (FORMAL_CLAIM_SCOPE, DEVELOPMENT_CLAIM_SCOPE):
        raise D105Target25InputError("D92 matrix index claim scope drift")
    authorities = _authority_receipts(
        index["authorities"],
        claim_scope=claim_scope,
        formal_launch_authority=formal,
    )
    phase1_dir = _regular_path(
        index["phase1_bundle_dir"], "phase1_bundle_dir", directory=True
    )
    checkpoint = _regular_path(index["checkpoint_path"], "checkpoint_path")
    candidate_runtime = _regular_path(
        index["candidate_runtime_manifest_path"], "candidate_runtime_manifest_path"
    )
    candidate_lock = _regular_path(
        index["candidate_method_lock_path"], "candidate_method_lock_path"
    )
    (
        phase1,
        candidate_runtime_sha,
        candidate_lock_sha,
        candidate_lock_document,
    ) = _phase1_authority(
        bundle_dir=phase1_dir,
        checkpoint_path=checkpoint,
        candidate_runtime_path=candidate_runtime,
        candidate_lock_path=candidate_lock,
    )
    target25_lock = candidate_lock_document["target25"]
    if (
        target25_lock["claim_scope"] != claim_scope
        or target25_lock["formal_launch_authority"] is not formal
    ):
        raise D105Target25InputError("Target25 claim scope/candidate lock drift")
    feature_batch_size = index["feature_batch_size"]
    score_chunk_size = index["score_chunk_size"]
    if (
        type(feature_batch_size) is not int
        or feature_batch_size < 1
        or (
            score_chunk_size is not None
            and (type(score_chunk_size) is not int or score_chunk_size < 1)
        )
    ):
        raise D105Target25InputError("D105 batch/chunk configuration drift")
    rows = index["rows"]
    if not isinstance(rows, list) or len(rows) != 25:
        raise D105Target25InputError("D92 matrix index must contain 25 rows")
    expected_keys = [
        (receiver, k_shot, new_count)
        for receiver in authorities
        for k_shot, new_count in TARGET25_SLICES
    ]
    if [
        (item.get("receiver"), item.get("k_shot"), item.get("new_count"))
        for item in rows
        if isinstance(item, Mapping)
    ] != expected_keys:
        raise D105Target25InputError("D92 matrix index row order/coverage drift")

    scenario_plans: dict[tuple[str, int, int], tuple[D105Target25ScenarioPlan, ...]] = {}
    context_sources: dict[tuple[str, int, int], dict[str, Any]] = {}
    common_data_runtime: str | None = None
    common_data_lock: str | None = None
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise D105Target25InputError("D92 row locator schema closure drift")
        receiver = str(row["receiver"])
        k_shot = row["k_shot"]
        new_count = row["new_count"]
        key = (receiver, k_shot, new_count)
        refs = {
            name: _package_ref(row[name], name)
            for name in (
                "before_enrollment",
                "before_apply",
                "after_enrollment",
                "after_apply",
            )
        }
        loaded = {
            name: _load_package(
                refs[name],
                name,
                authority=authorities[receiver],
            )
            for name in refs
        }
        before_support, before_manifest = loaded["before_enrollment"]
        before_query, before_apply_manifest = loaded["before_apply"]
        after_support, after_manifest = loaded["after_enrollment"]
        after_query, after_apply_manifest = loaded["after_apply"]
        try:
            _validate_matched_packages(before_manifest, before_apply_manifest)
            _validate_matched_packages(after_manifest, after_apply_manifest)
        except Exception as error:
            raise D105Target25InputError("D92 enrollment/apply pairing failed") from error
        for manifest, state in (
            (before_manifest, "before"),
            (after_manifest, "after"),
        ):
            if (
                manifest.get("receiver") != receiver
                or manifest.get("seed") != TARGET25_SEED
                or manifest.get("k_shot") != k_shot
                or manifest.get("registration_state") != state
            ):
                raise D105Target25InputError("D92 package row/lifecycle binding drift")
        old = _class_handles(before_manifest)
        registered_after = _class_handles(after_manifest)
        if registered_after[: len(old)] != old:
            raise D105Target25InputError("D92 after registry is not an old-prefix extension")
        new = registered_after[len(old) :]
        if len(new) != new_count:
            raise D105Target25InputError("D92 new-class count drift")
        data_runtime = _sha(
            before_manifest.get("feature_runtime_sha256"), "feature_runtime_sha256"
        )
        data_lock = _sha(before_manifest.get("method_lock_sha256"), "method_lock_sha256")
        if any(
            manifest.get("feature_runtime_sha256") != data_runtime
            or manifest.get("method_lock_sha256") != data_lock
            for manifest in (before_apply_manifest, after_manifest, after_apply_manifest)
        ):
            raise D105Target25InputError("D92 data runtime/materialization lock drift")
        if common_data_runtime is None:
            common_data_runtime, common_data_lock = data_runtime, data_lock
        elif (data_runtime, data_lock) != (common_data_runtime, common_data_lock):
            raise D105Target25InputError("Target25 common D92 identity plane drift")
        package_roots = {
            name: _sha(manifest["package_root_sha256"], f"{name}.package_root_sha256")
            for name, (_payload, manifest) in loaded.items()
        }
        try:
            qknn = Phase1ZIDStudentTLock(**dict(row["qknn_lock"]))
        except (TypeError, ValueError) as error:
            raise D105Target25InputError(
                "D105 qKNN lock reconstruction failed"
            ) from error
        _validate_qknn_lock(
            qknn,
            active_k=k_shot,
            candidate_lock_document=candidate_lock_document,
        )
        scenarios = []
        for scenario in LEO_SCENARIOS:
            before_support_tokens = _tokens(
                before_support, scenario=scenario, field="support_tokens"
            )
            before_query_tokens = _tokens(
                before_query, scenario=scenario, field="query_tokens"
            )
            after_support_tokens = _tokens(
                after_support, scenario=scenario, field="support_tokens"
            )
            after_query_tokens = _tokens(
                after_query, scenario=scenario, field="query_tokens"
            )
            scenarios.append(
                D105Target25ScenarioPlan(
                    scenario=scenario,
                    before=_state_plan(
                        stage="S_B",
                        scenario=scenario,
                        receiver=receiver,
                        authority=authorities[receiver],
                        support=before_support_tokens,
                        query=before_query_tokens,
                        registered=old,
                        old=old,
                        new=(),
                        active_k=k_shot,
                        qknn_lock_digest=qknn.lock_digest,
                        phase1=phase1,
                        data_runtime_sha=data_runtime,
                        data_lock_sha=data_lock,
                        candidate_runtime_sha=candidate_runtime_sha,
                        candidate_lock_sha=candidate_lock_sha,
                        package_roots=package_roots,
                    ),
                    after=_state_plan(
                        stage="S_C",
                        scenario=scenario,
                        receiver=receiver,
                        authority=authorities[receiver],
                        support=after_support_tokens,
                        query=after_query_tokens,
                        registered=registered_after,
                        old=old,
                        new=new,
                        active_k=k_shot,
                        qknn_lock_digest=qknn.lock_digest,
                        phase1=phase1,
                        data_runtime_sha=data_runtime,
                        data_lock_sha=data_lock,
                        candidate_runtime_sha=candidate_runtime_sha,
                        candidate_lock_sha=candidate_lock_sha,
                        package_roots=package_roots,
                    ),
                )
            )
        scenario_plans[key] = tuple(scenarios)
        context_sources[key] = {
            **refs,
            "qknn_lock": asdict(qknn),
        }
    plan = freeze_d105_target25_plan(
        candidate_runtime_manifest_path=candidate_runtime,
        candidate_method_lock_path=candidate_lock,
        receivers=tuple(authorities),
        scenario_plans=scenario_plans,
        claim_scope=claim_scope,
        formal_launch_authority=formal,
    )
    context_rows = []
    for planned in plan.rows:
        key = (planned.receiver, planned.k_shot, planned.new_count)
        split_authorities = []
        for scenario in planned.scenarios:
            for state in (scenario.before, scenario.after):
                split_authorities.append(
                    {
                        "registration_state": state.registration_state,
                        "scenario": scenario.scenario,
                        "capsule_id": state.capsule_id,
                        "split_id": state.split_id,
                        "validator_receipt_sha256": state.authority_receipt_sha256,
                        "support_token_root_sha256": state.support_physical_root_sha256,
                        "query_token_root_sha256": state.query_physical_root_sha256,
                        "protocol_schema": "p2_min_v1",
                        "phase2_data_status": "VALIDATED_ONCE",
                    }
                )
        context_rows.append(
            {
                "row_id": planned.row_id,
                "receiver": planned.receiver,
                "k_shot": planned.k_shot,
                "new_count": planned.new_count,
                **context_sources[key],
                "split_authorities": split_authorities,
                "phase1_bundle": phase1,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": phase1["checkpoint_sha256"],
                "data_feature_runtime_sha256": plan.data_feature_runtime_sha256,
                "data_materialization_lock_sha256": plan.data_materialization_lock_sha256,
                "feature_batch_size": feature_batch_size,
                "score_chunk_size": score_chunk_size,
            }
        )
    context = {
        "schema": CONTEXT_MANIFEST_SCHEMA,
        "plan_receipt_sha256": plan.plan_receipt_sha256,
        "claim_scope": plan.claim_scope,
        "formal_launch_authority": plan.formal_launch_authority,
        "authority_envelope_root_sha256": plan.authority_envelope_root_sha256,
        "data_feature_runtime_sha256": plan.data_feature_runtime_sha256,
        "data_materialization_lock_sha256": plan.data_materialization_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": (
            plan.d105_candidate_runtime_manifest_sha256
        ),
        "d105_candidate_method_lock_sha256": plan.d105_candidate_method_lock_sha256,
        "rows": context_rows,
    }
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable prepare output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D105Target25InputError("prepare output parent is unsafe")
    destination.mkdir()
    plan_path = destination / "target25_plan.json"
    context_path = destination / "target25_context.json"
    write_d105_target25_plan_manifest(plan_path, plan)
    seal_d105_target25_context_manifest(context_path, context)
    receipt = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "status": "TARGET25_INPUTS_PREPARED",
        "claim_scope": plan.claim_scope,
        "formal_launch_authority": plan.formal_launch_authority,
        "promotable": False,
        "matrix_index_sha256": expected_index_sha,
        "plan_manifest_sha256": _sha256_file(plan_path),
        "context_manifest_sha256": _sha256_file(context_path),
        "plan_receipt_sha256": plan.plan_receipt_sha256,
        "authority_envelope_root_sha256": plan.authority_envelope_root_sha256,
        "data_feature_runtime_sha256": plan.data_feature_runtime_sha256,
        "data_materialization_lock_sha256": plan.data_materialization_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": (
            plan.d105_candidate_runtime_manifest_sha256
        ),
        "d105_candidate_method_lock_sha256": (
            plan.d105_candidate_method_lock_sha256
        ),
        "outer_row_count": len(plan.rows),
    }
    receipt["prepare_receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = destination / "prepare_receipt.json"
    raw = json.dumps(
        receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    with receipt_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(receipt_path, stat.S_IREAD)
    return {
        **receipt,
        "plan_manifest": str(plan_path),
        "context_manifest": str(context_path),
        "prepare_receipt": str(receipt_path),
    }


def load_d105_target25_prepare_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    plan_manifest_path: Path,
    context_manifest_path: Path,
    plan: D105Target25Plan,
) -> dict[str, Any]:
    """Verify the unique prepare output before prediction or scoring starts."""

    if type(plan) is not D105Target25Plan:
        raise D105Target25InputError("exact frozen Target25 plan required")
    receipt_path = _regular_path(path, "Target25 prepare receipt")
    if _sha256_file(receipt_path) != _sha(
        expected_file_sha256, "expected prepare receipt file SHA256"
    ):
        raise D105Target25InputError("Target25 prepare receipt file SHA mismatch")
    plan_path = _regular_path(plan_manifest_path, "Target25 plan manifest")
    context_path = _regular_path(context_manifest_path, "Target25 context manifest")
    expected_parent = receipt_path.parent.resolve(strict=True)
    if (
        receipt_path.name != "prepare_receipt.json"
        or plan_path.parent.resolve(strict=True) != expected_parent
        or context_path.parent.resolve(strict=True) != expected_parent
        or plan_path.name != "target25_plan.json"
        or context_path.name != "target25_context.json"
    ):
        raise D105Target25InputError(
            "prepare receipt, plan and context must be one immutable prepare directory"
        )
    receipt = _immutable_json(receipt_path, "Target25 prepare receipt")
    if set(receipt) != _PREPARE_RECEIPT_FIELDS:
        raise D105Target25InputError("Target25 prepare receipt field closure drift")
    unsigned = {
        key: value
        for key, value in receipt.items()
        if key != "prepare_receipt_sha256"
    }
    expected = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "status": "TARGET25_INPUTS_PREPARED",
        "claim_scope": plan.claim_scope,
        "formal_launch_authority": plan.formal_launch_authority,
        "promotable": False,
        "plan_manifest_sha256": _sha256_file(plan_path),
        "context_manifest_sha256": _sha256_file(context_path),
        "plan_receipt_sha256": plan.plan_receipt_sha256,
        "authority_envelope_root_sha256": plan.authority_envelope_root_sha256,
        "data_feature_runtime_sha256": plan.data_feature_runtime_sha256,
        "data_materialization_lock_sha256": plan.data_materialization_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": (
            plan.d105_candidate_runtime_manifest_sha256
        ),
        "d105_candidate_method_lock_sha256": (
            plan.d105_candidate_method_lock_sha256
        ),
        "outer_row_count": len(plan.rows),
    }
    if (
        receipt.get("prepare_receipt_sha256") != canonical_sha256(unsigned)
        or any(receipt.get(name) != value for name, value in expected.items())
        or _sha(receipt.get("matrix_index_sha256"), "matrix_index_sha256")
        != receipt["matrix_index_sha256"]
    ):
        raise D105Target25InputError("Target25 prepare receipt binding drift")
    return dict(receipt)


def build_d105_target25_prepare_binding(
    *,
    prepare_receipt_path: Path,
    expected_prepare_receipt_file_sha256: str,
    matrix_index_path: Path,
    expected_matrix_index_sha256: str,
    plan_manifest_path: Path,
    context_manifest_path: Path,
    plan: D105Target25Plan,
    run_id: str,
    git_commit: str,
    nonce_ledger_identity_sha256: str,
) -> dict[str, Any]:
    """Recompute the exact binding later checked by the pinned prepare signer."""

    receipt = load_d105_target25_prepare_receipt(
        prepare_receipt_path,
        expected_file_sha256=expected_prepare_receipt_file_sha256,
        plan_manifest_path=plan_manifest_path,
        context_manifest_path=context_manifest_path,
        plan=plan,
    )
    index_path = _regular_path(matrix_index_path, "Target25 matrix index")
    index_sha = _sha(expected_matrix_index_sha256, "expected matrix index SHA256")
    if _sha256_file(index_path) != index_sha or receipt["matrix_index_sha256"] != index_sha:
        raise D105Target25InputError("signed prepare matrix index binding drift")
    run = str(run_id)
    commit = str(git_commit)
    if (
        not run
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run)
        or not 1 <= len(run) <= 128
        or not 40 <= len(commit) <= 64
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise D105Target25InputError("signed prepare run ID or Git commit drift")
    return {
        "run_id": run,
        "git_commit": commit,
        "matrix_index_sha256": index_sha,
        "prepare_receipt_file_sha256": _sha256_file(
            _regular_path(prepare_receipt_path, "Target25 prepare receipt")
        ),
        "prepare_receipt_sha256": receipt["prepare_receipt_sha256"],
        "nonce_ledger_identity_sha256": _sha(
            nonce_ledger_identity_sha256,
            "nonce ledger identity SHA256",
        ),
        "plan_manifest_sha256": receipt["plan_manifest_sha256"],
        "context_manifest_sha256": receipt["context_manifest_sha256"],
        "plan_receipt_sha256": receipt["plan_receipt_sha256"],
        "authority_envelope_root_sha256": receipt[
            "authority_envelope_root_sha256"
        ],
        "d105_candidate_runtime_manifest_sha256": receipt[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": receipt[
            "d105_candidate_method_lock_sha256"
        ],
        "claim_scope": receipt["claim_scope"],
        "formal_launch_authority": receipt["formal_launch_authority"],
    }


def d105_target25_runtime_entrypoint_closure(code_root: Path) -> dict[str, Any]:
    """Return the exact executable closure for later Phase1 inclusion."""

    root = Path(code_root).resolve(strict=True)
    relative_files = D105_CANDIDATE_RUNTIME_FILES
    members = []
    for relative in relative_files:
        path = _regular_path(root / relative, f"runtime closure:{relative}")
        members.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema": RUNTIME_CLOSURE_SCHEMA,
        "members": members,
        "closure_sha256": canonical_sha256(members),
    }


__all__ = [
    "D105Target25InputError",
    "INDEX_SCHEMA",
    "PREPARE_RECEIPT_SCHEMA",
    "RUNTIME_CLOSURE_SCHEMA",
    "build_d105_target25_prepare_binding",
    "d105_target25_runtime_entrypoint_closure",
    "load_d105_target25_prepare_receipt",
    "prepare_d105_target25_inputs",
]
