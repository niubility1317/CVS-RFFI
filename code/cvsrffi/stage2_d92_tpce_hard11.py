"""Frozen D92 D42 Tail-Pair Code Exchange Hard11 mechanical matrix.

This layer owns only matrix identity, immutable package joins and method-lock
validation. The TPCE science implementation is supplied by the paired core;
this module does not tune, score or select from query results.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from cvsrffi import stage2_d92_pareto_distill_hard11 as _base

_BASE_EXPECTED_LOCK = _base._expected_lock


ARM_ID = "E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE"
CANDIDATE_ID = "d92_e0_full_d42_tail_pair_code_exchange"
ARM_ORDER = (ARM_ID,)
ARM_CANDIDATE_IDS = {ARM_ID: CANDIDATE_ID}
ARM_ROLES = {ARM_ID: "primary"}
PRIMARY_ARM = ARM_ID
CLAIM_SCOPE = _base.CLAIM_SCOPE
SHARD_COUNT = _base.SHARD_COUNT
SCENES = _base.SCENES
SMOKE_OUTER_KEY = _base.SMOKE_OUTER_KEY
LIVENESS_OUTER_KEY = _base.LIVENESS_OUTER_KEY
HARD11_ROWS = _base.HARD11_ROWS
HARD11_V1_ROWS = HARD11_ROWS
CONTEXT_SHA256 = _base.CONTEXT_SHA256
GROUND_COMPONENT_DIR = _base.GROUND_COMPONENT_DIR
GROUND_MANIFEST_PATH = _base.GROUND_MANIFEST_PATH
GROUND_MANIFEST_SHA256 = _base.GROUND_MANIFEST_SHA256
SOURCE_D92_OUTPUT_ROOT = _base.SOURCE_D92_OUTPUT_ROOT
HISTORICAL_BASELINE_PATH = _base.HISTORICAL_BASELINE_PATH
HISTORICAL_BASELINE_SHA256 = _base.HISTORICAL_BASELINE_SHA256
HISTORICAL_PER_OLD_CLASS_PATH = _base.HISTORICAL_PER_OLD_CLASS_PATH
HISTORICAL_PER_OLD_CLASS_SHA256 = _base.HISTORICAL_PER_OLD_CLASS_SHA256
RAW_SCORE_ROOT = _base.RAW_SCORE_ROOT
STRICT_PARETO_THRESHOLDS = dict(_base.STRICT_PARETO_THRESHOLDS)
FIT_GATE = {"k_gt_2_total": 2, "k_gt_2_actual": 1, "k1_alias": "real_inventory"}
RESOURCE_GATE = {
    "registration_wall_p90_max_ns": 150_000_000,
    "registration_wall_ratio_max": 1.5,
    "registration_peak_delta_max_bytes": 512 * 1024,
    "registration_wall_p90_target_max_ns": 120_000_000,
    "registration_wall_ratio_target_max": 1.25,
    "component_fit_reduction_min_fraction_vs_d92": 0.80,
    "component_fit_baseline": "D92_FULL_TWO_STATE_COMPONENT_FIT_COUNT_8*(K+1)",
    "query_macs_equal": True,
    "state_bytes_equal": True,
}
DEPLOYMENT_POLICY = {
    "codec": "post_compile_direct_coef2_qint8_pair_exchange",
    "selection": "synchronous_int32_sum_then_single_boundary_check",
    "quantization": "no_requantize_no_scan",
    "all_fail": "exact_e0_fallback",
    "code_local_correction_max_count": 0,
    "negative_tail_accepted": False,
}
STATE_POSTPROCESS_MODE = "d42_tpce"
REGISTERED_MODE = "full_only"


class D92TPCEHard11Error(ValueError):
    """Raised when frozen TPCE Hard11 identity or closure drifts."""


D92TPCEHard11MatrixError = D92TPCEHard11Error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _expected_rows() -> list[dict[str, Any]]:
    return _base._expected_rows()


def canonical_selection_sha256() -> str:
    payload = {
        "schema": "cvs.phase2.d92_tpce_hard11.selection.v1",
        "selection_id": "D92-E0-FULL-D42-TPCE-Hard11-v1",
        "protocol_schema": "p2_min_v1",
        "claim_scope": CLAIM_SCOPE,
        "order": "explicit_pre_registered_performance_then_k1_liveness",
        "outer_rows": [dict(row) for row in HARD11_ROWS],
        "coverage": {"outer_count": 11, "performance_outer_count": 10, "liveness_outer_count": 1, "scene_count": 3, "scene_row_count": 33, "shard_count": 8},
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


CANONICAL_SELECTION_SHA256 = canonical_selection_sha256()


def _expected_lock() -> dict[str, Any]:
    # Start from the proven baseline lock to retain the sealed source/hash
    # inventory, then replace only frozen TPCE identity and receipts.
    lock = json.loads(json.dumps(_BASE_EXPECTED_LOCK()))
    lock.update({
        "schema": "cvs.phase2.d92_tpce_hard11.method_lock.v1",
        "matrix_schema": "cvs.phase2.d92_tpce_hard11.matrix.v1",
        "job_receipt_schema": "cvs.phase2.d92_tpce_hard11.job_receipt.v1",
        "experiment_id": "D92-E0-FULL-D42-TPCE-Hard11-v1",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "primary_arm": ARM_ID,
        "arms": {ARM_ID: {"candidate_id": CANDIDATE_ID, "role": "primary", "registered_mode": REGISTERED_MODE}},
        "only_promotion_candidate": ARM_ID,
        "fit_gate": dict(FIT_GATE),
        "state_postprocess_mode": STATE_POSTPROCESS_MODE,
        "deployment_policy": dict(DEPLOYMENT_POLICY),
        "resource_gate": dict(RESOURCE_GATE),
    })
    return lock


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, Mapping) or _canonical_bytes(lock) != _canonical_bytes(_expected_lock()):
        raise D92TPCEHard11Error("D42 TPCE Hard11 method lock identity drift")
    return dict(lock)


@contextmanager
def _base_context(*, disable_validation: bool = False) -> Iterator[None]:
    """Temporarily bind the proven mechanical implementation to TPCE identity."""
    names = {
        "ARM_ID": ARM_ID,
        "CANDIDATE_ID": CANDIDATE_ID,
        "ARM_ORDER": ARM_ORDER,
        "ARM_CANDIDATE_IDS": ARM_CANDIDATE_IDS,
        "ARM_ROLES": ARM_ROLES,
        "PRIMARY_ARM": PRIMARY_ARM,
        "CANONICAL_SELECTION_SHA256": CANONICAL_SELECTION_SHA256,
        "FIT_GATE": FIT_GATE,
        "RESOURCE_GATE": RESOURCE_GATE,
        "DEPLOYMENT_POLICY": DEPLOYMENT_POLICY,
        "SMOKE_OUTER_KEY": SMOKE_OUTER_KEY,
        "LIVENESS_OUTER_KEY": LIVENESS_OUTER_KEY,
        "_expected_lock": _expected_lock,
        "validate_method_lock": validate_method_lock,
    }
    old = {name: getattr(_base, name) for name in names}
    old_validator = _base.validate_hard11_manifest
    for name, value in names.items():
        setattr(_base, name, value)
    if disable_validation:
        _base.validate_hard11_manifest = lambda manifest, **_: dict(manifest)
    try:
        yield
    finally:
        _base.validate_hard11_manifest = old_validator
        for name, value in old.items():
            setattr(_base, name, value)


def _base_manifest_validator(manifest: Mapping[str, Any], *, expected_method_lock_sha256: str | None = None, require_package_hashes: bool = False) -> dict[str, Any]:
    probe = dict(manifest)
    probe["schema"] = "cvs.phase2.d92_pareto_distill_hard11.matrix.v1"
    with _base_context():
        return _base.validate_hard11_manifest(probe, expected_method_lock_sha256=expected_method_lock_sha256, require_package_hashes=require_package_hashes)


def validate_hard11_manifest(manifest: Mapping[str, Any], *, expected_method_lock_sha256: str | None = None, require_package_hashes: bool = False) -> dict[str, Any]:
    required_schema = "cvs.phase2.d92_tpce_hard11.matrix.v1"
    if not isinstance(manifest, Mapping) or manifest.get("schema") != required_schema:
        raise D92TPCEHard11Error("TPCE matrix schema drift")
    try:
        _base_manifest_validator(manifest, expected_method_lock_sha256=expected_method_lock_sha256, require_package_hashes=require_package_hashes)
    except (ValueError, KeyError, TypeError) as error:
        raise D92TPCEHard11Error(str(error)) from error
    return dict(manifest)


def build_hard11_manifest(*, context_path: str | Path, method_lock_path: str | Path, output_root: str | Path, require_package_files: bool = True) -> dict[str, Any]:
    with _base_context(disable_validation=True):
        manifest = _base.build_hard11_manifest(context_path=context_path, method_lock_path=method_lock_path, output_root=output_root, require_package_files=require_package_files)
    manifest["schema"] = "cvs.phase2.d92_tpce_hard11.matrix.v1"
    validate_hard11_manifest(manifest, expected_method_lock_sha256=manifest["method_lock_sha256"], require_package_hashes=require_package_files)
    return manifest


build_tpce_hard11_manifest = build_hard11_manifest
build_hard11_matrix_manifest = build_hard11_manifest
validate_manifest = validate_hard11_manifest


__all__ = [
    "ARM_ID", "ARM_ORDER", "ARM_CANDIDATE_IDS", "ARM_ROLES", "CANDIDATE_ID", "CANONICAL_SELECTION_SHA256",
    "CLAIM_SCOPE", "CONTEXT_SHA256", "DEPLOYMENT_POLICY", "D92TPCEHard11Error", "D92TPCEHard11MatrixError",
    "FIT_GATE", "HARD11_ROWS", "HARD11_V1_ROWS", "HISTORICAL_BASELINE_PATH", "HISTORICAL_BASELINE_SHA256",
    "HISTORICAL_PER_OLD_CLASS_PATH", "HISTORICAL_PER_OLD_CLASS_SHA256", "LIVENESS_OUTER_KEY", "PRIMARY_ARM",
    "RAW_SCORE_ROOT", "REGISTERED_MODE", "RESOURCE_GATE", "SCENES", "SHARD_COUNT", "SMOKE_OUTER_KEY",
    "SOURCE_D92_OUTPUT_ROOT", "STATE_POSTPROCESS_MODE", "STRICT_PARETO_THRESHOLDS", "build_hard11_manifest",
    "build_tpce_hard11_manifest", "build_hard11_matrix_manifest", "canonical_selection_sha256",
    "validate_hard11_manifest", "validate_manifest", "validate_method_lock",
]
