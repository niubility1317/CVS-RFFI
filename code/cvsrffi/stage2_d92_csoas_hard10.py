"""Frozen D92 E0_FULL_CSOAS Hard9+K1 mechanical matrix.

This module owns only immutable matrix identity, package joins and method-lock
validation.  The CSOAS science path remains in the existing E0D arm/query
implementation and is never changed here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Iterator, Mapping

from cvsrffi import stage2_d92_pareto_distill_hard11 as _base
from cvsrffi import stage2_d92_tcra_hard10 as _tcra

ARM_ID = "E0_FULL_CSOAS"
CANDIDATE_ID = "d92_e0_full_csoas"
ARM_ORDER = (ARM_ID,)
ARM_CANDIDATE_IDS = {ARM_ID: CANDIDATE_ID}
ARM_ROLES = {ARM_ID: "primary"}
PRIMARY_ARM = ARM_ID
CLAIM_SCOPE = "DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN"
REGISTERED_MODE = "csoas_full"
STATE_POSTPROCESS_MODE = "csoas_full"
SHARD_COUNT = 8
SCENES = tuple(_base.SCENES)
SMOKE_OUTER_KEY = "rx_7_7__seed_713104__k_5__new_20"
LIVENESS_OUTER_KEY = "rx_20_1__seed_713106__k_1__new_20"
G0_OUTER_KEY = "rx_7_7__seed_713106__k_10__new_5"
EXCLUDED_OUTER_KEYS = (G0_OUTER_KEY,)

HARD10_ROWS = tuple(
    {"outer_key": row["outer_key"], "role": row["role"], "hard_score": None}
    for row in _tcra.HARD10_ROWS
)
HARD10_V1_ROWS = HARD10_ROWS
HARD11_ROWS = HARD10_ROWS
HARD11_V1_ROWS = HARD10_ROWS

CONTEXT_SHA256 = _base.CONTEXT_SHA256
GROUND_COMPONENT_DIR = _base.GROUND_COMPONENT_DIR
GROUND_MANIFEST_PATH = _base.GROUND_MANIFEST_PATH
GROUND_MANIFEST_SHA256 = _base.GROUND_MANIFEST_SHA256
SOURCE_D92_OUTPUT_ROOT = _base.SOURCE_D92_OUTPUT_ROOT
HISTORICAL_BASELINE_PATH = _tcra.HISTORICAL_BASELINE_PATH
HISTORICAL_BASELINE_SHA256 = _tcra.HISTORICAL_BASELINE_SHA256
HISTORICAL_PER_OLD_CLASS_PATH = _tcra.HISTORICAL_PER_OLD_CLASS_PATH
HISTORICAL_PER_OLD_CLASS_SHA256 = _tcra.HISTORICAL_PER_OLD_CLASS_SHA256
RAW_SCORE_ROOT = _tcra.RAW_SCORE_ROOT
_RAW_SCORE_SHA = dict(getattr(_tcra, "_RAW_SCORE_SHA", {}))

QUERY_CONTRACT = {
    "decision": "per_sample_all_registered_classes",
    "truth_access": False,
    "fit_access": False,
    "update_access": False,
    "selection_access": False,
    "role_oracle_access": False,
    "class_quota_access": False,
    "global_reassignment": False,
}
QUERY_ZERO_FIELDS = (
    "query_truth_access", "query_fit_access", "query_update_access",
    "query_selection_access", "query_role_oracle_access",
    "query_class_quota_access", "query_global_reassignment",
)
CSOAS_QUERY_ZERO_FIELDS = tuple("d92_csoas_" + name for name in (
    "query_fit_access", "query_update_access", "query_selection_access",
    "query_truth_access", "query_role_oracle_access",
    "query_class_quota_access", "query_global_reassignment",
))

FIT_GATE = {
    "k_gt_2_total": 2,
    "k_gt_2_actual": 1,
    "postprocess_fit": 0,
    "k1_alias": "K1_K2_EXACT_D92_FULL_ALIAS",
    "k1_total": 3,
    "k1_actual": 3,
}
STRICT_PARETO_THRESHOLDS = {
    "h_old_new": 0.010,
    "old_balanced_accuracy": 0.015,
    "c_old_acc": 0.010,
    "old_floor": 0.040,
    "seen_new_acc": 0.005,
    "average_forgetting": -0.015,
    "new_to_old_rate": -0.005,
    "old_to_new_rate": -0.005,
}
DIRECTION_GATE = {
    key: {"direction": ">" if key not in {"average_forgetting", "new_to_old_rate", "old_to_new_rate"} else "<", "magnitude": value}
    for key, value in STRICT_PARETO_THRESHOLDS.items()
}
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
    "codec": "existing_d42_csoas_single_full_publish",
    "selection": "support_only_fixed_csoas_covariance",
    "full_synchronous_publish": False,
    "quantization": "existing_e0_d42_codec_no_retry",
    "numeric_fallback_formal_allowed": False,
    "modified_state_fields": ["full_affine_qint8"],
}
GATE = {
    "revision": "csoas_full_v1",
    "fallback": "K1_K2_EXACT_D92_FULL_ALIAS",
    "numeric_fallback_formal": False,
    "codec_retry_formal": False,
    "final": "strict_eight_metric_pareto_with_stability_and_resource_hard_gate",
}
STOP_RULE = {
    "same_normalized_exception_fingerprint_distinct_outer_count": 2,
    "pre_prediction_only": True,
    "shared_run_root_ledger": True,
    "fresh_run_retry_authorized": False,
}


class D92CSOASHard10Error(ValueError):
    """Raised when the frozen CSOAS Hard9+K1 identity drifts."""


D92CSOASHard10MatrixError = D92CSOASHard10Error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pure_path(value: Any) -> PurePath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise D92CSOASHard10Error("path identity drift")
    return PureWindowsPath(value) if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", value) or "\\" in value else PurePosixPath(value)


def _path_matches(actual: Any, root: Any, *parts: str) -> bool:
    try:
        a, r = _pure_path(actual), _pure_path(root)
    except D92CSOASHard10Error:
        return False
    return type(a) is type(r) and a == r.joinpath(*parts)


_OUTER_PATTERN = re.compile(r"^rx_(?P<receiver>[0-9_]+)__seed_(?P<seed>[0-9]+)__k_(?P<k>[0-9]+)__new_(?P<new>[0-9]+)$")


def _parse_outer(key: str) -> tuple[str, int, int, int]:
    match = _OUTER_PATTERN.fullmatch(str(key))
    if match is None:
        raise D92CSOASHard10Error(f"invalid outer key: {key}")
    return match.group("receiver").replace("_", "-"), int(match.group("seed")), int(match.group("k")), int(match.group("new"))


def _expected_rows() -> list[dict[str, Any]]:
    result = []
    for row in HARD10_ROWS:
        receiver, seed, k_shot, new_count = _parse_outer(row["outer_key"])
        result.append({"outer_key": row["outer_key"], "outer_role": row["role"], "hard_score": None, "receiver": receiver, "seed": seed, "k_shot": k_shot, "new_class_count": new_count})
    return result


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outer_count": len(rows), "scene_count": len(SCENES), "scene_row_count": len(rows) * len(SCENES),
        "receiver_counts": dict(sorted(Counter(str(row["receiver"]) for row in rows).items())),
        "seed_counts": dict(sorted(Counter(str(row["seed"]) for row in rows).items())),
        "slice_counts": dict(sorted(Counter(f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}" for row in rows).items())),
        "performance_outer_count": 9, "liveness_outer_count": 1,
    }


def canonical_selection_sha256() -> str:
    payload = {
        "schema": "cvs.phase2.d92_csoas_hard10.selection.v1",
        "selection_id": "D92-E0-FULL-CSOAS-Hard9-K1-v1",
        "protocol_schema": "p2_min_v1",
        "claim_scope": CLAIM_SCOPE,
        "order": "explicit_pre_registered_performance_then_k1_liveness",
        "outer_rows": [dict(row) for row in HARD10_ROWS],
        "excluded_outer_keys": list(EXCLUDED_OUTER_KEYS),
        "coverage": {"outer_count": 10, "performance_outer_count": 9, "liveness_outer_count": 1, "scene_count": 3, "scene_row_count": 30, "shard_count": 8},
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


CANONICAL_SELECTION_SHA256 = canonical_selection_sha256()


def _expected_lock() -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.d92_csoas_hard10.method_lock.v1",
        "matrix_schema": "cvs.phase2.d92_csoas_hard10.matrix.v1",
        "job_receipt_schema": "cvs.phase2.d92_csoas_hard10.job_receipt.v1",
        "experiment_id": "D92-E0-FULL-CSOAS-Hard9-K1-v1",
        "protocol_schema": "p2_min_v1",
        "claim_scope": CLAIM_SCOPE,
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "arms": {ARM_ID: {"candidate_id": CANDIDATE_ID, "role": "primary", "registered_mode": REGISTERED_MODE}},
        "primary_arm": ARM_ID,
        "smoke_outer_key": SMOKE_OUTER_KEY,
        "liveness_outer_key": LIVENESS_OUTER_KEY,
        "excluded_outer_keys": list(EXCLUDED_OUTER_KEYS),
        "registered_mode": REGISTERED_MODE,
        "state_postprocess_mode": STATE_POSTPROCESS_MODE,
        "gate": GATE,
        "deployment_policy": DEPLOYMENT_POLICY,
        "query_contract": QUERY_CONTRACT,
        "matrix": {"outer_count": 10, "performance_outer_count": 9, "liveness_outer_count": 1, "job_count": 10, "scene_count": 3, "scene_arm_count": 30, "shard_count": SHARD_COUNT},
        "fit_gate": FIT_GATE,
        "direction_gate": DIRECTION_GATE,
        "resource_gate": RESOURCE_GATE,
        "historical_baseline": {
            "paired_rows_path": HISTORICAL_BASELINE_PATH,
            "paired_rows_sha256": HISTORICAL_BASELINE_SHA256,
            "per_old_class_rows_path": HISTORICAL_PER_OLD_CLASS_PATH,
            "per_old_class_rows_sha256": HISTORICAL_PER_OLD_CLASS_SHA256,
            "e0_raw_scores": {key: {"path": f"{RAW_SCORE_ROOT}/{key}/E0_FULL_ONLY/scorer/diag_cosine_score.json", "sha256": sha} for key, sha in _RAW_SCORE_SHA.items()},
            "rerun": False,
        },
        "stop_rule": STOP_RULE,
        "fresh_run_retry": False,
        "only_promotion_candidate": ARM_ID,
        "outputs": {"summary": "summary.json", "gates": "gates.json", "paired_rows": "paired_rows.csv", "per_old_class_rows": "per_old_class_rows.csv", "markdown": "analysis.md"},
    }


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, Mapping) or _canonical_bytes(lock) != _canonical_bytes(_expected_lock()):
        raise D92CSOASHard10Error("CSOAS Hard10 method lock identity drift")
    return dict(lock)


def _package_layout(source_job: PurePath, *, require_files: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (pkg, seal) in _base._PACKAGE_LAYOUT.items():
        pkg_path, seal_path = source_job.joinpath(*pkg), source_job.joinpath(*seal)
        if require_files and (not Path(str(pkg_path)).is_dir() or not Path(str(seal_path)).is_file()):
            raise D92CSOASHard10Error(f"sealed source package missing: {name}")
        result[name] = {"package_root": str(pkg_path), "detached_seal_path": str(seal_path), "expected_seal_sha256": _sha256_file(Path(str(seal_path))) if require_files else "0" * 64}
    return result


def _truth_sha(source_job: PurePath, *, require_files: bool) -> str:
    path = Path(str(source_job.joinpath("offline", "scorer", "truth_sidecar.json")))
    if require_files:
        if not path.is_file() or path.is_symlink():
            raise D92CSOASHard10Error("truth sidecar missing")
        return _sha256_file(path)
    return "0" * 64


def validate_hard10_manifest(manifest: Mapping[str, Any], *, expected_method_lock_sha256: str | None = None, require_package_hashes: bool = False) -> dict[str, Any]:
    required = {"schema", "status", "claim_scope", "protocol_schema", "selection_sha256", "context_path", "context_sha256", "method_lock", "method_lock_sha256", "source_d92_output_root", "ground_component_dir", "ground_manifest_path", "ground_manifest_sha256", "output_root", "shard_count", "outer_count", "performance_outer_count", "liveness_outer_count", "job_count", "scene_count", "scene_arm_count", "arms", "candidate_ids", "primary_arm", "smoke_outer_key", "liveness_outer_key", "arm_roles", "coverage", "selected_rows", "jobs"}
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise D92CSOASHard10Error("manifest allowed-key drift")
    expected = {"schema": "cvs.phase2.d92_csoas_hard10.matrix.v1", "status": "FROZEN_DEVELOPMENT_MATRIX", "claim_scope": CLAIM_SCOPE, "protocol_schema": "p2_min_v1", "selection_sha256": CANONICAL_SELECTION_SHA256, "shard_count": 8, "outer_count": 10, "performance_outer_count": 9, "liveness_outer_count": 1, "job_count": 10, "scene_count": 3, "scene_arm_count": 30, "primary_arm": ARM_ID, "smoke_outer_key": SMOKE_OUTER_KEY, "liveness_outer_key": LIVENESS_OUTER_KEY}
    if any(manifest.get(key) != value for key, value in expected.items()) or manifest.get("arms") != [ARM_ID] or manifest.get("candidate_ids") != ARM_CANDIDATE_IDS or manifest.get("arm_roles") != ARM_ROLES:
        raise D92CSOASHard10Error("manifest identity/count drift")
    method_sha = manifest.get("method_lock_sha256")
    if not isinstance(method_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", method_sha.lower()) or (expected_method_lock_sha256 and method_sha.lower() != expected_method_lock_sha256.lower()):
        raise D92CSOASHard10Error("method-lock SHA drift")
    for field in ("context_path", "method_lock", "output_root", "source_d92_output_root", "ground_component_dir", "ground_manifest_path"):
        _pure_path(manifest.get(field))
    if manifest.get("source_d92_output_root") != SOURCE_D92_OUTPUT_ROOT or manifest.get("ground_component_dir") != GROUND_COMPONENT_DIR or manifest.get("ground_manifest_path") != GROUND_MANIFEST_PATH or manifest.get("ground_manifest_sha256") != GROUND_MANIFEST_SHA256:
        raise D92CSOASHard10Error("source identity drift")
    rows = _expected_rows()
    if manifest.get("selected_rows") != rows or manifest.get("coverage") != _coverage(rows):
        raise D92CSOASHard10Error("selected-row/coverage drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 10:
        raise D92CSOASHard10Error("job-count drift")
    job_keys = {"index", "outer_index", "arm_position", "planned_shard_index", "job_id", "outer_key", "outer_role", "hard_score", "receiver", "seed", "k_shot", "new_class_count", "arm_id", "candidate", "role", "primary", "scenarios", "source_job_root", "packages", "truth_sidecar", "truth_sidecar_sha256", "output_root"}
    for index, row in enumerate(rows):
        job = jobs[index]
        expected_job = {"index": index, "outer_index": index, "arm_position": 0, "planned_shard_index": index % SHARD_COUNT, "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}", **row, "arm_id": ARM_ID, "candidate": CANDIDATE_ID, "role": "primary", "primary": True, "scenarios": list(SCENES)}
        if not isinstance(job, Mapping) or set(job) != job_keys or any(job.get(k) != v for k, v in expected_job.items()):
            raise D92CSOASHard10Error("canonical job identity drift")
        if not _path_matches(job.get("output_root"), manifest["output_root"], "jobs", row["outer_key"], ARM_ID) or not _path_matches(job.get("source_job_root"), SOURCE_D92_OUTPUT_ROOT, "jobs", row["outer_key"]):
            raise D92CSOASHard10Error("job path drift")
        source = _pure_path(job["source_job_root"])
        if _pure_path(job.get("truth_sidecar")) != source.joinpath("offline", "scorer", "truth_sidecar.json"):
            raise D92CSOASHard10Error("truth sidecar path drift")
        if not isinstance(job.get("packages"), Mapping) or set(job["packages"]) != set(_base._PACKAGE_LAYOUT):
            raise D92CSOASHard10Error("package identity drift")
        for name, pkg in job["packages"].items():
            expected_pkg, expected_seal = _base._PACKAGE_LAYOUT[name]
            if not isinstance(pkg, Mapping) or set(pkg) != {"package_root", "detached_seal_path", "expected_seal_sha256"} or _pure_path(pkg.get("package_root")) != source.joinpath(*expected_pkg) or _pure_path(pkg.get("detached_seal_path")) != source.joinpath(*expected_seal):
                raise D92CSOASHard10Error("package path drift")
            if require_package_hashes and not re.fullmatch(r"[0-9a-f]{64}", str(pkg.get("expected_seal_sha256", "")).lower()):
                raise D92CSOASHard10Error("package SHA drift")
        if not re.fullmatch(r"[0-9a-f]{64}", str(job.get("truth_sidecar_sha256", "")).lower()):
            raise D92CSOASHard10Error("truth sidecar SHA drift")
    return dict(manifest)


def build_hard10_manifest(*, context_path: str | Path, method_lock_path: str | Path, output_root: str | Path, require_package_files: bool = True) -> dict[str, Any]:
    context_file = Path(context_path)
    lock_file = Path(method_lock_path).resolve(strict=True)
    validate_method_lock(json.loads(lock_file.read_text(encoding="utf-8-sig")))
    context_sha = _sha256_file(context_file) if context_file.is_file() else "0" * 64
    output = _pure_path(str(output_root))
    source = PurePosixPath(SOURCE_D92_OUTPUT_ROOT)
    rows = _expected_rows()
    jobs = []
    for index, row in enumerate(rows):
        source_job = source.joinpath("jobs", row["outer_key"])
        jobs.append({"index": index, "outer_index": index, "arm_position": 0, "planned_shard_index": index % SHARD_COUNT, "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}", **row, "arm_id": ARM_ID, "candidate": CANDIDATE_ID, "role": "primary", "primary": True, "scenarios": list(SCENES), "source_job_root": str(source_job), "packages": _package_layout(source_job, require_files=require_package_files), "truth_sidecar": str(source_job.joinpath("offline", "scorer", "truth_sidecar.json")), "truth_sidecar_sha256": _truth_sha(source_job, require_files=require_package_files), "output_root": str(output.joinpath("jobs", row["outer_key"], ARM_ID))})
    manifest = {"schema": "cvs.phase2.d92_csoas_hard10.matrix.v1", "status": "FROZEN_DEVELOPMENT_MATRIX", "claim_scope": CLAIM_SCOPE, "protocol_schema": "p2_min_v1", "selection_sha256": CANONICAL_SELECTION_SHA256, "context_path": str(context_file.resolve()), "context_sha256": context_sha, "method_lock": str(lock_file), "method_lock_sha256": _sha256_file(lock_file), "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT, "ground_component_dir": GROUND_COMPONENT_DIR, "ground_manifest_path": GROUND_MANIFEST_PATH, "ground_manifest_sha256": GROUND_MANIFEST_SHA256, "output_root": str(output_root), "shard_count": 8, "outer_count": 10, "performance_outer_count": 9, "liveness_outer_count": 1, "job_count": 10, "scene_count": 3, "scene_arm_count": 30, "arms": [ARM_ID], "candidate_ids": dict(ARM_CANDIDATE_IDS), "primary_arm": ARM_ID, "smoke_outer_key": SMOKE_OUTER_KEY, "liveness_outer_key": LIVENESS_OUTER_KEY, "arm_roles": dict(ARM_ROLES), "coverage": _coverage(rows), "selected_rows": rows, "jobs": jobs}
    validate_hard10_manifest(manifest, expected_method_lock_sha256=manifest["method_lock_sha256"], require_package_hashes=require_package_files)
    return manifest


build_csoas_hard10_manifest = build_hard10_manifest
build_hard10_matrix_manifest = build_hard10_manifest
build_hard11_manifest = build_hard10_manifest
validate_hard11_manifest = validate_hard10_manifest
validate_manifest = validate_hard10_manifest


@contextmanager
def _base_context(*, disable_validation: bool = False) -> Iterator[None]:
    names = {
        "ARM_ID": ARM_ID, "CANDIDATE_ID": CANDIDATE_ID, "ARM_ORDER": ARM_ORDER,
        "ARM_CANDIDATE_IDS": ARM_CANDIDATE_IDS, "ARM_ROLES": ARM_ROLES,
        "PRIMARY_ARM": PRIMARY_ARM, "CANONICAL_SELECTION_SHA256": CANONICAL_SELECTION_SHA256,
        "FIT_GATE": FIT_GATE, "RESOURCE_GATE": RESOURCE_GATE, "DEPLOYMENT_POLICY": DEPLOYMENT_POLICY,
        "SMOKE_OUTER_KEY": SMOKE_OUTER_KEY, "LIVENESS_OUTER_KEY": LIVENESS_OUTER_KEY,
        "HARD11_ROWS": HARD10_ROWS, "HARD11_V1_ROWS": HARD10_ROWS,
        "CLAIM_SCOPE": CLAIM_SCOPE, "_expected_lock": _expected_lock,
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


__all__ = [
    "ARM_ID", "ARM_ORDER", "ARM_CANDIDATE_IDS", "ARM_ROLES", "CANDIDATE_ID", "CANONICAL_SELECTION_SHA256",
    "CLAIM_SCOPE", "CONTEXT_SHA256", "DEPLOYMENT_POLICY", "D92CSOASHard10Error", "D92CSOASHard10MatrixError",
    "FIT_GATE", "HARD10_ROWS", "HARD10_V1_ROWS", "HARD11_ROWS", "HARD11_V1_ROWS", "HISTORICAL_BASELINE_PATH",
    "HISTORICAL_BASELINE_SHA256", "HISTORICAL_PER_OLD_CLASS_PATH", "HISTORICAL_PER_OLD_CLASS_SHA256", "LIVENESS_OUTER_KEY",
    "PRIMARY_ARM", "RAW_SCORE_ROOT", "REGISTERED_MODE", "RESOURCE_GATE", "SCENES", "SHARD_COUNT", "SMOKE_OUTER_KEY",
    "SOURCE_D92_OUTPUT_ROOT", "STATE_POSTPROCESS_MODE", "STRICT_PARETO_THRESHOLDS", "QUERY_ZERO_FIELDS", "CSOAS_QUERY_ZERO_FIELDS",
    "GATE", "DIRECTION_GATE", "STOP_RULE", "EXCLUDED_OUTER_KEYS", "G0_OUTER_KEY", "build_hard10_manifest", "build_csoas_hard10_manifest",
    "build_hard10_matrix_manifest", "build_hard11_manifest", "canonical_selection_sha256", "validate_hard10_manifest", "validate_hard11_manifest",
    "validate_manifest", "validate_method_lock",
]
