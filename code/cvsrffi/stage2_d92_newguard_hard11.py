"""Frozen D92 NewGuard Hard11 mechanical matrix.

This module owns only the immutable single-arm matrix identity and package
joins. The NewGuard scientific implementation is supplied by the Task1 arm;
this wrapper never tunes or selects a method from query/scorer results.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from cvsrffi.stage2_d92_e0_full_only_target125 import (
    CONTEXT_SHA256,
    GROUND_COMPONENT_DIR,
    GROUND_MANIFEST_PATH,
    GROUND_MANIFEST_SHA256,
    SCENES,
    SOURCE_D92_OUTPUT_ROOT,
)


ARM_ID = "E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN"
CANDIDATE_ID = "d92_e0_full_bidirectional_newguard_maxmin"
ARM_ORDER = (ARM_ID,)
ARM_CANDIDATE_IDS = {ARM_ID: CANDIDATE_ID}
ARM_ROLES = {ARM_ID: "primary"}
PRIMARY_ARM = ARM_ID
CLAIM_SCOPE = "DEVELOPMENT_ONLY_HARD_SCREEN"
SHARD_COUNT = 8
SMOKE_OUTER_KEY = "rx_7_7__seed_713106__k_10__new_5"
LIVENESS_OUTER_KEY = "rx_20_1__seed_713106__k_1__new_20"
HISTORICAL_BASELINE_PATH = "E:/type10-7/local_artifacts/d92_e0_full_only_target125_20260812_v1/analysis/paired_rows.csv"
HISTORICAL_BASELINE_SHA256 = "6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a"
HISTORICAL_PER_OLD_CLASS_PATH = "E:/type10-7/local_artifacts/d92_e0_full_only_target125_20260812_v1/analysis/per_old_class_rows.csv"
HISTORICAL_PER_OLD_CLASS_SHA256 = "c0fc1e02b66b01d06da68bdd824594f3281e601d72b32726fa1e97a1e49788e6"
RAW_SCORE_ROOT = "E:/type10-7/local_artifacts/d92_e0_full_only_target125_20260812_v1/output/jobs"

_RAW_SCORE_SHA = {
    "rx_7_7__seed_713106__k_10__new_5": "c4b90161d18482b0eedf978389557871cbf9676197f0a2889d547c95c76fbf97",
    "rx_7_7__seed_713104__k_5__new_20": "492044d89de05fbee79bfd6ca493c51778e2f2b18536038067c329acedd7cee9",
    "rx_7_7__seed_713103__k_10__new_5": "f8f593fe5b26983ae16a7903f3943cd07fb9e0e958beea3c142a724119f7c93b",
    "rx_8_8__seed_713103__k_5__new_20": "00b217da83ffce70655360ce243ad88e37ad1e1a221980488cbb04655b091306",
    "rx_8_8__seed_713103__k_10__new_5": "69ab6c617db8f657c4a21d044049984c5913b5dd0af7a76456877564f031bd32",
    "rx_8_8__seed_713106__k_5__new_20": "9a42a6306669811cda5b058fa342619abf4ef20c01d499fb682d6c4700d5a360",
    "rx_7_14__seed_713104__k_10__new_10": "953e9bccfad63e5e5ca7b7b87e5f48d458318b02c069d4db8c47a5d083087dd0",
    "rx_3_19__seed_713102__k_10__new_5": "6488c4f516e41703cd529d6e4837d0ef0e1fe4eae008fec3beee4cf56cee7bc3",
    "rx_7_7__seed_713105__k_10__new_20": "bee2068990f890cb4233834f1f4ccfb1cfb6d8ed67094d66031c0aa00323712d",
    "rx_7_7__seed_713104__k_10__new_5": "01384fedde246cb017773f69516700bd3bb7a15459b7863d417cc9d2ecc602c1",
    LIVENESS_OUTER_KEY: "bf60d1231127c51b9a9dbe06c9c78bbad7bfd34d0b2ffc5c7809dc94d47677f2",
}

_FROZEN_OUTER_ROWS = (
    ("rx_7_7__seed_713106__k_10__new_5", "performance"),
    ("rx_7_7__seed_713104__k_5__new_20", "performance"),
    ("rx_7_7__seed_713103__k_10__new_5", "performance"),
    ("rx_8_8__seed_713103__k_5__new_20", "performance"),
    ("rx_8_8__seed_713103__k_10__new_5", "performance"),
    ("rx_8_8__seed_713106__k_5__new_20", "performance"),
    ("rx_7_14__seed_713104__k_10__new_10", "performance"),
    ("rx_3_19__seed_713102__k_10__new_5", "performance"),
    ("rx_7_7__seed_713105__k_10__new_20", "performance"),
    ("rx_7_7__seed_713104__k_10__new_5", "performance"),
    (LIVENESS_OUTER_KEY, "liveness"),
)
HARD11_ROWS = tuple({"outer_key": key, "role": role, "hard_score": None} for key, role in _FROZEN_OUTER_ROWS)
HARD11_V1_ROWS = HARD11_ROWS
OUTER_PATTERN = re.compile(r"^rx_(?P<receiver>[0-9_]+)__seed_(?P<seed>[0-9]+)__k_(?P<k>[0-9]+)__new_(?P<new>[0-9]+)$")
_FIXED_COMPONENTS = {
    "A": "joint288_z160_fft96_rf32",
    "B": "ground-spectrum_cauchy_robust_center",
    "C": "task_balanced_covariance_0.5_0.5",
    "F": "f0_fp32_weight_fp32_bias",
    "query": "single_f0_all_registered_classes",
}
_QUERY_CONTRACT = {
    "decision": "per_sample_all_registered_classes",
    "truth_access": False,
    "fit_access": False,
    "update_access": False,
    "selection_access": False,
    "role_oracle_access": False,
    "class_quota_access": False,
    "global_reassignment": False,
}
_MATRIX = {"outer_count": 11, "performance_outer_count": 10, "liveness_outer_count": 1, "job_count": 11, "scene_count": 3, "scene_arm_count": 33, "shard_count": 8}
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
FIT_GATE = {"k_gt_2_total": 2, "k_gt_2_actual": 1, "k1_alias": "real_inventory"}
RESOURCE_GATE = {
    "registration_wall_p90_max_ns": 150_000_000,
    "registration_wall_ratio_max": 1.5,
    "registration_peak_delta_max_bytes": 512 * 1024,
    "query_macs_equal": True,
    "state_bytes_equal": True,
}
STOP_RULE = {"same_normalized_exception_fingerprint_distinct_outer_count": 2, "pre_prediction_only": True, "shared_run_root_ledger": True, "fresh_run_retry_authorized": False}
_OUTPUTS = {"summary": "summary.json", "gates": "gates.json", "paired_rows": "paired_rows.csv", "per_old_class_rows": "per_old_class_rows.csv", "markdown": "analysis.md"}
_PACKAGE_LAYOUT = {
    "before_enrollment": (("offline", "predictor", "before", "enrollment_only"), ("offline", "seals", "before_enrollment.seal.json")),
    "before_apply": (("offline", "predictor", "before", "apply_only_staging"), ("apply_seals", "before_apply.seal.json")),
    "after_enrollment": (("offline", "predictor", "after", "enrollment_only"), ("offline", "seals", "after_enrollment.seal.json")),
    "after_apply": (("offline", "predictor", "after", "apply_only_staging"), ("apply_seals", "after_apply.seal.json")),
}


class D92NewGuardHard11Error(ValueError):
    """Raised when frozen NewGuard Hard11 identity or closure drifts."""


D92NewGuardHard11MatrixError = D92NewGuardHard11Error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pure_path(value: Any) -> PurePath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise D92NewGuardHard11Error("path identity drift")
    return PureWindowsPath(value) if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", value) or "\\" in value else PurePosixPath(value)


def _path_matches(actual: Any, root: Any, *parts: str) -> bool:
    try:
        a, r = _pure_path(actual), _pure_path(root)
    except D92NewGuardHard11Error:
        return False
    return type(a) is type(r) and a == r.joinpath(*parts)


def _parse_outer(key: str) -> tuple[str, int, int, int]:
    match = OUTER_PATTERN.fullmatch(str(key))
    if match is None:
        raise D92NewGuardHard11Error(f"invalid outer key: {key}")
    return match.group("receiver").replace("_", "-"), int(match.group("seed")), int(match.group("k")), int(match.group("new"))


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outer_count": len(rows), "scene_count": 3, "scene_row_count": len(rows) * 3,
        "receiver_counts": dict(sorted(Counter(str(row["receiver"]) for row in rows).items())),
        "seed_counts": dict(sorted(Counter(str(row["seed"]) for row in rows).items())),
        "slice_counts": dict(sorted(Counter(f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}" for row in rows).items())),
        "performance_outer_count": 10, "liveness_outer_count": 1,
    }


def _expected_rows() -> list[dict[str, Any]]:
    result = []
    for row in HARD11_ROWS:
        receiver, seed, k_shot, new_count = _parse_outer(row["outer_key"])
        result.append({"outer_key": row["outer_key"], "outer_role": row["role"], "hard_score": None, "receiver": receiver, "seed": seed, "k_shot": k_shot, "new_class_count": new_count})
    return result


def _package_layout(source_job_root: PurePath, *, require_files: bool) -> dict[str, Any]:
    result = {}
    for name, (pkg, seal) in _PACKAGE_LAYOUT.items():
        pkg_path, seal_path = source_job_root.joinpath(*pkg), source_job_root.joinpath(*seal)
        if require_files and (not Path(str(pkg_path)).is_dir() or not Path(str(seal_path)).is_file()):
            raise D92NewGuardHard11Error(f"sealed source package missing: {name}")
        result[name] = {"package_root": str(pkg_path), "detached_seal_path": str(seal_path), "expected_seal_sha256": _sha256_file(Path(str(seal_path))) if require_files else None}
    return result


def canonical_selection_sha256() -> str:
    payload = {"schema": "cvs.phase2.d92_newguard_hard11.selection.v1", "selection_id": "D92-E0-FULL-BIDIRECTIONAL-NEWGUARD-MAXMIN-Hard11-v1", "protocol_schema": "p2_min_v1", "claim_scope": CLAIM_SCOPE, "order": "explicit_pre_registered_performance_then_k1_liveness", "outer_rows": [dict(row) for row in HARD11_ROWS], "coverage": {"outer_count": 11, "performance_outer_count": 10, "liveness_outer_count": 1, "scene_count": 3, "scene_row_count": 33, "shard_count": 8}}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


CANONICAL_SELECTION_SHA256 = canonical_selection_sha256()


def _expected_lock() -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.d92_newguard_hard11.method_lock.v1", "matrix_schema": "cvs.phase2.d92_newguard_hard11.matrix.v1", "job_receipt_schema": "cvs.phase2.d92_newguard_hard11.job_receipt.v1", "experiment_id": "D92-E0-FULL-BIDIRECTIONAL-NEWGUARD-MAXMIN-Hard11-v1", "protocol_schema": "p2_min_v1", "claim_scope": CLAIM_SCOPE, "selection_sha256": CANONICAL_SELECTION_SHA256,
        "arms": {ARM_ID: {"candidate_id": CANDIDATE_ID, "role": "primary", "registered_mode": "newguard_maxmin"}}, "primary_arm": ARM_ID, "smoke_outer_key": SMOKE_OUTER_KEY, "liveness_outer_key": LIVENESS_OUTER_KEY, "fixed_components": _FIXED_COMPONENTS, "fallback": "K1_K2_exact_D92_FULL_alias", "query_contract": _QUERY_CONTRACT, "matrix": _MATRIX, "fit_gate": FIT_GATE, "strict_pareto_gate": STRICT_PARETO_THRESHOLDS, "resource_gate": RESOURCE_GATE, "historical_baseline": {"paired_rows_path": HISTORICAL_BASELINE_PATH, "paired_rows_sha256": HISTORICAL_BASELINE_SHA256, "per_old_class_rows_path": HISTORICAL_PER_OLD_CLASS_PATH, "per_old_class_rows_sha256": HISTORICAL_PER_OLD_CLASS_SHA256, "e0_raw_scores": {key: {"path": f"{RAW_SCORE_ROOT}/{key}/E0_FULL_ONLY/scorer/diag_cosine_score.json", "sha256": value} for key, value in _RAW_SCORE_SHA.items()}, "rerun": False}, "stop_rule": STOP_RULE, "fresh_run_retry": False, "only_promotion_candidate": ARM_ID, "outputs": _OUTPUTS,
    }


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, Mapping) or _canonical_bytes(lock) != _canonical_bytes(_expected_lock()):
        raise D92NewGuardHard11Error("NewGuard Hard11 method lock identity drift")
    return dict(lock)


def validate_hard11_manifest(manifest: Mapping[str, Any], *, expected_method_lock_sha256: str | None = None, require_package_hashes: bool = False) -> dict[str, Any]:
    required = {"schema", "status", "claim_scope", "protocol_schema", "selection_sha256", "context_path", "context_sha256", "method_lock", "method_lock_sha256", "source_d92_output_root", "ground_component_dir", "ground_manifest_path", "ground_manifest_sha256", "output_root", "shard_count", "outer_count", "performance_outer_count", "liveness_outer_count", "job_count", "scene_count", "scene_arm_count", "arms", "candidate_ids", "primary_arm", "smoke_outer_key", "liveness_outer_key", "arm_roles", "coverage", "selected_rows", "jobs"}
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise D92NewGuardHard11Error("manifest allowed-key drift")
    expected = {"schema": "cvs.phase2.d92_newguard_hard11.matrix.v1", "status": "FROZEN_DEVELOPMENT_MATRIX", "claim_scope": CLAIM_SCOPE, "protocol_schema": "p2_min_v1", "selection_sha256": CANONICAL_SELECTION_SHA256, "context_sha256": CONTEXT_SHA256, "shard_count": 8, "outer_count": 11, "performance_outer_count": 10, "liveness_outer_count": 1, "job_count": 11, "scene_count": 3, "scene_arm_count": 33, "primary_arm": ARM_ID, "smoke_outer_key": SMOKE_OUTER_KEY, "liveness_outer_key": LIVENESS_OUTER_KEY}
    if any(manifest.get(key) != value for key, value in expected.items()) or manifest.get("arms") != [ARM_ID] or manifest.get("candidate_ids") != ARM_CANDIDATE_IDS or manifest.get("arm_roles") != ARM_ROLES:
        raise D92NewGuardHard11Error("manifest identity/count drift")
    method_sha = manifest.get("method_lock_sha256")
    if not isinstance(method_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", method_sha) or (expected_method_lock_sha256 and method_sha.lower() != expected_method_lock_sha256.lower()):
        raise D92NewGuardHard11Error("method-lock SHA drift")
    for field in ("context_path", "method_lock", "output_root", "source_d92_output_root", "ground_component_dir", "ground_manifest_path"):
        _pure_path(manifest.get(field))
    if not _path_matches(manifest["source_d92_output_root"], SOURCE_D92_OUTPUT_ROOT) or not _path_matches(manifest["ground_component_dir"], GROUND_COMPONENT_DIR) or not _path_matches(manifest["ground_manifest_path"], GROUND_MANIFEST_PATH) or manifest.get("ground_manifest_sha256") != GROUND_MANIFEST_SHA256:
        raise D92NewGuardHard11Error("source identity drift")
    rows = _expected_rows()
    if manifest.get("selected_rows") != rows or manifest.get("coverage") != _coverage(rows):
        raise D92NewGuardHard11Error("selected-row/coverage drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 11:
        raise D92NewGuardHard11Error("job-count drift")
    seen = set()
    job_keys = {"index", "outer_index", "arm_position", "planned_shard_index", "job_id", "outer_key", "outer_role", "hard_score", "receiver", "seed", "k_shot", "new_class_count", "arm_id", "candidate", "role", "primary", "scenarios", "source_job_root", "packages", "truth_sidecar", "output_root"}
    for index, row in enumerate(rows):
        job = jobs[index]
        expected_job = {"index": index, "outer_index": index, "arm_position": 0, "planned_shard_index": index % 8, "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}", **row, "arm_id": ARM_ID, "candidate": CANDIDATE_ID, "role": "primary", "primary": True, "scenarios": list(SCENES)}
        if not isinstance(job, Mapping) or set(job) != job_keys or any(job.get(k) != v for k, v in expected_job.items()):
            raise D92NewGuardHard11Error("canonical job identity drift")
        if not _path_matches(job.get("output_root"), manifest["output_root"], "jobs", row["outer_key"], ARM_ID) or not _path_matches(job.get("source_job_root"), manifest["source_d92_output_root"], "jobs", row["outer_key"]):
            raise D92NewGuardHard11Error("job path drift")
        expected_truth = _pure_path(job["source_job_root"]).joinpath("offline", "scorer", "truth_sidecar.json")
        if _pure_path(job.get("truth_sidecar")) != expected_truth:
            raise D92NewGuardHard11Error("truth sidecar path drift")
        if not isinstance(job.get("packages"), Mapping) or set(job["packages"]) != {"before_enrollment", "before_apply", "after_enrollment", "after_apply"}:
            raise D92NewGuardHard11Error("package identity drift")
        source_job_root = _pure_path(job["source_job_root"])
        for name, pkg in job["packages"].items():
            if set(pkg) != {"package_root", "detached_seal_path", "expected_seal_sha256"}:
                raise D92NewGuardHard11Error("package key drift")
            expected_pkg, expected_seal = _PACKAGE_LAYOUT[name]
            if _pure_path(pkg["package_root"]) != source_job_root.joinpath(*expected_pkg) or _pure_path(pkg["detached_seal_path"]) != source_job_root.joinpath(*expected_seal):
                raise D92NewGuardHard11Error("package path drift")
            if require_package_hashes and not isinstance(pkg.get("expected_seal_sha256"), str):
                raise D92NewGuardHard11Error("package SHA missing")
            if require_package_hashes and not re.fullmatch(r"[0-9a-f]{64}", str(pkg.get("expected_seal_sha256"))):
                raise D92NewGuardHard11Error("package SHA format drift")
        if job["job_id"] in seen:
            raise D92NewGuardHard11Error("duplicate job identity")
        seen.add(job["job_id"])
    return dict(manifest)


def build_hard11_manifest(*, context_path: str | Path, method_lock_path: str | Path, output_root: str | Path, require_package_files: bool = True) -> dict[str, Any]:
    context_file = Path(context_path).resolve(strict=True)
    if _sha256_file(context_file) != CONTEXT_SHA256:
        raise D92NewGuardHard11Error("target125 context SHA drift")
    context = json.loads(context_file.read_text(encoding="utf-8-sig"))
    if context.get("protocol_schema") != "p2_min_v1" or not isinstance(context.get("rows"), list) or len(context["rows"]) != 125:
        raise D92NewGuardHard11Error("target125 context identity drift")
    lock_file = Path(method_lock_path).resolve(strict=True)
    validate_method_lock(json.loads(lock_file.read_text(encoding="utf-8-sig")))
    source_root = PurePosixPath(SOURCE_D92_OUTPUT_ROOT)
    output = _pure_path(str(Path(output_root)))
    jobs = []
    for index, row in enumerate(_expected_rows()):
        source_job_root = source_root.joinpath("jobs", row["outer_key"])
        truth_sidecar = source_job_root.joinpath("offline", "scorer", "truth_sidecar.json")
        jobs.append({"index": index, "outer_index": index, "arm_position": 0, "planned_shard_index": index % 8, "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}", **row, "arm_id": ARM_ID, "candidate": CANDIDATE_ID, "role": "primary", "primary": True, "scenarios": list(SCENES), "source_job_root": str(source_job_root), "packages": _package_layout(source_job_root, require_files=require_package_files), "truth_sidecar": str(truth_sidecar), "output_root": str(output.joinpath("jobs", row["outer_key"], ARM_ID))})
    manifest = {"schema": "cvs.phase2.d92_newguard_hard11.matrix.v1", "status": "FROZEN_DEVELOPMENT_MATRIX", "claim_scope": CLAIM_SCOPE, "protocol_schema": "p2_min_v1", "selection_sha256": CANONICAL_SELECTION_SHA256, "context_path": str(context_file), "context_sha256": CONTEXT_SHA256, "method_lock": str(lock_file), "method_lock_sha256": _sha256_file(lock_file), "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT, "ground_component_dir": GROUND_COMPONENT_DIR, "ground_manifest_path": GROUND_MANIFEST_PATH, "ground_manifest_sha256": GROUND_MANIFEST_SHA256, "output_root": str(output), "shard_count": 8, "outer_count": 11, "performance_outer_count": 10, "liveness_outer_count": 1, "job_count": 11, "scene_count": 3, "scene_arm_count": 33, "arms": [ARM_ID], "candidate_ids": dict(ARM_CANDIDATE_IDS), "primary_arm": ARM_ID, "smoke_outer_key": SMOKE_OUTER_KEY, "liveness_outer_key": LIVENESS_OUTER_KEY, "arm_roles": dict(ARM_ROLES), "coverage": _coverage(_expected_rows()), "selected_rows": _expected_rows(), "jobs": jobs}
    validate_hard11_manifest(manifest, expected_method_lock_sha256=manifest["method_lock_sha256"], require_package_hashes=require_package_files)
    return manifest


build_newguard_hard11_manifest = build_hard11_manifest
build_hard11_matrix_manifest = build_hard11_manifest
validate_manifest = validate_hard11_manifest

__all__ = ["ARM_ID", "ARM_ORDER", "ARM_CANDIDATE_IDS", "ARM_ROLES", "CANDIDATE_ID", "CANONICAL_SELECTION_SHA256", "CLAIM_SCOPE", "CONTEXT_SHA256", "D92NewGuardHard11Error", "D92NewGuardHard11MatrixError", "FIT_GATE", "HARD11_ROWS", "HARD11_V1_ROWS", "HISTORICAL_BASELINE_PATH", "HISTORICAL_BASELINE_SHA256", "HISTORICAL_PER_OLD_CLASS_PATH", "HISTORICAL_PER_OLD_CLASS_SHA256", "LIVENESS_OUTER_KEY", "PRIMARY_ARM", "RAW_SCORE_ROOT", "SCENES", "SHARD_COUNT", "SMOKE_OUTER_KEY", "SOURCE_D92_OUTPUT_ROOT", "STRICT_PARETO_THRESHOLDS", "RESOURCE_GATE", "build_hard11_manifest", "build_newguard_hard11_manifest", "build_hard11_matrix_manifest", "canonical_selection_sha256", "validate_hard11_manifest", "validate_manifest", "validate_method_lock"]
