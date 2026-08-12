"""Frozen D92 D42 Tail-Pair Code Exchange Hard10 mechanical matrix.

This layer owns only matrix identity, immutable package joins and method-lock
validation. The TCRA science implementation is supplied by the paired core;
this module does not tune, score or select from query results.
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

_BASE_EXPECTED_LOCK = _base._expected_lock


ARM_ID = "E0_FULL_D42_TAIL_CLASS_ROW_ASCENT"
CANDIDATE_ID = "d92_e0_full_d42_tail_class_row_ascent"
ARM_ORDER = (ARM_ID,)
ARM_CANDIDATE_IDS = {ARM_ID: CANDIDATE_ID}
ARM_ROLES = {ARM_ID: "primary"}
PRIMARY_ARM = ARM_ID
CLAIM_SCOPE = "DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN"
SHARD_COUNT = _base.SHARD_COUNT
SCENES = _base.SCENES
SMOKE_OUTER_KEY = "rx_7_7__seed_713104__k_5__new_20"
LIVENESS_OUTER_KEY = _base.LIVENESS_OUTER_KEY
G0_OUTER_KEY = "rx_7_7__seed_713106__k_10__new_5"
EXCLUDED_OUTER_KEYS = (G0_OUTER_KEY,)
HARD10_ROWS = (
    {"outer_key": "rx_7_7__seed_713104__k_5__new_20", "role": "performance", "hard_score": None},
    {"outer_key": "rx_7_7__seed_713103__k_10__new_5", "role": "performance", "hard_score": None},
    {"outer_key": "rx_8_8__seed_713103__k_5__new_20", "role": "performance", "hard_score": None},
    {"outer_key": "rx_8_8__seed_713103__k_10__new_5", "role": "performance", "hard_score": None},
    {"outer_key": "rx_8_8__seed_713106__k_5__new_20", "role": "performance", "hard_score": None},
    {"outer_key": "rx_7_14__seed_713104__k_10__new_10", "role": "performance", "hard_score": None},
    {"outer_key": "rx_3_19__seed_713102__k_10__new_5", "role": "performance", "hard_score": None},
    {"outer_key": "rx_7_7__seed_713105__k_10__new_20", "role": "performance", "hard_score": None},
    {"outer_key": "rx_7_7__seed_713104__k_10__new_5", "role": "performance", "hard_score": None},
    {"outer_key": "rx_20_1__seed_713106__k_1__new_20", "role": "liveness", "hard_score": None},
)
HARD10_V1_ROWS = HARD10_ROWS
HARD11_ROWS = HARD10_ROWS
HARD11_V1_ROWS = HARD10_ROWS
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
STRICT_PARETO_THRESHOLDS = {
    "h_old_new": 0.010, "old_balanced_accuracy": 0.015, "c_old_acc": 0.010,
    "old_floor": 0.040, "seen_new_acc": 0.005, "average_forgetting": -0.015,
    "new_to_old_rate": -0.005, "old_to_new_rate": -0.005,
}
FIT_GATE = {"k_gt_2_total": 2, "k_gt_2_actual": 1, "postprocess_fit": 0, "k1_alias": "real_inventory", "k1_total": 3, "k1_actual": 3}
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
    "codec": "post_compile_direct_coef2_qint8_class_row_ascent",
    "selection": "deterministic_class_row_ascent_atomic_subset",
    "full_synchronous_publish": False,
    "quantization": "no_requantize_no_scan",
    "negative_tail_accepted": False,
    "competitor_code_decrement": False,
    "modified_state_fields": ["coef2_qint8"],
}
STATE_POSTPROCESS_MODE = "d42_tcra"
REGISTERED_MODE = "full_only"
QUERY_CONTRACT = {
    "decision": "per_sample_all_registered_classes",
    "truth_access": False, "fit_access": False, "update_access": False,
    "selection_access": False, "role_oracle_access": False,
    "class_quota_access": False, "global_reassignment": False,
}
QUERY_ZERO_FIELDS = (
    "query_truth_access", "query_fit_access", "query_update_access",
    "query_selection_access", "query_role_oracle_access",
    "query_class_quota_access", "query_global_reassignment",
)
DIRECTION_GATE = {
    "h_old_new": {"direction": ">", "magnitude": 0.01},
    "old_balanced_accuracy": {"direction": ">", "magnitude": 0.015},
    "c_old_acc": {"direction": ">", "magnitude": 0.01},
    "old_floor": {"direction": ">", "magnitude": 0.04},
    "seen_new_acc": {"direction": ">", "magnitude": 0.005},
    "average_forgetting": {"direction": "<", "magnitude": -0.015},
    "new_to_old_rate": {"direction": "<", "magnitude": -0.005},
    "old_to_new_rate": {"direction": "<", "magnitude": -0.005},
}
GATE = {
    "revision": "safe_directional_v2", "tol_formula": "1024*float32_eps",
    "tail_quantile": 0.2, "tail_quantile_method": "lower",
    "atomic": "one_true_class_row_ascent_per_class_block",
    "sorting": "uncovered_tail_then_worst_gain_then_total_gain_then_semantic_handle",
    "fallback": "exact_e0_byte_identity",
    "final": "within_tol_nonnegative_with_strict_old_gain_sum",
}
STOP_RULE = {
    "same_normalized_exception_fingerprint_distinct_outer_count": 2,
    "pre_prediction_only": True, "shared_run_root_ledger": True,
    "fresh_run_retry_authorized": False,
}
_RAW_SCORE_SHA = {
    "rx_7_7__seed_713104__k_5__new_20": "492044d89de05fbee79bfd6ca493c51778e2f2b18536038067c329acedd7cee9",
    "rx_7_7__seed_713103__k_10__new_5": "f8f593fe5b26983ae16a7903f3943cd07fb9e0e958beea3c142a724119f7c93b",
    "rx_8_8__seed_713103__k_5__new_20": "00b217da83ffce70655360ce243ad88e37ad1e1a221980488cbb04655b091306",
    "rx_8_8__seed_713103__k_10__new_5": "69ab6c617db8f657c4a21d044049984c5913b5dd0af7a76456877564f031bd32",
    "rx_8_8__seed_713106__k_5__new_20": "9a42a6306669811cda5b058fa342619abf4ef20c01d499fb682d6c4700d5a360",
    "rx_7_14__seed_713104__k_10__new_10": "953e9bccfad63e5e5ca7b7b87e5f48d458318b02c069d4db8c47a5d083087dd0",
    "rx_3_19__seed_713102__k_10__new_5": "6488c4f516e41703cd529d6e4837d0ef0e1fe4eae008fec3beee4cf56cee7bc3",
    "rx_7_7__seed_713105__k_10__new_20": "bee2068990f890cb4233834f1f4ccfb1cfb6d8ed67094d66031c0aa00323712d",
    "rx_7_7__seed_713104__k_10__new_5": "01384fedde246cb017773f69516700bd3bb7a15459b7863d417cc9d2ecc602c1",
    "rx_20_1__seed_713106__k_1__new_20": "bf60d1231127c51b9a9dbe06c9c78bbad7bfd34d0b2ffc5c7809dc94d47677f2",
}


class D92TCRAHard10Error(ValueError):
    """Raised when frozen TCRA Hard10 identity or closure drifts."""


D92TCRAHard10MatrixError = D92TCRAHard10Error


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
        raise D92TCRAHard10Error("path identity drift")
    return PureWindowsPath(value) if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", value) or "\\" in value else PurePosixPath(value)


def _path_matches(actual: Any, root: Any, *parts: str) -> bool:
    try:
        a, r = _pure_path(actual), _pure_path(root)
    except D92TCRAHard10Error:
        return False
    return type(a) is type(r) and a == r.joinpath(*parts)


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outer_count": len(rows), "scene_count": len(SCENES), "scene_row_count": len(rows) * len(SCENES),
        "receiver_counts": dict(sorted(Counter(str(row["receiver"]) for row in rows).items())),
        "seed_counts": dict(sorted(Counter(str(row["seed"]) for row in rows).items())),
        "slice_counts": dict(sorted(Counter(f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}" for row in rows).items())),
        "performance_outer_count": 9, "liveness_outer_count": 1,
    }


_OUTER_PATTERN = re.compile(r"^rx_(?P<receiver>[0-9_]+)__seed_(?P<seed>[0-9]+)__k_(?P<k>[0-9]+)__new_(?P<new>[0-9]+)$")


def _parse_outer(key: str) -> tuple[str, int, int, int]:
    match = _OUTER_PATTERN.fullmatch(str(key))
    if match is None:
        raise D92TCRAHard10Error(f"invalid outer key: {key}")
    return match.group("receiver").replace("_", "-"), int(match.group("seed")), int(match.group("k")), int(match.group("new"))


def _expected_rows() -> list[dict[str, Any]]:
    result = []
    for row in HARD10_ROWS:
        receiver, seed, k_shot, new_count = _parse_outer(row["outer_key"])
        result.append({"outer_key": row["outer_key"], "outer_role": row["role"], "hard_score": None, "receiver": receiver, "seed": seed, "k_shot": k_shot, "new_class_count": new_count})
    return result


def canonical_selection_sha256() -> str:
    payload = {
        "schema": "cvs.phase2.d92_tcra_hard10.selection.v1",
        "selection_id": "D92-E0-FULL-D42-TCRA-Hard10-v1",
        "protocol_schema": "p2_min_v1",
        "claim_scope": CLAIM_SCOPE,
        "order": "explicit_pre_registered_performance_then_k1_liveness",
        "outer_rows": [dict(row) for row in HARD11_ROWS],
        "excluded_outer_keys": ["rx_7_7__seed_713106__k_10__new_5"],
        "coverage": {"outer_count": 10, "performance_outer_count": 9, "liveness_outer_count": 1, "scene_count": 3, "scene_row_count": 30, "shard_count": 8},
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


CANONICAL_SELECTION_SHA256 = canonical_selection_sha256()


def _expected_lock() -> dict[str, Any]:
    return {"schema": "cvs.phase2.d92_tcra_hard10.method_lock.v1", "matrix_schema": "cvs.phase2.d92_tcra_hard10.matrix.v1", "job_receipt_schema": "cvs.phase2.d92_tcra_hard10.job_receipt.v1", "experiment_id": "D92-E0-FULL-D42-TCRA-Hard10-v1", "protocol_schema": "p2_min_v1", "claim_scope": CLAIM_SCOPE, "selection_sha256": CANONICAL_SELECTION_SHA256, "arms": {ARM_ID: {"candidate_id": CANDIDATE_ID, "role": "primary", "registered_mode": REGISTERED_MODE}}, "primary_arm": ARM_ID, "smoke_outer_key": SMOKE_OUTER_KEY, "liveness_outer_key": LIVENESS_OUTER_KEY, "excluded_outer_keys": list(EXCLUDED_OUTER_KEYS), "state_postprocess_mode": STATE_POSTPROCESS_MODE, "gate": GATE, "deployment_policy": DEPLOYMENT_POLICY, "query_contract": QUERY_CONTRACT, "matrix": {"outer_count": 10, "performance_outer_count": 9, "liveness_outer_count": 1, "job_count": 10, "scene_count": 3, "scene_arm_count": 30, "shard_count": SHARD_COUNT}, "fit_gate": FIT_GATE, "direction_gate": DIRECTION_GATE, "resource_gate": RESOURCE_GATE, "historical_baseline": {"paired_rows_path": HISTORICAL_BASELINE_PATH, "paired_rows_sha256": HISTORICAL_BASELINE_SHA256, "per_old_class_rows_path": HISTORICAL_PER_OLD_CLASS_PATH, "per_old_class_rows_sha256": HISTORICAL_PER_OLD_CLASS_SHA256, "e0_raw_scores": {key: {"path": f"{RAW_SCORE_ROOT}/{key}/E0_FULL_ONLY/scorer/diag_cosine_score.json", "sha256": sha} for key, sha in _RAW_SCORE_SHA.items()}, "rerun": False}, "stop_rule": STOP_RULE, "fresh_run_retry": False, "only_promotion_candidate": ARM_ID, "outputs": {"summary": "summary.json", "gates": "gates.json", "paired_rows": "paired_rows.csv", "per_old_class_rows": "per_old_class_rows.csv", "markdown": "analysis.md"}}


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, Mapping) or _canonical_bytes(lock) != _canonical_bytes(_expected_lock()):
        raise D92TCRAHard10Error("D42 TCRA Hard10 method lock identity drift")
    return dict(lock)


@contextmanager
def _base_context(*, disable_validation: bool = False) -> Iterator[None]:
    """Temporarily bind the proven mechanical implementation to TCRA identity."""
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
        "HARD11_ROWS": HARD10_ROWS,
        "HARD11_V1_ROWS": HARD10_ROWS,
        "CLAIM_SCOPE": CLAIM_SCOPE,
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


def validate_hard10_manifest(manifest: Mapping[str, Any], *, expected_method_lock_sha256: str | None = None, require_package_hashes: bool = False) -> dict[str, Any]:
    required = {"schema", "status", "claim_scope", "protocol_schema", "selection_sha256", "context_path", "context_sha256", "method_lock", "method_lock_sha256", "source_d92_output_root", "ground_component_dir", "ground_manifest_path", "ground_manifest_sha256", "output_root", "shard_count", "outer_count", "performance_outer_count", "liveness_outer_count", "job_count", "scene_count", "scene_arm_count", "arms", "candidate_ids", "primary_arm", "smoke_outer_key", "liveness_outer_key", "arm_roles", "coverage", "selected_rows", "jobs"}
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise D92TCRAHard10Error("manifest allowed-key drift")
    expected = {"schema": "cvs.phase2.d92_tcra_hard10.matrix.v1", "status": "FROZEN_DEVELOPMENT_MATRIX", "claim_scope": CLAIM_SCOPE, "protocol_schema": "p2_min_v1", "selection_sha256": CANONICAL_SELECTION_SHA256, "shard_count": 8, "outer_count": 10, "performance_outer_count": 9, "liveness_outer_count": 1, "job_count": 10, "scene_count": 3, "scene_arm_count": 30, "primary_arm": ARM_ID, "smoke_outer_key": SMOKE_OUTER_KEY, "liveness_outer_key": LIVENESS_OUTER_KEY}
    if any(manifest.get(key) != value for key, value in expected.items()) or manifest.get("arms") != [ARM_ID] or manifest.get("candidate_ids") != ARM_CANDIDATE_IDS or manifest.get("arm_roles") != ARM_ROLES:
        raise D92TCRAHard10Error("manifest identity/count drift")
    method_sha = manifest.get("method_lock_sha256")
    if not isinstance(method_sha, str) or len(method_sha) != 64 or any(ch not in "0123456789abcdef" for ch in method_sha.lower()) or (expected_method_lock_sha256 and method_sha.lower() != expected_method_lock_sha256.lower()):
        raise D92TCRAHard10Error("method-lock SHA drift")
    for field in ("context_path", "method_lock", "output_root", "source_d92_output_root", "ground_component_dir", "ground_manifest_path"):
        if not isinstance(manifest.get(field), str) or not manifest.get(field):
            raise D92TCRAHard10Error("path identity drift")
    if manifest.get("source_d92_output_root") != SOURCE_D92_OUTPUT_ROOT or manifest.get("ground_component_dir") != GROUND_COMPONENT_DIR or manifest.get("ground_manifest_path") != GROUND_MANIFEST_PATH or manifest.get("ground_manifest_sha256") != GROUND_MANIFEST_SHA256:
        raise D92TCRAHard10Error("source identity drift")
    rows = _expected_rows()
    if manifest.get("selected_rows") != rows or manifest.get("coverage") != _coverage(rows):
        raise D92TCRAHard10Error("selected-row/coverage drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 10:
        raise D92TCRAHard10Error("job-count drift")
    for index, row in enumerate(rows):
        job = jobs[index]
        expected_job = {"index": index, "outer_index": index, "arm_position": 0, "planned_shard_index": index % SHARD_COUNT, "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}", **row, "arm_id": ARM_ID, "candidate": CANDIDATE_ID, "role": "primary", "primary": True, "scenarios": list(SCENES)}
        if not isinstance(job, Mapping) or any(job.get(k) != value for k, value in expected_job.items()):
            raise D92TCRAHard10Error("canonical job identity drift")
        if not _path_matches(job.get("output_root"), manifest["output_root"], "jobs", row["outer_key"], ARM_ID) or not _path_matches(job.get("source_job_root"), SOURCE_D92_OUTPUT_ROOT, "jobs", row["outer_key"]):
            raise D92TCRAHard10Error("job path drift")
        source = _pure_path(job["source_job_root"])
        if _pure_path(job.get("truth_sidecar")) != source.joinpath("offline", "scorer", "truth_sidecar.json"):
            raise D92TCRAHard10Error("truth sidecar path drift")
        if not isinstance(job.get("packages"), Mapping) or set(job["packages"]) != set(_base._PACKAGE_LAYOUT):
            raise D92TCRAHard10Error("package identity drift")
        for name, pkg in job["packages"].items():
            expected_pkg, expected_seal = _base._PACKAGE_LAYOUT[name]
            if not isinstance(pkg, Mapping) or _pure_path(pkg.get("package_root")) != source.joinpath(*expected_pkg) or _pure_path(pkg.get("detached_seal_path")) != source.joinpath(*expected_seal):
                raise D92TCRAHard10Error("package path drift")
            if require_package_hashes and not isinstance(pkg.get("expected_seal_sha256"), str):
                raise D92TCRAHard10Error("package SHA drift")
        if not isinstance(job.get("truth_sidecar_sha256"), str) or len(job["truth_sidecar_sha256"]) != 64:
            raise D92TCRAHard10Error("truth sidecar SHA drift")
    return dict(manifest)


def build_hard10_manifest(*, context_path: str | Path, method_lock_path: str | Path, output_root: str | Path, require_package_files: bool = True) -> dict[str, Any]:
    context_file = Path(context_path)
    lock_file = Path(method_lock_path).resolve(strict=True)
    validate_method_lock(json.loads(lock_file.read_text(encoding="utf-8-sig")))
    if context_file.is_file():
        with _base_context(disable_validation=True):
            manifest = _base.build_hard11_manifest(context_path=context_file, method_lock_path=lock_file, output_root=output_root, require_package_files=require_package_files)
        manifest["context_path"] = str(context_file.resolve())
        manifest["context_sha256"] = CONTEXT_SHA256
    else:
        output = _pure_path(str(output_root))
        source = PurePosixPath(SOURCE_D92_OUTPUT_ROOT)
        rows = _expected_rows()
        jobs = []
        for index, row in enumerate(rows):
            source_job = source.joinpath("jobs", row["outer_key"])
            packages = {}
            for name, (pkg, seal) in _base._PACKAGE_LAYOUT.items():
                packages[name] = {"package_root": str(source_job.joinpath(*pkg)), "detached_seal_path": str(source_job.joinpath(*seal)), "expected_seal_sha256": "0" * 64}
            jobs.append({"index": index, "outer_index": index, "arm_position": 0, "planned_shard_index": index % SHARD_COUNT, "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}", **row, "arm_id": ARM_ID, "candidate": CANDIDATE_ID, "role": "primary", "primary": True, "scenarios": list(SCENES), "source_job_root": str(source_job), "packages": packages, "truth_sidecar": str(source_job.joinpath("offline", "scorer", "truth_sidecar.json")), "truth_sidecar_sha256": "0" * 64, "output_root": str(output.joinpath("jobs", row["outer_key"], ARM_ID))})
        manifest = {"context_path": str(context_file), "context_sha256": "0" * 64, "method_lock": str(lock_file), "method_lock_sha256": _sha256_file(lock_file), "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT, "ground_component_dir": GROUND_COMPONENT_DIR, "ground_manifest_path": GROUND_MANIFEST_PATH, "ground_manifest_sha256": GROUND_MANIFEST_SHA256, "output_root": str(output), "jobs": jobs}
    manifest.update({"schema": "cvs.phase2.d92_tcra_hard10.matrix.v1", "status": "FROZEN_DEVELOPMENT_MATRIX", "claim_scope": CLAIM_SCOPE, "protocol_schema": "p2_min_v1", "selection_sha256": CANONICAL_SELECTION_SHA256, "shard_count": SHARD_COUNT, "outer_count": 10, "performance_outer_count": 9, "liveness_outer_count": 1, "job_count": 10, "scene_count": len(SCENES), "scene_arm_count": 30, "arms": [ARM_ID], "candidate_ids": dict(ARM_CANDIDATE_IDS), "primary_arm": ARM_ID, "smoke_outer_key": SMOKE_OUTER_KEY, "liveness_outer_key": LIVENESS_OUTER_KEY, "arm_roles": dict(ARM_ROLES), "coverage": _coverage(_expected_rows()), "selected_rows": _expected_rows()})
    validate_hard10_manifest(manifest, expected_method_lock_sha256=manifest["method_lock_sha256"], require_package_hashes=require_package_files)
    return manifest


build_tcra_hard10_manifest = build_hard10_manifest
build_hard10_matrix_manifest = build_hard10_manifest
validate_manifest = validate_hard10_manifest
# Compatibility names consumed only by the proven generic Hard11 runner while
# its globals are bound through _base_context.  The public schema remains TCRA.
build_hard11_manifest = build_hard10_manifest
validate_hard11_manifest = validate_hard10_manifest


__all__ = [
    "ARM_ID", "ARM_ORDER", "ARM_CANDIDATE_IDS", "ARM_ROLES", "CANDIDATE_ID", "CANONICAL_SELECTION_SHA256",
    "CLAIM_SCOPE", "CONTEXT_SHA256", "DEPLOYMENT_POLICY", "D92TCRAHard10Error", "D92TCRAHard10MatrixError",
    "FIT_GATE", "HARD10_ROWS", "HARD10_V1_ROWS", "HARD11_ROWS", "HARD11_V1_ROWS", "HISTORICAL_BASELINE_PATH", "HISTORICAL_BASELINE_SHA256",
    "HISTORICAL_PER_OLD_CLASS_PATH", "HISTORICAL_PER_OLD_CLASS_SHA256", "LIVENESS_OUTER_KEY", "PRIMARY_ARM",
    "RAW_SCORE_ROOT", "REGISTERED_MODE", "RESOURCE_GATE", "SCENES", "SHARD_COUNT", "SMOKE_OUTER_KEY",
    "SOURCE_D92_OUTPUT_ROOT", "STATE_POSTPROCESS_MODE", "STRICT_PARETO_THRESHOLDS", "QUERY_ZERO_FIELDS", "GATE", "DIRECTION_GATE", "STOP_RULE", "EXCLUDED_OUTER_KEYS", "G0_OUTER_KEY", "build_hard10_manifest",
    "build_tcra_hard10_manifest", "build_hard10_matrix_manifest", "canonical_selection_sha256",
    "validate_hard10_manifest", "validate_hard11_manifest", "validate_manifest", "validate_method_lock",
]
