"""Frozen single-arm D92 ``E0_FULL_ONLY`` Target125 matrix.

The matrix is the complete Cartesian product of the five target receivers, five
seeds, and five registered-state slices.  It intentionally contains no
historical hardness ranking: rows are emitted in deterministic scientific key
order and every row receives the same frozen arm.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


CONTEXT_SHA256 = "067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f"
ARM_ID = "E0_FULL_ONLY"
CANDIDATE_ID = "d92_e0d_e0_full_only"
ARM_ORDER = (ARM_ID,)
ARM_CANDIDATE_IDS = {ARM_ID: CANDIDATE_ID}
ARM_ROLES = {ARM_ID: "primary"}
PRIMARY_ARM = ARM_ID
SMOKE_OUTER_KEY = "rx_20_1__seed_713106__k_1__new_20"
CLAIM_SCOPE = "TARGET125_CONFIRMATION"
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SHARD_COUNT = 8
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713102, 713103, 713104, 713105, 713106)
SLICES = ((1, 20), (5, 20), (10, 5), (10, 10), (10, 20))
SOURCE_D92_OUTPUT_ROOT = "/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720"
GROUND_COMPONENT_DIR = "/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
GROUND_MANIFEST_PATH = f"{GROUND_COMPONENT_DIR}/manifest.json"
GROUND_MANIFEST_SHA256 = "15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c"
OUTER_PATTERN = re.compile(
    r"^rx_(?P<receiver>[0-9_]+)__seed_(?P<seed>[0-9]+)"
    r"__k_(?P<k>[0-9]+)__new_(?P<new>[0-9]+)$"
)

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
_STRICT_GEOMETRY_GATE = {
    "complete_outer_count": 125,
    "k_gt_2_row_count": 100,
    "k_gt_2_mean_delta_h_min_exclusive": 0.0,
    "k_gt_2_delta_h_nonnegative_row_min": 80,
    "all125_mean_delta_old_acc_min": 0.0,
    "all125_mean_delta_old_floor_min": 0.0,
    "all125_mean_delta_seen_new_min": 0.0,
    "all125_mean_delta_forgetting_max": 0.0,
    "k5_k10_two_state_fit": "2/2",
    "k5_k10_after_actual_fit": "1/1",
    "k1_alias_fit": "3/3",
    "wall_comparison": "absolute_only",
    "paired_wall_baseline": None,
}
_STOP_RULE = {
    "same_normalized_exception_fingerprint_distinct_outer_count": 2,
    "pre_prediction_only": True,
    "shared_run_root_ledger": True,
    "fresh_run_retry_authorized": False,
}
_OUTPUTS = {
    "summary": "summary.json",
    "gates": "gates.json",
    "paired_rows": "paired_rows.csv",
    "markdown": "analysis.md",
}


def _outer_key(receiver: str, seed: int, k_shot: int, new_count: int) -> str:
    return (
        f"rx_{str(receiver).replace('-', '_')}__seed_{int(seed)}"
        f"__k_{int(k_shot)}__new_{int(new_count)}"
    )


def _build_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for k_shot, new_count in SLICES:
                rows.append(
                    {
                        "outer_key": _outer_key(receiver, seed, k_shot, new_count),
                        "role": "liveness" if k_shot == 1 else "performance",
                        "receiver": receiver,
                        "seed": int(seed),
                        "k_shot": int(k_shot),
                        "new_class_count": int(new_count),
                        "hard_score": None,
                    }
                )
    return tuple(rows)


TARGET125_ROWS = _build_rows()
TARGET125_V1_ROWS = TARGET125_ROWS


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


SELECTION_PAYLOAD: dict[str, Any] = {
    "schema": "cvs.phase2.d92_e0_full_only_target125.selection.v1",
    "selection_id": "Target125-complete-cartesian",
    "protocol_schema": "p2_min_v1",
    "claim_scope": CLAIM_SCOPE,
    "order": "receiver_ascending_then_seed_ascending_then_frozen_slice_order",
    "receivers": list(RECEIVERS),
    "seeds": list(SEEDS),
    "slices": [{"k_shot": k, "new_class_count": n} for k, n in SLICES],
    "outer_rows": [dict(row) for row in TARGET125_ROWS],
    "coverage": {
        "outer_count": 125,
        "scene_count": 3,
        "scene_row_count": 375,
        "receiver_counts": {receiver: 25 for receiver in RECEIVERS},
        "seed_counts": {str(seed): 25 for seed in SEEDS},
        "slice_counts": {
            "K1_new20": 25,
            "K5_new20": 25,
            "K10_new5": 25,
            "K10_new10": 25,
            "K10_new20": 25,
        },
    },
}
CANONICAL_SELECTION_SHA256 = hashlib.sha256(_canonical_bytes(SELECTION_PAYLOAD)).hexdigest()


class D92E0FullOnlyTarget125Error(ValueError):
    """Raised when the frozen Target125 identity or closure drifts."""


Target125Error = D92E0FullOnlyTarget125Error


def canonical_selection_sha256() -> str:
    return hashlib.sha256(_canonical_bytes(SELECTION_PAYLOAD)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_json_identity(actual: Any, expected: Any) -> bool:
    try:
        return _canonical_bytes(actual) == _canonical_bytes(expected)
    except (TypeError, ValueError):
        return False


def _pure_frozen_path(value: Any) -> PurePath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise D92E0FullOnlyTarget125Error("Target125 manifest path identity drift")
    if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", value) or "\\" in value:
        return PureWindowsPath(value)
    return PurePosixPath(value)


def _path_matches(actual: Any, root: Any, *parts: str) -> bool:
    try:
        actual_path = _pure_frozen_path(actual)
        root_path = _pure_frozen_path(root)
    except D92E0FullOnlyTarget125Error:
        return False
    return type(actual_path) is type(root_path) and actual_path == root_path.joinpath(*parts)


def _parse_outer(outer_key: str) -> tuple[str, int, int, int]:
    match = OUTER_PATTERN.fullmatch(str(outer_key))
    if match is None:
        raise D92E0FullOnlyTarget125Error(f"invalid Target125 outer key: {outer_key}")
    return (
        match.group("receiver").replace("_", "-"),
        int(match.group("seed")),
        int(match.group("k")),
        int(match.group("new")),
    )


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outer_count": len(rows),
        "scene_count": len(SCENES),
        "scene_row_count": len(rows) * len(SCENES),
        "receiver_counts": dict(sorted(Counter(str(row["receiver"]) for row in rows).items())),
        "seed_counts": dict(sorted(Counter(str(row["seed"]) for row in rows).items())),
        "slice_counts": dict(
            sorted(
                Counter(
                    f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}"
                    for row in rows
                ).items()
            )
        ),
    }


def _package_layout(source_job_root: PurePath, *, require_files: bool) -> dict[str, Any]:
    layout = {
        "before_enrollment": (
            ("offline", "predictor", "before", "enrollment_only"),
            ("offline", "seals", "before_enrollment.seal.json"),
        ),
        "before_apply": (
            ("offline", "predictor", "before", "apply_only_staging"),
            ("apply_seals", "before_apply.seal.json"),
        ),
        "after_enrollment": (
            ("offline", "predictor", "after", "enrollment_only"),
            ("offline", "seals", "after_enrollment.seal.json"),
        ),
        "after_apply": (
            ("offline", "predictor", "after", "apply_only_staging"),
            ("apply_seals", "after_apply.seal.json"),
        ),
    }
    result: dict[str, Any] = {}
    for name, (package_parts, seal_parts) in layout.items():
        package_root = source_job_root.joinpath(*package_parts)
        seal_path = source_job_root.joinpath(*seal_parts)
        local_package = Path(str(package_root))
        local_seal = Path(str(seal_path))
        if require_files and (
            not local_package.is_dir()
            or local_package.is_symlink()
            or not local_seal.is_file()
            or local_seal.is_symlink()
        ):
            raise D92E0FullOnlyTarget125Error(f"sealed source package is missing: {name}")
        result[name] = {
            "package_root": str(package_root),
            "detached_seal_path": str(seal_path),
            "expected_seal_sha256": _sha256_file(local_seal) if require_files else None,
        }
    return result


_PACKAGE_LAYOUT = {
    "before_enrollment": (
        ("offline", "predictor", "before", "enrollment_only"),
        ("offline", "seals", "before_enrollment.seal.json"),
    ),
    "before_apply": (
        ("offline", "predictor", "before", "apply_only_staging"),
        ("apply_seals", "before_apply.seal.json"),
    ),
    "after_enrollment": (
        ("offline", "predictor", "after", "enrollment_only"),
        ("offline", "seals", "after_enrollment.seal.json"),
    ),
    "after_apply": (
        ("offline", "predictor", "after", "apply_only_staging"),
        ("apply_seals", "after_apply.seal.json"),
    ),
}
_PACKAGE_KEYS = frozenset({"package_root", "detached_seal_path", "expected_seal_sha256"})
_JOB_KEYS = frozenset(
    {
        "index", "outer_index", "arm_position", "planned_shard_index", "job_id",
        "outer_key", "outer_role", "hard_score", "receiver", "seed", "k_shot",
        "new_class_count", "arm_id", "candidate", "role", "primary", "scenarios",
        "source_job_root", "packages", "truth_sidecar", "output_root",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema", "status", "claim_scope", "protocol_schema", "selection_sha256",
        "context_path", "context_sha256", "method_lock", "method_lock_sha256",
        "source_d92_output_root", "ground_component_dir", "ground_manifest_path",
        "ground_manifest_sha256", "output_root", "shard_count", "outer_count",
        "performance_outer_count", "liveness_outer_count", "job_count", "scene_count",
        "scene_arm_count", "arms", "candidate_ids", "primary_arm", "smoke_outer_key",
        "arm_roles", "coverage", "selected_rows", "jobs",
    }
)


def _expected_method_lock() -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.d92_e0_full_only_target125.method_lock.v1",
        "matrix_schema": "cvs.phase2.d92_e0_full_only_target125.matrix.v1",
        "job_receipt_schema": "cvs.phase2.d92_e0_full_only_target125.job_receipt.v1",
        "experiment_id": "D92-E0-FULL-ONLY-Target125-v1",
        "protocol_schema": "p2_min_v1",
        "claim_scope": CLAIM_SCOPE,
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "arms": {ARM_ID: {"candidate_id": CANDIDATE_ID, "role": "primary"}},
        "primary_arm": ARM_ID,
        "smoke_outer_key": SMOKE_OUTER_KEY,
        "fixed_components": _FIXED_COMPONENTS,
        "fallback": "K1_exact_D92_FULL_alias",
        "query_contract": _QUERY_CONTRACT,
        "matrix": {
            "outer_count": 125,
            "performance_outer_count": 100,
            "liveness_outer_count": 25,
            "job_count": 125,
            "scene_count": 3,
            "scene_arm_count": 375,
            "shard_count": SHARD_COUNT,
        },
        "strict_geometry_gate": _STRICT_GEOMETRY_GATE,
        "stop_rule": _STOP_RULE,
        "fresh_run_retry": False,
        "only_promotion_candidate": ARM_ID,
        "outputs": _OUTPUTS,
    }


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, Mapping) or not _exact_json_identity(lock, _expected_method_lock()):
        raise D92E0FullOnlyTarget125Error("D92-E0-FULL-ONLY Target125 method lock identity drift")
    return dict(lock)


def _expected_selected_rows() -> list[dict[str, Any]]:
    return [
        {
            "outer_key": row["outer_key"],
            "outer_role": row["role"],
            "hard_score": row["hard_score"],
            "receiver": row["receiver"],
            "seed": row["seed"],
            "k_shot": row["k_shot"],
            "new_class_count": row["new_class_count"],
        }
        for row in TARGET125_ROWS
    ]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_target125_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_method_lock_sha256: str | None = None,
    require_package_hashes: bool = False,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise D92E0FullOnlyTarget125Error("Target125 manifest must be an object")
    if set(manifest) != _MANIFEST_KEYS:
        raise D92E0FullOnlyTarget125Error("Target125 manifest allowed-key drift")
    expected_top = {
        "schema": "cvs.phase2.d92_e0_full_only_target125.matrix.v1",
        "status": "FROZEN_TARGET125_CONFIRMATION_MATRIX",
        "claim_scope": CLAIM_SCOPE,
        "protocol_schema": "p2_min_v1",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "context_sha256": CONTEXT_SHA256,
        "shard_count": SHARD_COUNT,
        "outer_count": 125,
        "performance_outer_count": 100,
        "liveness_outer_count": 25,
        "job_count": 125,
        "scene_count": 3,
        "scene_arm_count": 375,
        "primary_arm": ARM_ID,
        "smoke_outer_key": SMOKE_OUTER_KEY,
    }
    if not _exact_json_identity({key: manifest.get(key) for key in expected_top}, expected_top):
        raise D92E0FullOnlyTarget125Error("Target125 manifest identity/count drift")
    if (
        manifest.get("arms") != [ARM_ID]
        or not _exact_json_identity(manifest.get("candidate_ids"), ARM_CANDIDATE_IDS)
        or not _exact_json_identity(manifest.get("arm_roles"), ARM_ROLES)
    ):
        raise D92E0FullOnlyTarget125Error("Target125 manifest arm identity drift")
    method_sha = manifest.get("method_lock_sha256")
    if not _is_sha256(method_sha) or (
        expected_method_lock_sha256 is not None
        and method_sha != str(expected_method_lock_sha256).lower()
    ):
        raise D92E0FullOnlyTarget125Error("Target125 manifest method-lock SHA drift")
    for field in (
        "method_lock", "context_path", "output_root", "source_d92_output_root",
        "ground_component_dir", "ground_manifest_path",
    ):
        _pure_frozen_path(manifest.get(field))
    if (
        not _path_matches(manifest["source_d92_output_root"], SOURCE_D92_OUTPUT_ROOT)
        or not _path_matches(manifest["ground_component_dir"], GROUND_COMPONENT_DIR)
        or not _path_matches(manifest["ground_manifest_path"], GROUND_MANIFEST_PATH)
        or not _path_matches(manifest["ground_manifest_path"], manifest["ground_component_dir"], "manifest.json")
        or manifest.get("ground_manifest_sha256") != GROUND_MANIFEST_SHA256
    ):
        raise D92E0FullOnlyTarget125Error("Target125 manifest source identity drift")
    expected_rows = _expected_selected_rows()
    if not _exact_json_identity(manifest.get("selected_rows"), expected_rows):
        raise D92E0FullOnlyTarget125Error("Target125 manifest selected row identity drift")
    if not _exact_json_identity(manifest.get("coverage"), _coverage(expected_rows)):
        raise D92E0FullOnlyTarget125Error("Target125 manifest coverage drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 125:
        raise D92E0FullOnlyTarget125Error("Target125 manifest job-count drift")
    seen: set[str] = set()
    package_hash_states: set[bool] = set()
    for outer_index, row in enumerate(expected_rows):
        job = jobs[outer_index]
        if not isinstance(job, Mapping) or set(job) != _JOB_KEYS:
            raise D92E0FullOnlyTarget125Error("Target125 manifest job allowed-key drift")
        expected_job = {
            "index": outer_index,
            "outer_index": outer_index,
            "arm_position": 0,
            "planned_shard_index": outer_index % SHARD_COUNT,
            "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}",
            "outer_key": row["outer_key"],
            "outer_role": row["outer_role"],
            "hard_score": row["hard_score"],
            "receiver": row["receiver"],
            "seed": row["seed"],
            "k_shot": row["k_shot"],
            "new_class_count": row["new_class_count"],
            "arm_id": ARM_ID,
            "candidate": CANDIDATE_ID,
            "role": "primary",
            "primary": True,
            "scenarios": list(SCENES),
        }
        if not _exact_json_identity({key: job.get(key) for key in expected_job}, expected_job):
            raise D92E0FullOnlyTarget125Error("Target125 manifest canonical job identity drift")
        if (
            not _path_matches(job.get("output_root"), manifest["output_root"], "jobs", row["outer_key"], ARM_ID)
            or not _path_matches(job.get("source_job_root"), manifest["source_d92_output_root"], "jobs", row["outer_key"])
            or not _path_matches(job.get("truth_sidecar"), job.get("source_job_root"), "offline", "scorer", "truth_sidecar.json")
        ):
            raise D92E0FullOnlyTarget125Error("Target125 manifest canonical job path drift")
        packages = job.get("packages")
        if not isinstance(packages, Mapping) or set(packages) != set(_PACKAGE_LAYOUT):
            raise D92E0FullOnlyTarget125Error("Target125 manifest package identity drift")
        for package_name, (package_parts, seal_parts) in _PACKAGE_LAYOUT.items():
            package = packages[package_name]
            if not isinstance(package, Mapping) or set(package) != _PACKAGE_KEYS:
                raise D92E0FullOnlyTarget125Error("Target125 manifest package key drift")
            package_sha = package.get("expected_seal_sha256")
            if package_sha is None:
                package_hash_states.add(False)
            elif _is_sha256(package_sha):
                package_hash_states.add(True)
            else:
                raise D92E0FullOnlyTarget125Error("Target125 manifest package seal hash drift")
            if (
                not _path_matches(package.get("package_root"), job.get("source_job_root"), *package_parts)
                or not _path_matches(package.get("detached_seal_path"), job.get("source_job_root"), *seal_parts)
            ):
                raise D92E0FullOnlyTarget125Error("Target125 manifest package path drift")
        job_id = str(job["job_id"])
        if job_id in seen:
            raise D92E0FullOnlyTarget125Error("Target125 manifest duplicate job identity")
        seen.add(job_id)
    if len(seen) != 125 or len(package_hash_states) != 1 or (
        require_package_hashes and package_hash_states != {True}
    ):
        raise D92E0FullOnlyTarget125Error("Target125 manifest closure drift")
    return dict(manifest)


def build_target125_manifest(
    *,
    context_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    require_package_files: bool = True,
) -> dict[str, Any]:
    if canonical_selection_sha256() != CANONICAL_SELECTION_SHA256:
        raise D92E0FullOnlyTarget125Error("canonical Target125 selection identity drift")
    context_file = Path(context_path).resolve(strict=True)
    if _sha256_file(context_file) != CONTEXT_SHA256:
        raise D92E0FullOnlyTarget125Error("target125 context SHA drift")
    try:
        context = json.loads(context_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92E0FullOnlyTarget125Error("target125 context JSON drift") from error
    identity = context.get("identity")
    ground_identity = identity.get("ground_component") if isinstance(identity, Mapping) else None
    if (
        context.get("schema") != "cvs.phase2.d108.cbrrc_smme.target125.input_context.v1"
        or context.get("protocol_schema") != "p2_min_v1"
        or not isinstance(context.get("rows"), list)
        or len(context["rows"]) != 125
        or not isinstance(identity, Mapping)
        or not isinstance(ground_identity, Mapping)
        or not _path_matches(identity.get("d92_output_root"), SOURCE_D92_OUTPUT_ROOT)
        or not _path_matches(ground_identity.get("directory"), GROUND_COMPONENT_DIR)
        or not _path_matches(ground_identity.get("manifest_path"), GROUND_MANIFEST_PATH)
        or ground_identity.get("manifest_sha256") != GROUND_MANIFEST_SHA256
    ):
        raise D92E0FullOnlyTarget125Error("target125 context source identity drift")
    context_keys = {
        (str(row["receiver"]), int(row["active_k"]), int(row["seed"]), int(row["new_count"]))
        for row in context["rows"]
    }
    if len(context_keys) != 125:
        raise D92E0FullOnlyTarget125Error("target125 context outer identity duplicated")
    lock_file = Path(method_lock_path).resolve(strict=True)
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92E0FullOnlyTarget125Error("method lock JSON drift") from error
    validate_method_lock(lock)
    method_sha = _sha256_file(lock_file)
    source_root = PurePosixPath(SOURCE_D92_OUTPUT_ROOT)
    output = _pure_frozen_path(str(Path(output_root)))
    selected_rows = _expected_selected_rows()
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        key = (row["receiver"], row["k_shot"], row["seed"], row["new_class_count"])
        if key not in context_keys:
            raise D92E0FullOnlyTarget125Error(f"Target125/context join failed: {row['outer_key']}")
        source_job_root = source_root.joinpath("jobs", row["outer_key"])
        packages = _package_layout(source_job_root, require_files=bool(require_package_files))
        truth_sidecar = source_job_root.joinpath("offline", "scorer", "truth_sidecar.json")
        truth_local = Path(str(truth_sidecar))
        if require_package_files and (not truth_local.is_file() or truth_local.is_symlink()):
            raise D92E0FullOnlyTarget125Error("source truth sidecar is missing")
        jobs.append(
            {
                "index": index,
                "outer_index": index,
                "arm_position": 0,
                "planned_shard_index": index % SHARD_COUNT,
                "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}",
                "outer_key": row["outer_key"],
                "outer_role": row["outer_role"],
                "hard_score": row["hard_score"],
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
                "new_class_count": row["new_class_count"],
                "arm_id": ARM_ID,
                "candidate": CANDIDATE_ID,
                "role": "primary",
                "primary": True,
                "scenarios": list(SCENES),
                "source_job_root": str(source_job_root),
                "packages": packages,
                "truth_sidecar": str(truth_sidecar),
                "output_root": str(output.joinpath("jobs", row["outer_key"], ARM_ID)),
            }
        )
    coverage = _coverage(selected_rows)
    expected_coverage = dict(SELECTION_PAYLOAD["coverage"])
    if coverage != expected_coverage or len(jobs) != 125:
        raise D92E0FullOnlyTarget125Error("Target125 coverage/job-count drift")
    manifest: dict[str, Any] = {
        "schema": "cvs.phase2.d92_e0_full_only_target125.matrix.v1",
        "status": "FROZEN_TARGET125_CONFIRMATION_MATRIX",
        "claim_scope": CLAIM_SCOPE,
        "protocol_schema": "p2_min_v1",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "context_path": str(context_file),
        "context_sha256": CONTEXT_SHA256,
        "method_lock": str(lock_file),
        "method_lock_sha256": method_sha,
        "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT,
        "ground_component_dir": GROUND_COMPONENT_DIR,
        "ground_manifest_path": GROUND_MANIFEST_PATH,
        "ground_manifest_sha256": GROUND_MANIFEST_SHA256,
        "output_root": str(output),
        "shard_count": SHARD_COUNT,
        "outer_count": 125,
        "performance_outer_count": 100,
        "liveness_outer_count": 25,
        "job_count": 125,
        "scene_count": len(SCENES),
        "scene_arm_count": 375,
        "arms": [ARM_ID],
        "candidate_ids": dict(ARM_CANDIDATE_IDS),
        "primary_arm": ARM_ID,
        "smoke_outer_key": SMOKE_OUTER_KEY,
        "arm_roles": dict(ARM_ROLES),
        "coverage": coverage,
        "selected_rows": selected_rows,
        "jobs": jobs,
    }
    validate_target125_manifest(
        manifest,
        expected_method_lock_sha256=method_sha,
        require_package_hashes=bool(require_package_files),
    )
    return manifest


build_target125_matrix_manifest = build_target125_manifest
validate_manifest = validate_target125_manifest


__all__ = [
    "ARM_ID", "ARM_ORDER", "ARM_CANDIDATE_IDS", "ARM_ROLES", "CANDIDATE_ID",
    "CANONICAL_SELECTION_SHA256", "CLAIM_SCOPE", "CONTEXT_SHA256",
    "D92E0FullOnlyTarget125Error", "GROUND_COMPONENT_DIR", "GROUND_MANIFEST_PATH",
    "GROUND_MANIFEST_SHA256", "PRIMARY_ARM", "RECEIVERS", "SCENES", "SEEDS",
    "SELECTION_PAYLOAD", "SHARD_COUNT", "SLICES", "SMOKE_OUTER_KEY",
    "SOURCE_D92_OUTPUT_ROOT", "TARGET125_ROWS", "TARGET125_V1_ROWS",
    "build_target125_manifest", "build_target125_matrix_manifest",
    "canonical_selection_sha256", "validate_method_lock", "validate_manifest",
    "validate_target125_manifest",
]
