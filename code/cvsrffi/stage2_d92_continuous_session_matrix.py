"""Frozen mechanical matrix for the D92 E0 continuous-session screen.

This module owns only the pre-registered Cartesian matrix and its sealed-input
path identities.  It deliberately does not open predictor packages, truth
sidecars, query arrays, or any other experiment payload while building a
manifest.  The later runner/core owns package loading and session execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


MATRIX_SCHEMA = "cvs.phase2.d92_e0_continuous_session.matrix.v1"
METHOD_LOCK_SCHEMA = "cvs.phase2.d92_e0_continuous_session.method_lock.v1"
JOB_RECEIPT_SCHEMA = "cvs.phase2.d92_e0_continuous_session.job_receipt.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
CLAIM_SCOPE = "DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN"
METHOD_ID = "D92_E0_CUMULATIVE_REPLAY_SESSION_V1"
EXPERIMENT_ID = "D92-E0-CONTINUOUS-SESSION-v1"
STATUS = "FROZEN_DEVELOPMENT_ONLY_CONTINUOUS_SESSION_MATRIX"

RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SEED = 713106
K_SHOT = 10
OLD_CLASS_COUNT = 6
NEW_CLASS_COUNT = 5
SCHEDULE_NAMES = (
    "batch_5",
    "singleton_forward",
    "singleton_reverse",
    "chunk_2_2_1",
)
SCHEDULES = {
    "batch_5": (5,),
    "singleton_forward": (1, 1, 1, 1, 1),
    "singleton_reverse": (1, 1, 1, 1, 1),
    "chunk_2_2_1": (2, 2, 1),
}
ARRIVAL_ORDERS = {
    "batch_5": (0, 1, 2, 3, 4),
    "singleton_forward": (0, 1, 2, 3, 4),
    "singleton_reverse": (4, 3, 2, 1, 0),
    "chunk_2_2_1": (0, 1, 2, 3, 4),
}

# These are the existing E0 package-relative paths under each sealed source
# job root.  Keep this four-entry layout in sync with the Target125 builder;
# no new truth or delta package path is invented by the matrix layer.
PACKAGE_LAYOUT = {
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
SOURCE_D92_OUTPUT_ROOT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "d92_registration_balanced_125_retry2_20260720"
)
GROUND_COMPONENT_DIR = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "d19_ciaf_int8_proto_20260717_1039/input/int8_component"
)
GROUND_MANIFEST_PATH = f"{GROUND_COMPONENT_DIR}/manifest.json"
GROUND_MANIFEST_SHA256 = "15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c"
CONTEXT_SHA256 = "067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f"
SHARD_COUNT = 8

FIXED_COMPONENTS = {
    "A": "joint288_z160_fft96_rf32",
    "B": "ground-spectrum_cauchy_robust_center",
    "C": "task_balanced_covariance_0.5_0.5",
    "F": "f0_fp32_weight_fp32_bias",
    "query": "single_f0_all_registered_classes",
}
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
RESOURCE_GATE = {
    "registration_wall_target_ns": 300_000_000,
    "registration_incremental_peak_hard_max_bytes": 4 * 1024 * 1024,
    "query_state_bytes_equal": True,
    "query_macs_equal": True,
}
PACKAGE_KEYS = frozenset(
    {"package_root", "detached_seal_path", "expected_seal_sha256"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContinuousSessionMatrixError(ValueError):
    """Raised when the frozen continuous-session matrix loses identity."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of one local file without opening experiment payloads."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outer_key(receiver: str) -> str:
    return f"rx_{receiver.replace('-', '_')}__seed_{SEED}__k_{K_SHOT}__new_{NEW_CLASS_COUNT}"


def _outer_rows() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "outer_index": index,
            "outer_key": _outer_key(receiver),
            "receiver": receiver,
            "seed": SEED,
            "k_shot": K_SHOT,
            "old_class_count": OLD_CLASS_COUNT,
            "new_class_count": NEW_CLASS_COUNT,
        }
        for index, receiver in enumerate(RECEIVERS)
    )


OUTER_ROWS = _outer_rows()


def _selection_payload() -> dict[str, Any]:
    return {
        "schema": f"{MATRIX_SCHEMA}.selection.v1",
        "selection_id": "D92-E0-continuous-session-5outer-3scene-4schedule",
        "protocol_schema": PROTOCOL_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "order": "receiver_order_then_scene_order_then_schedule_order",
        "receivers": list(RECEIVERS),
        "scenes": list(SCENES),
        "seed": SEED,
        "k_shot": K_SHOT,
        "old_class_count": OLD_CLASS_COUNT,
        "new_class_count": NEW_CLASS_COUNT,
        "schedules": {
            name: {
                "session_increments": list(SCHEDULES[name]),
                "arrival_order": list(ARRIVAL_ORDERS[name]),
            }
            for name in SCHEDULE_NAMES
        },
        "outer_rows": [dict(row) for row in OUTER_ROWS],
    }


SELECTION_PAYLOAD = _selection_payload()
CANONICAL_SELECTION_SHA256 = hashlib.sha256(
    _canonical_bytes(SELECTION_PAYLOAD)
).hexdigest()


def canonical_selection_sha256() -> str:
    return hashlib.sha256(_canonical_bytes(_selection_payload())).hexdigest()


def _package_records(source_job_root: PurePosixPath) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (package_parts, seal_parts) in PACKAGE_LAYOUT.items():
        result[name] = {
            "package_root": str(source_job_root.joinpath(*package_parts)),
            "detached_seal_path": str(source_job_root.joinpath(*seal_parts)),
            # The matrix does not open the remote sidecar to derive a digest.
            "expected_seal_sha256": None,
        }
    return result


def _expected_matrix_counts() -> dict[str, int]:
    return {
        "receiver_count": len(RECEIVERS),
        "scene_count": len(SCENES),
        "schedule_count": len(SCHEDULE_NAMES),
        # One runner job per receiver outer; scenes and schedules are nested
        # so the shared backbone/support feature cache is not recomputed 12x.
        "job_count": len(RECEIVERS),
        "session_fit_count": len(RECEIVERS)
        * len(SCENES)
        * sum(len(SCHEDULES[name]) for name in SCHEDULE_NAMES),
    }


def _expected_lock_fields() -> dict[str, Any]:
    return {
        "schema": METHOD_LOCK_SCHEMA,
        "matrix_schema": MATRIX_SCHEMA,
        "job_receipt_schema": JOB_RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "fixed_components": FIXED_COMPONENTS,
        "query_contract": QUERY_CONTRACT,
        "matrix": _expected_matrix_counts(),
        "receivers": list(RECEIVERS),
        "scenes": list(SCENES),
        "seed": SEED,
        "k_shot": K_SHOT,
        "old_class_count": OLD_CLASS_COUNT,
        "new_class_count": NEW_CLASS_COUNT,
        "schedules": {
            name: {
                "session_increments": list(SCHEDULES[name]),
                "arrival_order": list(ARRIVAL_ORDERS[name]),
            }
            for name in SCHEDULE_NAMES
        },
        "resource_gate": RESOURCE_GATE,
        "sealed_inputs": {
            "context_sha256": CONTEXT_SHA256,
            "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT,
            "ground_component_dir": GROUND_COMPONENT_DIR,
            "ground_manifest_path": GROUND_MANIFEST_PATH,
            "ground_manifest_sha256": GROUND_MANIFEST_SHA256,
            "package_layout": {
                name: {
                    "package_relative_path": list(package_parts),
                    "seal_relative_path": list(seal_parts),
                }
                for name, (package_parts, seal_parts) in PACKAGE_LAYOUT.items()
            },
        },
    }


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact scientific fields consumed by this matrix layer."""

    if not isinstance(lock, Mapping):
        raise ContinuousSessionMatrixError("method lock must be an object")
    expected = _expected_lock_fields()
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ContinuousSessionMatrixError(f"method lock identity drift: {key}")
    if lock.get("selection_sha256") != CANONICAL_SELECTION_SHA256:
        raise ContinuousSessionMatrixError("method lock selection SHA drift")
    if lock.get("development_only") is not True or lock.get("fresh_run_retry") is not False:
        raise ContinuousSessionMatrixError("method lock lifecycle drift")
    return dict(lock)


def _job_rows(output_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    source_root = PurePosixPath(SOURCE_D92_OUTPUT_ROOT)
    index = 0
    for outer in OUTER_ROWS:
        source_job_root = source_root.joinpath("jobs", outer["outer_key"])
        job_id = f"{outer['outer_key']}__continuous_session"
        jobs.append(
            {
                "index": index,
                "outer_index": outer["outer_index"],
                "planned_shard_index": index % SHARD_COUNT,
                "job_id": job_id,
                "outer_key": outer["outer_key"],
                "receiver": outer["receiver"],
                "seed": outer["seed"],
                "k_shot": outer["k_shot"],
                "old_class_count": outer["old_class_count"],
                "new_class_count": outer["new_class_count"],
                "scenes": list(SCENES),
                "schedules": list(SCHEDULE_NAMES),
                "scene_schedule_count": len(SCENES) * len(SCHEDULE_NAMES),
                "method_id": METHOD_ID,
                "role": "development",
                "source_job_root": str(source_job_root),
                "packages": _package_records(source_job_root),
                "output_root": str(output_root / "jobs" / job_id),
            }
        )
        index += 1
    return jobs


_JOB_KEYS = frozenset(
    {
        "index",
        "outer_index",
        "planned_shard_index",
        "job_id",
        "outer_key",
        "receiver",
        "seed",
        "k_shot",
        "old_class_count",
        "new_class_count",
        "scenes",
        "schedules",
        "scene_schedule_count",
        "method_id",
        "role",
        "source_job_root",
        "packages",
        "output_root",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "claim_scope",
        "protocol_schema",
        "method_id",
        "experiment_id",
        "selection_sha256",
        "method_lock",
        "method_lock_sha256",
        "source_d92_output_root",
        "ground_component_dir",
        "ground_manifest_path",
        "ground_manifest_sha256",
        "sealed_inputs",
        "output_root",
        "shard_count",
        "receiver_count",
        "outer_count",
        "scene_count",
        "schedule_count",
        "job_count",
        "session_fit_count",
        "receivers",
        "scenes",
        "seed",
        "k_shot",
        "old_class_count",
        "new_class_count",
        "schedules",
        "arrival_orders",
        "query_contract",
        "resource_gate",
        "jobs",
    }
)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_package(
    package: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    require_hash: bool,
) -> None:
    if set(package) != PACKAGE_KEYS:
        raise ContinuousSessionMatrixError("package allowed-key drift")
    if package.get("package_root") != expected.get("package_root"):
        raise ContinuousSessionMatrixError("package path identity drift")
    if package.get("detached_seal_path") != expected.get("detached_seal_path"):
        raise ContinuousSessionMatrixError("seal path identity drift")
    seal_sha = package.get("expected_seal_sha256")
    if seal_sha is None and not require_hash:
        return
    if not _is_sha256(seal_sha):
        raise ContinuousSessionMatrixError("seal SHA identity drift")


def validate_continuous_session_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_method_lock_sha256: str | None = None,
    require_package_hashes: bool = False,
) -> dict[str, Any]:
    """Validate deterministic matrix closure without opening any package."""

    if not isinstance(manifest, Mapping):
        raise ContinuousSessionMatrixError("continuous-session manifest must be an object")
    if set(manifest) != _MANIFEST_KEYS:
        raise ContinuousSessionMatrixError("manifest allowed-key drift")
    counts = _expected_matrix_counts()
    expected_top = {
        "schema": MATRIX_SCHEMA,
        "status": STATUS,
        "claim_scope": CLAIM_SCOPE,
        "protocol_schema": PROTOCOL_SCHEMA,
        "method_id": METHOD_ID,
        "experiment_id": EXPERIMENT_ID,
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "shard_count": SHARD_COUNT,
        "receiver_count": len(RECEIVERS),
        "outer_count": len(OUTER_ROWS),
        **counts,
        "receivers": list(RECEIVERS),
        "scenes": list(SCENES),
        "seed": SEED,
        "k_shot": K_SHOT,
        "old_class_count": OLD_CLASS_COUNT,
        "new_class_count": NEW_CLASS_COUNT,
        "schedules": list(SCHEDULE_NAMES),
        "arrival_orders": {
            name: list(ARRIVAL_ORDERS[name]) for name in SCHEDULE_NAMES
        },
        "query_contract": QUERY_CONTRACT,
        "resource_gate": RESOURCE_GATE,
    }
    if any(manifest.get(key) != value for key, value in expected_top.items()):
        raise ContinuousSessionMatrixError("manifest identity/count drift")
    method_sha = manifest.get("method_lock_sha256")
    if not _is_sha256(method_sha):
        raise ContinuousSessionMatrixError("method-lock SHA drift")
    if expected_method_lock_sha256 is not None and method_sha != expected_method_lock_sha256.lower():
        raise ContinuousSessionMatrixError("method-lock SHA drift")
    for field, expected in (
        ("source_d92_output_root", SOURCE_D92_OUTPUT_ROOT),
        ("ground_component_dir", GROUND_COMPONENT_DIR),
        ("ground_manifest_path", GROUND_MANIFEST_PATH),
    ):
        if manifest.get(field) != expected:
            raise ContinuousSessionMatrixError(f"{field} identity drift")
    if manifest.get("ground_manifest_sha256") != GROUND_MANIFEST_SHA256:
        raise ContinuousSessionMatrixError("ground manifest SHA drift")
    sealed_inputs = manifest.get("sealed_inputs")
    expected_inputs = _expected_lock_fields()["sealed_inputs"]
    if sealed_inputs != expected_inputs:
        raise ContinuousSessionMatrixError("sealed input identity drift")
    output_root = Path(str(manifest.get("output_root"))).resolve()
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != counts["job_count"]:
        raise ContinuousSessionMatrixError("job count drift")
    expected_jobs = _job_rows(output_root)
    seen: set[str] = set()
    for actual, expected in zip(jobs, expected_jobs):
        if not isinstance(actual, Mapping) or set(actual) != _JOB_KEYS:
            raise ContinuousSessionMatrixError("job allowed-key drift")
        if any(actual.get(key) != expected.get(key) for key in _JOB_KEYS if key != "packages"):
            raise ContinuousSessionMatrixError("job identity/schedule drift")
        if actual["job_id"] in seen:
            raise ContinuousSessionMatrixError("duplicate job identity")
        seen.add(str(actual["job_id"]))
        packages = actual.get("packages")
        if not isinstance(packages, Mapping) or set(packages) != set(PACKAGE_LAYOUT):
            raise ContinuousSessionMatrixError("package layout drift")
        for name in PACKAGE_LAYOUT:
            package = packages[name]
            expected_package = expected["packages"][name]
            if not isinstance(package, Mapping):
                raise ContinuousSessionMatrixError("package record drift")
            _validate_package(
                package,
                expected_package,
                require_hash=require_package_hashes,
            )
    if len(seen) != counts["job_count"]:
        raise ContinuousSessionMatrixError("job identity closure drift")
    return dict(manifest)


def build_continuous_session_manifest(
    *,
    method_lock_path: str | Path,
    output_root: str | Path,
    require_package_files: bool = False,
) -> dict[str, Any]:
    """Build the frozen matrix from metadata only.

    ``require_package_files`` is intentionally unsupported at this layer: the
    existing package/seal files live on N607, and checking them would still be
    a runner concern.  The argument is accepted for compatibility with the
    existing matrix builders; when true, the caller receives a clear error
    instead of silently opening a package or truth sidecar.
    """

    if require_package_files:
        raise ContinuousSessionMatrixError(
            "continuous-session matrix build does not open remote packages; "
            "validate them in the bounded runner"
        )
    lock_file = Path(method_lock_path).resolve(strict=True)
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ContinuousSessionMatrixError("method lock JSON cannot be read") from error
    validate_method_lock(lock)
    output = Path(output_root).resolve()
    manifest: dict[str, Any] = {
        "schema": MATRIX_SCHEMA,
        "status": STATUS,
        "claim_scope": CLAIM_SCOPE,
        "protocol_schema": PROTOCOL_SCHEMA,
        "method_id": METHOD_ID,
        "experiment_id": EXPERIMENT_ID,
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "method_lock": str(lock_file),
        "method_lock_sha256": sha256_file(lock_file),
        "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT,
        "ground_component_dir": GROUND_COMPONENT_DIR,
        "ground_manifest_path": GROUND_MANIFEST_PATH,
        "ground_manifest_sha256": GROUND_MANIFEST_SHA256,
        "sealed_inputs": _expected_lock_fields()["sealed_inputs"],
        "output_root": str(output),
        "shard_count": SHARD_COUNT,
        "outer_count": len(OUTER_ROWS),
        **_expected_matrix_counts(),
        "receivers": list(RECEIVERS),
        "scenes": list(SCENES),
        "seed": SEED,
        "k_shot": K_SHOT,
        "old_class_count": OLD_CLASS_COUNT,
        "new_class_count": NEW_CLASS_COUNT,
        "schedules": list(SCHEDULE_NAMES),
        "arrival_orders": {
            name: list(ARRIVAL_ORDERS[name]) for name in SCHEDULE_NAMES
        },
        "query_contract": dict(QUERY_CONTRACT),
        "resource_gate": dict(RESOURCE_GATE),
        "jobs": _job_rows(output),
    }
    validate_continuous_session_manifest(
        manifest,
        expected_method_lock_sha256=manifest["method_lock_sha256"],
    )
    return manifest


build_matrix_manifest = build_continuous_session_manifest
validate_manifest = validate_continuous_session_manifest


__all__ = [
    "ARRIVAL_ORDERS",
    "CANONICAL_SELECTION_SHA256",
    "CLAIM_SCOPE",
    "CONTEXT_SHA256",
    "ContinuousSessionMatrixError",
    "EXPERIMENT_ID",
    "FIXED_COMPONENTS",
    "GROUND_COMPONENT_DIR",
    "GROUND_MANIFEST_PATH",
    "GROUND_MANIFEST_SHA256",
    "JOB_RECEIPT_SCHEMA",
    "K_SHOT",
    "MATRIX_SCHEMA",
    "METHOD_ID",
    "METHOD_LOCK_SCHEMA",
    "NEW_CLASS_COUNT",
    "OLD_CLASS_COUNT",
    "OUTER_ROWS",
    "PACKAGE_KEYS",
    "PACKAGE_LAYOUT",
    "PROTOCOL_SCHEMA",
    "QUERY_CONTRACT",
    "RECEIVERS",
    "RESOURCE_GATE",
    "SCHEDULE_NAMES",
    "SCHEDULES",
    "SCENES",
    "SEED",
    "SELECTION_PAYLOAD",
    "SHARD_COUNT",
    "SOURCE_D92_OUTPUT_ROOT",
    "build_continuous_session_manifest",
    "build_matrix_manifest",
    "canonical_selection_sha256",
    "sha256_file",
    "validate_continuous_session_manifest",
    "validate_manifest",
    "validate_method_lock",
]
