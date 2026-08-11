"""Frozen Hard12-v3 matrix for the D92-E0OCF five-arm stress screen.

The module only expands the pre-registered rows into immutable jobs.  It does
not choose an arm from results and it reuses the sealed package layout owned by
the established E0D runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from cvsrffi.stage2_d92_be_hard12 import HARD12_ROWS as HARD12_V1_ROWS
from cvsrffi.stage2_d92_e0d_hard12 import (
    CONTEXT_SHA256,
    SCENES,
    SELECTION_PAYLOAD as E0D_SELECTION_PAYLOAD,
)
from cvsrffi.stage2_d92_e0d_slim import D92_E0D_ARMS
from cvsrffi.stage2_d92_e0d_hard12 import HARD12_ROWS as HARD12_V2_ROWS


ARM_ORDER = (
    "D92_FULL",
    "E0_FULL_ONLY",
    "E0_FIXED50",
    "E0_OCF25",
    "E0_OCF50",
)
PRIMARY_ARM = "E0_OCF25"
DIAGNOSTIC_ONLY_ARM = "E0_OCF50"
SMOKE_OUTER_KEY = "rx_20_1__seed_713106__k_1__new_20"
CLAIM_SCOPE = "DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN"
ARM_CANDIDATE_IDS = {
    "D92_FULL": "d92_e0d_d92_full",
    "E0_FULL_ONLY": "d92_e0d_e0_full_only",
    "E0_FIXED50": "d92_e0d_e0_fixed50",
    "E0_OCF25": "d92_e0ocf_e0_ocf25",
    "E0_OCF50": "d92_e0ocf_e0_ocf50",
}
ARM_ROLES = {
    "D92_FULL": "baseline",
    "E0_FULL_ONLY": "baseline",
    "E0_FIXED50": "baseline",
    "E0_OCF25": "primary",
    "E0_OCF50": "diagnostic_only",
}
_METHOD_LOCK_ARMS = {
    "D92_FULL": {"candidate_id": "d92_e0d_d92_full", "role": "baseline"},
    "E0_FULL_ONLY": {
        "candidate_id": "d92_e0d_e0_full_only",
        "role": "baseline",
    },
    "E0_FIXED50": {"candidate_id": "d92_e0d_e0_fixed50", "role": "baseline"},
    "E0_OCF25": {
        "candidate_id": "d92_e0ocf_e0_ocf25",
        "role": "primary",
        "lambda": 0.25,
    },
    "E0_OCF50": {
        "candidate_id": "d92_e0ocf_e0_ocf50",
        "role": "diagnostic_only",
        "lambda": 0.50,
    },
}
_MATRIX_COUNTS = {
    "outer_count": 12,
    "performance_outer_count": 10,
    "liveness_outer_count": 2,
    "job_count": 60,
    "scene_arm_count": 180,
    "scene_count": 3,
    "shard_count": 8,
}
_STRICT_GEOMETRY_GATE = {
    "mean_delta_old_floor_vs_full_only_min_exclusive": 0.0,
    "old_floor_nonnegative_vs_full_only_min": 8,
    "mean_delta_h_vs_full_only_min": 0.0,
    "mean_delta_old_balanced_vs_full_only_min": 0.0,
    "mean_delta_seen_new_vs_full_only_min": 0.0,
    "mean_delta_forgetting_vs_full_only_max": 0.0,
    "mean_delta_h_vs_d92_full_min": 0.005,
    "h_nonnegative_vs_d92_full_min": 8,
    "mean_delta_old_balanced_vs_d92_full_min": 0.0,
    "mean_delta_old_floor_vs_d92_full_min": 0.0,
    "mean_delta_seen_new_vs_d92_full_min": 0.0,
    "mean_delta_forgetting_vs_d92_full_max": 0.0,
    "median_wall_reduction_vs_d92_full_min": 0.6,
    "median_incremental_peak_vs_d92_full_max": 0.0,
    "query_macs_equal_full_only": True,
    "state_bytes_equal_full_only": True,
    "ocf_k5_k10_two_state_fit": "4/4",
    "ocf_k5_k10_after_actual_fit": "2/2",
}
OUTER_PATTERN = re.compile(
    r"^rx_(?P<receiver>[0-9_]+)__seed_(?P<seed>[0-9]+)"
    r"__k_(?P<k>[0-9]+)__new_(?P<new>[0-9]+)$"
)

_FROZEN_OUTER_ROWS = (
    ("rx_20_1__seed_713104__k_5__new_20", "performance", "0.629334677419"),
    ("rx_20_1__seed_713106__k_10__new_20", "performance", "0.520866935484"),
    ("rx_20_1__seed_713106__k_1__new_20", "liveness", "0.910584677419"),
    ("rx_3_19__seed_713102__k_10__new_5", "performance", "0.429435483871"),
    ("rx_3_19__seed_713103__k_10__new_20", "performance", "0.720463709677"),
    ("rx_3_19__seed_713105__k_10__new_5", "performance", "0.454032258065"),
    ("rx_7_14__seed_713102__k_10__new_10", "performance", "0.412600806452"),
    ("rx_7_14__seed_713105__k_1__new_20", "liveness", "0.875403225806"),
    ("rx_7_7__seed_713104__k_10__new_10", "performance", "0.297479838710"),
    ("rx_7_7__seed_713106__k_5__new_20", "performance", "0.521471774194"),
    ("rx_8_8__seed_713103__k_10__new_20", "performance", "0.456451612903"),
    ("rx_8_8__seed_713104__k_5__new_20", "performance", "0.590826612903"),
)
HARD12_ROWS = tuple(
    {"outer_key": key, "role": role, "hard_score": score}
    for key, role, score in _FROZEN_OUTER_ROWS
)
HARD12_V3_ROWS = HARD12_ROWS
HARD12_V1_ROWS = tuple(dict(row) for row in HARD12_V1_ROWS)
HARD12_V2_ROWS = tuple(dict(row) for row in HARD12_V2_ROWS)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exact_json_identity(actual: Any, expected: Any) -> bool:
    try:
        return json.dumps(
            actual,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) == json.dumps(
            expected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False


def _historical_exclusions() -> tuple[str, ...]:
    v1 = {str(row["outer_key"]) for row in HARD12_V1_ROWS}
    v2 = {str(row["outer_key"]) for row in HARD12_V2_ROWS}
    selected = {str(row["outer_key"]) for row in HARD12_ROWS}
    # Hard12-v3 must be disjoint from both historical selections.  Keep the
    # check at import time so an accidental historical drift fails closed.
    if selected & (v1 | v2):
        raise D92E0OCFHard12V3Error("Hard12-v3 intersects historical Hard12 rows")
    return tuple(sorted(v1 | v2))


SELECTION_PAYLOAD: dict[str, Any] = copy.deepcopy(E0D_SELECTION_PAYLOAD)
SELECTION_PAYLOAD.update(
    {
        "schema": "cvs.phase2.d92_e0ocf_hard12v3.selection.v1",
        "selection_id": "Hard12-v3",
        "outer_rows": [dict(row) for row in HARD12_ROWS],
        "excluded_outer_keys": list(_historical_exclusions()),
        "coverage": {
            "outer_count": 12,
            "scene_count": 36,
            "liveness_outer_count": 2,
            "performance_outer_count": 10,
            "v1_intersection_count": 0,
            "v2_intersection_count": 0,
            "receiver_counts": {"20-1": 3, "3-19": 3, "7-14": 2, "7-7": 2, "8-8": 2},
            "seed_counts": {"713102": 2, "713103": 2, "713104": 3, "713105": 2, "713106": 3},
            "slice_counts": {"K1_new20": 2, "K5_new20": 3, "K10_new5": 2, "K10_new10": 2, "K10_new20": 3},
            "historical_hard_sum": "6.818951612903226",
        },
        "constraints": {
            "excluded_selection_ids": ["Hard12-v1", "Hard12-v2"],
            "outer_count": 12,
            "receiver_count_range": [2, 3],
            "seed_count_range": [2, 3],
            "objective": "maximize_sum_historical_hard",
            "tie_break": "outer_key_ascending_with_1e-9_lexicographic_perturbation",
            "liveness": {"count": 2, "distinct_receiver": True, "k_shot": 1, "new_class_count": 20},
            "performance_slice_counts": [
                {"count": 3, "k_shot": 5, "new_class_count": 20},
                {"count": 2, "k_shot": 10, "new_class_count": 5},
                {"count": 2, "k_shot": 10, "new_class_count": 10},
                {"count": 3, "k_shot": 10, "new_class_count": 20},
            ],
        },
        "frozen_matrix": {"outer_count": 12, "job_count": 60, "scene_count": 3, "scene_arm_count": 180, "shard_count": 8},
        "arms": list(ARM_ORDER),
        "primary_arm": PRIMARY_ARM,
        "diagnostic_only_arm": DIAGNOSTIC_ONLY_ARM,
    }
)


class D92E0OCFHard12V3Error(ValueError):
    """Raised when the frozen Hard12-v3 matrix cannot be reproduced."""


D92E0OCFHard12Error = D92E0OCFHard12V3Error
EXCLUDED_OUTER_KEYS = tuple(_historical_exclusions())
CANONICAL_SELECTION_SHA256 = hashlib.sha256(_canonical_bytes(SELECTION_PAYLOAD)).hexdigest()


def canonical_selection_sha256() -> str:
    return hashlib.sha256(_canonical_bytes(SELECTION_PAYLOAD)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_outer(outer_key: str) -> tuple[str, int, int, int]:
    match = OUTER_PATTERN.fullmatch(str(outer_key))
    if match is None:
        raise D92E0OCFHard12V3Error(f"invalid Hard12-v3 outer key: {outer_key}")
    return (
        match.group("receiver").replace("_", "-"),
        int(match.group("seed")),
        int(match.group("k")),
        int(match.group("new")),
    )


def _coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "receiver_counts": dict(sorted(Counter(str(row["receiver"]) for row in rows).items())),
        "seed_counts": dict(sorted(Counter(str(row["seed"]) for row in rows).items())),
        "slice_counts": dict(sorted(Counter(f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}" for row in rows).items())),
    }


def _package_layout(source_job_root: Path, *, require_files: bool) -> dict[str, Any]:
    paths = {
        "before_enrollment": (source_job_root / "offline" / "predictor" / "before" / "enrollment_only", source_job_root / "offline" / "seals" / "before_enrollment.seal.json"),
        "before_apply": (source_job_root / "offline" / "predictor" / "before" / "apply_only_staging", source_job_root / "apply_seals" / "before_apply.seal.json"),
        "after_enrollment": (source_job_root / "offline" / "predictor" / "after" / "enrollment_only", source_job_root / "offline" / "seals" / "after_enrollment.seal.json"),
        "after_apply": (source_job_root / "offline" / "predictor" / "after" / "apply_only_staging", source_job_root / "apply_seals" / "after_apply.seal.json"),
    }
    result: dict[str, Any] = {}
    for name, (package_root, seal_path) in paths.items():
        if require_files and (
            not package_root.is_dir() or package_root.is_symlink() or not seal_path.is_file() or seal_path.is_symlink()
        ):
            raise D92E0OCFHard12V3Error(f"sealed source package is missing: {name}")
        result[name] = {
            "package_root": str(package_root),
            "detached_seal_path": str(seal_path),
            "expected_seal_sha256": _sha256_file(seal_path) if require_files else None,
        }
    return result


def _arm_candidate(arm_id: str) -> str:
    if arm_id not in ARM_ORDER:
        raise D92E0OCFHard12V3Error(f"unknown Hard12-v3 arm: {arm_id}")
    try:
        candidate_id = str(D92_E0D_ARMS[arm_id].candidate_id)
    except KeyError as error:
        raise D92E0OCFHard12V3Error(f"D92-E0D arm has no candidate identity: {arm_id}") from error
    if candidate_id != ARM_CANDIDATE_IDS[arm_id]:
        raise D92E0OCFHard12V3Error(f"D92-E0D candidate identity drift: {arm_id}")
    return candidate_id


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _expected_selected_rows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for frozen in HARD12_ROWS:
        receiver, seed, k_shot, new_count = _parse_outer(str(frozen["outer_key"]))
        result.append(
            {
                "outer_key": str(frozen["outer_key"]),
                "outer_role": str(frozen["role"]),
                "hard_score": str(frozen["hard_score"]),
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "new_class_count": new_count,
            }
        )
    return result


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every frozen scientific identity in the E0OCF method lock."""

    if not isinstance(lock, Mapping):
        raise D92E0OCFHard12V3Error("D92-E0OCF method lock must be an object")
    expected_top = {
        "schema": "cvs.phase2.d92_e0ocf.method_lock.v1",
        "matrix_schema": "cvs.phase2.d92_e0ocf_hard12v3.matrix.v1",
        "job_receipt_schema": "cvs.phase2.d92_e0ocf_hard12v3.job_receipt.v1",
        "protocol_schema": "p2_min_v1",
        "claim_scope": CLAIM_SCOPE,
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "primary_arm": PRIMARY_ARM,
        "diagnostic_only_arm": DIAGNOSTIC_ONLY_ARM,
        "only_promotion_candidate": PRIMARY_ARM,
        "smoke_outer_key": SMOKE_OUTER_KEY,
    }
    if any(lock.get(key) != value for key, value in expected_top.items()):
        raise D92E0OCFHard12V3Error("D92-E0OCF method lock identity drift")
    if not _exact_json_identity(lock.get("arms"), _METHOD_LOCK_ARMS):
        raise D92E0OCFHard12V3Error("D92-E0OCF method lock arm identity drift")
    if not _exact_json_identity(lock.get("matrix"), _MATRIX_COUNTS):
        raise D92E0OCFHard12V3Error("D92-E0OCF method lock matrix identity drift")
    if not _exact_json_identity(
        lock.get("strict_geometry_gate"), _STRICT_GEOMETRY_GATE
    ):
        raise D92E0OCFHard12V3Error("D92-E0OCF method lock strict gate identity drift")
    return dict(lock)


def validate_hard12v3_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_method_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the complete canonical Hard12-v3 matrix identity."""

    if not isinstance(manifest, Mapping):
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest must be an object")
    expected_top = {
        "schema": "cvs.phase2.d92_e0ocf_hard12v3.matrix.v1",
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "claim_scope": CLAIM_SCOPE,
        "protocol_schema": "p2_min_v1",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "context_sha256": CONTEXT_SHA256,
        "shard_count": 8,
        "outer_count": 12,
        "performance_outer_count": 10,
        "liveness_outer_count": 2,
        "job_count": 60,
        "scene_count": 3,
        "scene_arm_count": 180,
        "primary_arm": PRIMARY_ARM,
        "diagnostic_only_arm": DIAGNOSTIC_ONLY_ARM,
        "smoke_outer_key": SMOKE_OUTER_KEY,
    }
    if not _exact_json_identity(
        {key: manifest.get(key) for key in expected_top}, expected_top
    ):
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest identity/count drift")
    if (
        manifest.get("arms") != list(ARM_ORDER)
        or not _exact_json_identity(manifest.get("candidate_ids"), ARM_CANDIDATE_IDS)
        or not _exact_json_identity(manifest.get("arm_roles"), ARM_ROLES)
    ):
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest arm identity drift")
    method_lock_sha256 = manifest.get("method_lock_sha256")
    if not _is_sha256(method_lock_sha256):
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest method-lock SHA drift")
    if (
        expected_method_lock_sha256 is not None
        and method_lock_sha256 != str(expected_method_lock_sha256).lower()
    ):
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest method-lock SHA drift")
    if (
        not isinstance(manifest.get("method_lock"), str)
        or not str(manifest["method_lock"])
        or not isinstance(manifest.get("context_path"), str)
        or not str(manifest["context_path"])
        or not isinstance(manifest.get("output_root"), str)
        or not str(manifest["output_root"])
        or not isinstance(manifest.get("ground_component_dir"), str)
        or not str(manifest["ground_component_dir"])
        or not _is_sha256(manifest.get("ground_manifest_sha256"))
    ):
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest path/hash identity drift")

    expected_rows = _expected_selected_rows()
    if not _exact_json_identity(manifest.get("selected_rows"), expected_rows):
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest selected outer identity drift")
    expected_coverage = _coverage(expected_rows)
    if not _exact_json_identity(manifest.get("coverage"), expected_coverage):
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest coverage drift")

    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 60:
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest job-count drift")
    output_root = Path(str(manifest["output_root"]))
    seen_job_ids: set[str] = set()
    for outer_index, row in enumerate(expected_rows):
        rotation = outer_index % len(ARM_ORDER)
        arm_order = ARM_ORDER[rotation:] + ARM_ORDER[:rotation]
        for arm_position, arm_id in enumerate(arm_order):
            job_index = outer_index * len(ARM_ORDER) + arm_position
            job = jobs[job_index]
            if not isinstance(job, Mapping):
                raise D92E0OCFHard12V3Error("Hard12-v3 manifest job identity drift")
            job_id = f"{row['outer_key']}__arm_{arm_id.lower()}"
            expected_job = {
                "index": job_index,
                "outer_index": outer_index,
                "arm_position": arm_position,
                "planned_shard_index": outer_index % 8,
                "job_id": job_id,
                "outer_key": row["outer_key"],
                "outer_role": row["outer_role"],
                "hard_score": row["hard_score"],
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
                "new_class_count": row["new_class_count"],
                "arm_id": arm_id,
                "candidate": ARM_CANDIDATE_IDS[arm_id],
                "role": ARM_ROLES[arm_id],
                "primary": arm_id == PRIMARY_ARM,
                "diagnostic_only": arm_id == DIAGNOSTIC_ONLY_ARM,
                "scenarios": list(SCENES),
                "output_root": str(output_root / "jobs" / row["outer_key"] / arm_id),
            }
            if not _exact_json_identity(
                {key: job.get(key) for key in expected_job}, expected_job
            ):
                raise D92E0OCFHard12V3Error("Hard12-v3 manifest canonical job identity drift")
            if job_id in seen_job_ids:
                raise D92E0OCFHard12V3Error("Hard12-v3 manifest duplicate job identity")
            seen_job_ids.add(job_id)
    if len(seen_job_ids) != 60:
        raise D92E0OCFHard12V3Error("Hard12-v3 manifest job closure drift")
    return dict(manifest)


def build_hard12v3_manifest(
    *,
    context_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    require_package_files: bool = True,
) -> dict[str, Any]:
    """Join frozen v3 rows to the original sealed D92 packages."""

    if canonical_selection_sha256() != CANONICAL_SELECTION_SHA256:
        raise D92E0OCFHard12V3Error("canonical Hard12-v3 selection identity drift")
    context_file = Path(context_path).resolve(strict=True)
    if _sha256_file(context_file) != CONTEXT_SHA256:
        raise D92E0OCFHard12V3Error("target125 context SHA drift")
    context = json.loads(context_file.read_text(encoding="utf-8-sig"))
    if context.get("schema") != "cvs.phase2.d108.cbrrc_smme.target125.input_context.v1" or context.get("protocol_schema") != "p2_min_v1" or not isinstance(context.get("rows"), list) or len(context["rows"]) != 125:
        raise D92E0OCFHard12V3Error("target125 context schema/count drift")
    context_rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in context["rows"]:
        key = (str(row["receiver"]), int(row["active_k"]), int(row["seed"]), int(row["new_count"]))
        if key in context_rows:
            raise D92E0OCFHard12V3Error("target125 context outer identity duplicated")
        context_rows[key] = dict(row)
    lock_file = Path(method_lock_path).resolve(strict=True)
    lock = json.loads(lock_file.read_text(encoding="utf-8-sig"))
    validate_method_lock(lock)
    method_lock_sha256 = _sha256_file(lock_file)
    smoke_rows = []
    for row in HARD12_ROWS:
        if str(row.get("outer_key")) != SMOKE_OUTER_KEY:
            continue
        try:
            _, _, row_k, _ = _parse_outer(str(row["outer_key"]))
        except (KeyError, TypeError, ValueError) as error:
            raise D92E0OCFHard12V3Error("smoke outer identity drift") from error
        if row.get("role") == "liveness" and row_k == 1:
            smoke_rows.append(row)
    if len(smoke_rows) != 1:
        raise D92E0OCFHard12V3Error("smoke outer must identify exactly one frozen liveness K1 outer")
    source_root = Path(str(context["identity"]["d92_output_root"]))
    output = Path(output_root)
    selected_rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for outer_index, frozen in enumerate(HARD12_ROWS):
        receiver, seed, k_shot, new_count = _parse_outer(frozen["outer_key"])
        if (receiver, k_shot, seed, new_count) not in context_rows:
            raise D92E0OCFHard12V3Error(f"Hard12-v3/context join failed: {frozen['outer_key']}")
        selected_rows.append({"outer_key": frozen["outer_key"], "outer_role": frozen["role"], "hard_score": frozen["hard_score"], "receiver": receiver, "seed": seed, "k_shot": k_shot, "new_class_count": new_count})
        source_job_root = source_root / "jobs" / frozen["outer_key"]
        packages = _package_layout(source_job_root, require_files=bool(require_package_files))
        truth_sidecar = source_job_root / "offline" / "scorer" / "truth_sidecar.json"
        if require_package_files and (not truth_sidecar.is_file() or truth_sidecar.is_symlink()):
            raise D92E0OCFHard12V3Error("source truth sidecar is missing")
        rotation = outer_index % len(ARM_ORDER)
        arm_order = ARM_ORDER[rotation:] + ARM_ORDER[:rotation]
        for arm_position, arm_id in enumerate(arm_order):
            arm_role = "primary" if arm_id == PRIMARY_ARM else ("diagnostic_only" if arm_id == DIAGNOSTIC_ONLY_ARM else "baseline")
            jobs.append({
                "index": len(jobs), "outer_index": outer_index, "arm_position": arm_position, "planned_shard_index": outer_index % 8,
                "job_id": f"{frozen['outer_key']}__arm_{arm_id.lower()}", "outer_key": frozen["outer_key"], "outer_role": frozen["role"], "hard_score": frozen["hard_score"],
                "receiver": receiver, "seed": seed, "k_shot": k_shot, "new_class_count": new_count, "arm_id": arm_id, "candidate": _arm_candidate(arm_id),
                "role": arm_role, "primary": arm_id == PRIMARY_ARM, "diagnostic_only": arm_id == DIAGNOSTIC_ONLY_ARM, "scenarios": list(SCENES),
                "source_job_root": str(source_job_root), "packages": packages, "truth_sidecar": str(truth_sidecar), "output_root": str(output / "jobs" / frozen["outer_key"] / arm_id),
            })
    coverage = _coverage(selected_rows)
    expected = {name: dict(SELECTION_PAYLOAD["coverage"][name]) for name in ("receiver_counts", "seed_counts", "slice_counts")}
    if coverage != expected or len(jobs) != 60:
        raise D92E0OCFHard12V3Error("Hard12-v3 coverage/job-count drift")
    identity = context["identity"]
    manifest = {
        "schema": "cvs.phase2.d92_e0ocf_hard12v3.matrix.v1", "status": "FROZEN_DEVELOPMENT_MATRIX", "claim_scope": SELECTION_PAYLOAD["claim_scope"],
        "protocol_schema": "p2_min_v1", "selection_sha256": CANONICAL_SELECTION_SHA256, "context_path": str(context_file), "context_sha256": CONTEXT_SHA256,
        "method_lock": str(lock_file), "method_lock_sha256": method_lock_sha256, "ground_component_dir": identity["ground_component"]["directory"], "ground_manifest_sha256": identity["ground_component"]["manifest_sha256"],
        "output_root": str(output), "shard_count": 8, "outer_count": len(selected_rows), "performance_outer_count": sum(row["outer_role"] == "performance" for row in selected_rows), "liveness_outer_count": sum(row["outer_role"] == "liveness" for row in selected_rows),
        "job_count": len(jobs), "scene_count": len(SCENES), "scene_arm_count": len(jobs) * len(SCENES), "arms": list(ARM_ORDER), "candidate_ids": {arm: _arm_candidate(arm) for arm in ARM_ORDER}, "primary_arm": PRIMARY_ARM, "diagnostic_only_arm": DIAGNOSTIC_ONLY_ARM, "smoke_outer_key": SMOKE_OUTER_KEY,
        "arm_roles": dict(ARM_ROLES), "coverage": coverage, "selected_rows": selected_rows, "jobs": jobs,
    }
    validate_hard12v3_manifest(
        manifest,
        expected_method_lock_sha256=method_lock_sha256,
    )
    return manifest


build_hard12_manifest = build_hard12v3_manifest


__all__ = [
    "ARM_CANDIDATE_IDS", "ARM_ORDER", "ARM_ROLES", "CANONICAL_SELECTION_SHA256", "CLAIM_SCOPE", "CONTEXT_SHA256", "D92E0OCFHard12Error", "D92E0OCFHard12V3Error", "DIAGNOSTIC_ONLY_ARM", "EXCLUDED_OUTER_KEYS", "HARD12_ROWS", "HARD12_V1_ROWS", "HARD12_V2_ROWS", "HARD12_V3_ROWS", "PRIMARY_ARM", "SCENES", "SELECTION_PAYLOAD", "SMOKE_OUTER_KEY", "build_hard12_manifest", "build_hard12v3_manifest", "canonical_selection_sha256", "validate_hard12v3_manifest", "validate_method_lock", "_canonical_bytes",
]
