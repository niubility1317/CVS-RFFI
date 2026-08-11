"""Frozen Hard12-v2 matrix for the D92-E0D five-arm stress screen.

This module owns only frozen selection and package/job expansion.  It does not
choose a method or inspect target/query results.  The selection digest is a
design-provided identity; the canonical payload is retained for auditability.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from cvsrffi.stage2_d92_e0d_slim import D92_E0D_ARMS
except ModuleNotFoundError as error:  # temporary interface bridge until the slim module lands
    if error.name != "cvsrffi.stage2_d92_e0d_slim":
        raise
    from dataclasses import dataclass
    from types import MappingProxyType

    @dataclass(frozen=True)
    class _FallbackArm:
        arm_id: str
        candidate_id: str

    D92_E0D_ARMS = MappingProxyType(
        {
            arm_id: _FallbackArm(arm_id, f"d92_e0d_{arm_id.lower()}")
            for arm_id in (
                "D92_FULL",
                "E0_FUSION",
                "E0_FULL_ONLY",
                "E0_BLOCK_ONLY",
                "E0_FIXED50",
            )
        }
    )


CONTEXT_SHA256 = "067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f"
CANONICAL_SELECTION_SHA256 = (
    "2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a"
)
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARM_ORDER = (
    "D92_FULL",
    "E0_FUSION",
    "E0_FULL_ONLY",
    "E0_BLOCK_ONLY",
    "E0_FIXED50",
)
OUTER_PATTERN = re.compile(
    r"^rx_(?P<receiver>[0-9_]+)__seed_(?P<seed>[0-9]+)"
    r"__k_(?P<k>[0-9]+)__new_(?P<new>[0-9]+)$"
)


SELECTION_PAYLOAD: dict[str, Any] = {
    "schema": "cvs.phase2.hard12.selection.v2",
    "selection_id": "Hard12-v2",
    "protocol_schema": "p2_min_v1",
    "claim_scope": "DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN",
    "inputs": [
        {
            "name": "d92_retry2_row_metrics",
            "path": r"E:\type10-7\code\snapshots\d92_125wt\automation_reports\CV-SincNet\d92_registration_balanced_125_20260720\artifacts\retry2\row_metrics.csv",
            "sha256": "bc8070cd9235ab41eda5bafd2ec66e9afad48b6466d2066508d0bab46980fa62",
        },
        {
            "name": "next_r5_r11_score",
            "path": r"E:\type10-7\automation_reports\CV-SincNet\next_r5_fa_rdce3_q_target125_20260805_r11_truth\retrieved\score\score.json",
            "sha256": "fa2344ae037e4ab5dfec6fea9bb0f534c7d5c9cdeb3596797bdc403b3c9fcc23",
            "state": "DA0_REG1",
        },
    ],
    "scenes": list(SCENES),
    "hardness": {
        "rank_population": 125,
        "rank_ties": "average_ascending_rank",
        "pi_down": "(125-rank_avg)/124",
        "pi_up": "(rank_avg-1)/124",
        "d92_component": {
            "aggregation": "mean",
            "pi_down": ["H_old_new", "c_old_acc", "c_old_floor", "seen_new_acc"],
            "pi_up": ["average_forgetting"],
        },
        "r5_component": {
            "aggregation": "mean",
            "scene_aggregation": "equal_mean_over_three_scenes",
            "state": "DA0_REG1",
            "pi_down": ["H_old_new", "old_balanced_accuracy", "old_floor", "seen_new_acc"],
        },
        "combined": "0.5*D92_i+0.5*R5_i",
        "hard_score_representation": "fixed_decimal_12",
    },
    "constraints": {
        "excluded_selection_id": "Hard12-v1",
        "outer_count": 12,
        "receiver_count_range": [2, 3],
        "seed_count_range": [2, 3],
        "liveness": {
            "count": 2,
            "distinct_receiver": True,
            "k_shot": 1,
            "new_class_count": 20,
        },
        "performance_slice_counts": [
            {"count": 3, "k_shot": 5, "new_class_count": 20},
            {"count": 2, "k_shot": 10, "new_class_count": 5},
            {"count": 2, "k_shot": 10, "new_class_count": 10},
            {"count": 3, "k_shot": 10, "new_class_count": 20},
        ],
        "objective": "maximize_sum_historical_hard",
        "tie_break": "outer_key_ascending_with_1e-9_lexicographic_perturbation",
    },
    "excluded_outer_keys": [
        "rx_20_1__seed_713103__k_1__new_20",
        "rx_20_1__seed_713105__k_10__new_20",
        "rx_20_1__seed_713106__k_5__new_20",
        "rx_3_19__seed_713103__k_10__new_10",
        "rx_3_19__seed_713104__k_1__new_20",
        "rx_3_19__seed_713106__k_10__new_20",
        "rx_7_14__seed_713102__k_10__new_20",
        "rx_7_14__seed_713102__k_5__new_20",
        "rx_7_7__seed_713104__k_10__new_5",
        "rx_7_7__seed_713105__k_10__new_5",
        "rx_8_8__seed_713104__k_10__new_10",
        "rx_8_8__seed_713105__k_5__new_20",
    ],
    "outer_rows": [
        {"outer_key": "rx_20_1__seed_713103__k_10__new_20", "role": "performance", "hard_score": "0.554637096774"},
        {"outer_key": "rx_20_1__seed_713103__k_5__new_20", "role": "performance", "hard_score": "0.674294354839"},
        {"outer_key": "rx_20_1__seed_713105__k_5__new_20", "role": "performance", "hard_score": "0.707963709677"},
        {"outer_key": "rx_3_19__seed_713102__k_10__new_20", "role": "performance", "hard_score": "0.686693548387"},
        {"outer_key": "rx_3_19__seed_713106__k_10__new_10", "role": "performance", "hard_score": "0.630947580645"},
        {"outer_key": "rx_3_19__seed_713106__k_10__new_5", "role": "performance", "hard_score": "0.459375000000"},
        {"outer_key": "rx_7_14__seed_713102__k_10__new_5", "role": "performance", "hard_score": "0.243346774194"},
        {"outer_key": "rx_7_14__seed_713103__k_5__new_20", "role": "performance", "hard_score": "0.666129032258"},
        {"outer_key": "rx_7_7__seed_713104__k_1__new_20", "role": "liveness", "hard_score": "0.781653225806"},
        {"outer_key": "rx_7_7__seed_713105__k_10__new_10", "role": "performance", "hard_score": "0.316229838710"},
        {"outer_key": "rx_8_8__seed_713104__k_10__new_20", "role": "performance", "hard_score": "0.499899193548"},
        {"outer_key": "rx_8_8__seed_713104__k_1__new_20", "role": "liveness", "hard_score": "0.854838709677"},
    ],
    "coverage": {
        "outer_count": 12,
        "scene_count": 36,
        "liveness_outer_count": 2,
        "performance_outer_count": 10,
        "v1_intersection_count": 0,
        "receiver_counts": {"20-1": 3, "3-19": 3, "7-14": 2, "7-7": 2, "8-8": 2},
        "seed_counts": {"713102": 2, "713103": 3, "713104": 3, "713105": 2, "713106": 2},
        "slice_counts": {"K1_new20": 2, "K5_new20": 3, "K10_new5": 2, "K10_new10": 2, "K10_new20": 3},
        "historical_hard_sum": "7.076008064516129",
    },
}
HARD12_ROWS = tuple(dict(row) for row in SELECTION_PAYLOAD["outer_rows"])
HARD12_V2_ROWS = HARD12_ROWS


class D92E0DHard12V2Error(ValueError):
    """Raised when the frozen Hard12-v2 matrix cannot be reproduced."""


D92E0DHard12Error = D92E0DHard12V2Error


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_selection_sha256() -> str:
    """Return the canonical SHA256 of the frozen selection payload."""

    return hashlib.sha256(_canonical_bytes(SELECTION_PAYLOAD)).hexdigest()


def _parse_outer(outer_key: str) -> tuple[str, int, int, int]:
    match = OUTER_PATTERN.fullmatch(str(outer_key))
    if match is None:
        raise D92E0DHard12V2Error(f"invalid Hard12-v2 outer key: {outer_key}")
    return (
        match.group("receiver").replace("_", "-"),
        int(match.group("seed")),
        int(match.group("k")),
        int(match.group("new")),
    )


def _package_layout(source_job_root: Path, *, require_files: bool) -> dict[str, Any]:
    paths = {
        "before_enrollment": (
            source_job_root / "offline" / "predictor" / "before" / "enrollment_only",
            source_job_root / "offline" / "seals" / "before_enrollment.seal.json",
        ),
        "before_apply": (
            source_job_root / "offline" / "predictor" / "before" / "apply_only_staging",
            source_job_root / "apply_seals" / "before_apply.seal.json",
        ),
        "after_enrollment": (
            source_job_root / "offline" / "predictor" / "after" / "enrollment_only",
            source_job_root / "offline" / "seals" / "after_enrollment.seal.json",
        ),
        "after_apply": (
            source_job_root / "offline" / "predictor" / "after" / "apply_only_staging",
            source_job_root / "apply_seals" / "after_apply.seal.json",
        ),
    }
    result: dict[str, Any] = {}
    for name, (package_root, seal_path) in paths.items():
        if require_files and (
            not package_root.is_dir()
            or package_root.is_symlink()
            or not seal_path.is_file()
            or seal_path.is_symlink()
        ):
            raise D92E0DHard12V2Error(f"sealed source package is missing: {name}")
        result[name] = {
            "package_root": str(package_root),
            "detached_seal_path": str(seal_path),
            "expected_seal_sha256": _sha256_file(seal_path) if require_files else None,
        }
    return result


def _coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "receiver_counts": dict(sorted(Counter(str(row["receiver"]) for row in rows).items())),
        "seed_counts": dict(sorted(Counter(str(row["seed"]) for row in rows).items())),
        "slice_counts": dict(sorted(Counter(f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}" for row in rows).items())),
    }


def _arm_candidate(arm_id: str) -> str:
    try:
        arm = D92_E0D_ARMS[arm_id]
    except (KeyError, TypeError) as error:
        raise D92E0DHard12V2Error(f"unknown D92-E0D arm: {arm_id}") from error
    if isinstance(arm, Mapping):
        candidate = arm.get("candidate_id") or arm.get("candidate")
    else:
        candidate = getattr(arm, "candidate_id", None) or getattr(arm, "candidate", None)
    if not candidate:
        raise D92E0DHard12V2Error(f"D92-E0D arm has no candidate identity: {arm_id}")
    return str(candidate)


def build_hard12v2_manifest(
    *,
    context_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    require_package_files: bool = True,
) -> dict[str, Any]:
    """Join frozen v2 rows to sealed original D92 packages."""

    if canonical_selection_sha256() != CANONICAL_SELECTION_SHA256:
        raise D92E0DHard12V2Error("canonical Hard12-v2 selection identity drift")
    context_file = Path(context_path).resolve(strict=True)
    if _sha256_file(context_file) != CONTEXT_SHA256:
        raise D92E0DHard12V2Error("target125 context SHA drift")
    context = json.loads(context_file.read_text(encoding="utf-8-sig"))
    if (
        context.get("schema") != "cvs.phase2.d108.cbrrc_smme.target125.input_context.v1"
        or context.get("protocol_schema") != "p2_min_v1"
        or not isinstance(context.get("rows"), list)
        or len(context["rows"]) != 125
    ):
        raise D92E0DHard12V2Error("target125 context schema/count drift")
    context_rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in context["rows"]:
        key = (str(row["receiver"]), int(row["active_k"]), int(row["seed"]), int(row["new_count"]))
        if key in context_rows:
            raise D92E0DHard12V2Error("target125 context outer identity duplicated")
        context_rows[key] = dict(row)
    lock_file = Path(method_lock_path).resolve(strict=True)
    lock = json.loads(lock_file.read_text(encoding="utf-8-sig"))
    if (
        lock.get("schema") != "cvs.phase2.d92_e0d.method_lock.v1"
        or lock.get("protocol_schema") != "p2_min_v1"
        or lock.get("selection_sha256") != CANONICAL_SELECTION_SHA256
    ):
        raise D92E0DHard12V2Error("D92-E0D method lock drift")
    source_root = Path(str(context["identity"]["d92_output_root"]))
    output = Path(output_root)
    selected_rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for outer_index, frozen in enumerate(HARD12_ROWS):
        receiver, seed, k_shot, new_count = _parse_outer(frozen["outer_key"])
        context_key = (receiver, k_shot, seed, new_count)
        if context_key not in context_rows:
            raise D92E0DHard12V2Error(f"Hard12-v2/context join failed: {frozen['outer_key']}")
        selected_rows.append(
            {
                "outer_key": frozen["outer_key"],
                "outer_role": frozen["role"],
                "hard_score": frozen["hard_score"],
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "new_class_count": new_count,
            }
        )
        source_job_root = source_root / "jobs" / str(frozen["outer_key"])
        packages = _package_layout(source_job_root, require_files=bool(require_package_files))
        truth_sidecar = source_job_root / "offline" / "scorer" / "truth_sidecar.json"
        if require_package_files and (not truth_sidecar.is_file() or truth_sidecar.is_symlink()):
            raise D92E0DHard12V2Error("source truth sidecar is missing")
        rotation = outer_index % len(ARM_ORDER)
        arm_order = ARM_ORDER[rotation:] + ARM_ORDER[:rotation]
        for arm_position, arm_id in enumerate(arm_order):
            jobs.append(
                {
                    "index": len(jobs),
                    "outer_index": outer_index,
                    "arm_position": arm_position,
                    "planned_shard_index": outer_index % 8,
                    "job_id": f"{frozen['outer_key']}__arm_{arm_id.lower()}",
                    "outer_key": frozen["outer_key"],
                    "outer_role": frozen["role"],
                    "hard_score": frozen["hard_score"],
                    "receiver": receiver,
                    "seed": seed,
                    "k_shot": k_shot,
                    "new_class_count": new_count,
                    "arm_id": arm_id,
                    "candidate": _arm_candidate(arm_id),
                    "scenarios": list(SCENES),
                    "source_job_root": str(source_job_root),
                    "packages": packages,
                    "truth_sidecar": str(truth_sidecar),
                    "output_root": str(output / "jobs" / str(frozen["outer_key"]) / arm_id),
                }
            )
    expected_coverage = {
        name: dict(SELECTION_PAYLOAD["coverage"][name])
        for name in ("receiver_counts", "seed_counts", "slice_counts")
    }
    coverage = _coverage(selected_rows)
    if coverage != expected_coverage or len(jobs) != 60:
        raise D92E0DHard12V2Error("Hard12-v2 coverage/job-count drift")
    identity = context["identity"]
    return {
        "schema": "cvs.phase2.d92_e0d_hard12v2.matrix.v1",
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "claim_scope": SELECTION_PAYLOAD["claim_scope"],
        "protocol_schema": "p2_min_v1",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "context_path": str(context_file),
        "context_sha256": CONTEXT_SHA256,
        "method_lock": str(lock_file),
        "method_lock_sha256": _sha256_file(lock_file),
        "ground_component_dir": identity["ground_component"]["directory"],
        "ground_manifest_sha256": identity["ground_component"]["manifest_sha256"],
        "output_root": str(output),
        "shard_count": 8,
        "outer_count": len(selected_rows),
        "performance_outer_count": sum(row["outer_role"] == "performance" for row in selected_rows),
        "liveness_outer_count": sum(row["outer_role"] == "liveness" for row in selected_rows),
        "job_count": len(jobs),
        "scene_arm_count": len(jobs) * len(SCENES),
        "coverage": coverage,
        "selected_rows": selected_rows,
        "jobs": jobs,
    }


build_hard12_manifest = build_hard12v2_manifest


__all__ = [
    "ARM_ORDER",
    "CANONICAL_SELECTION_SHA256",
    "CONTEXT_SHA256",
    "D92E0DHard12Error",
    "D92E0DHard12V2Error",
    "D92_E0D_ARMS",
    "HARD12_ROWS",
    "HARD12_V2_ROWS",
    "SCENES",
    "SELECTION_PAYLOAD",
    "build_hard12_manifest",
    "build_hard12v2_manifest",
    "canonical_selection_sha256",
]
