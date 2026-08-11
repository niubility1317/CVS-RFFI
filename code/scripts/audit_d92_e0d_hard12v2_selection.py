#!/usr/bin/env python3
"""Recompute and audit the frozen D92-E0D Hard12-v2 selection.

The audit is a local deterministic check of the two historical source files.
It does not validate Phase2 data, inspect candidate outputs, or add another
runtime/security hash layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import rankdata

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_e0d_hard12 import (
    CANONICAL_SELECTION_SHA256,
    HARD12_ROWS,
    SCENES,
    SELECTION_PAYLOAD,
    canonical_selection_sha256,
)
from cvsrffi.stage2_d92_be_hard12 import HARD12_ROWS as HARD12_V1_ROWS


class D92E0DSelectionAuditError(RuntimeError):
    """Raised when the frozen selection cannot be reproduced exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise D92E0DSelectionAuditError("selection audit output must not overwrite an existing file")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as error:
        raise D92E0DSelectionAuditError(
            "selection audit output must not overwrite an existing file"
        ) from error
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, stat.S_IREAD)


def _source_paths() -> tuple[Path, Path]:
    inputs = SELECTION_PAYLOAD.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise D92E0DSelectionAuditError("selection payload input list drift")
    paths: list[Path] = []
    for item in inputs:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise D92E0DSelectionAuditError("selection payload input path drift")
        paths.append(Path(str(item["path"])).resolve(strict=True))
    return paths[0], paths[1]


def _parse_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    try:
        receiver = str(row["receiver"])
        seed = int(row["seed"])
        k_shot = int(row.get("k_shot", row.get("source_pool_k")))
        new_count = int(row.get("new_class_count", row.get("new_count")))
    except (KeyError, TypeError, ValueError) as error:
        raise D92E0DSelectionAuditError("historical row identity field drift") from error
    return receiver, seed, k_shot, new_count


def _require_finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92E0DSelectionAuditError(f"non-numeric historical metric: {field}") from error
    if not math.isfinite(result):
        raise D92E0DSelectionAuditError(f"non-finite historical metric: {field}")
    return result


def _load_d92(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 125:
        raise D92E0DSelectionAuditError(f"D92 row count drift: {len(rows)}")
    ids = [str(row.get("job_id", "")) for row in rows]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise D92E0DSelectionAuditError("D92 outer identity drift")
    for row in rows:
        _parse_key(row)
        for metric in (
            "h_old_new",
            "c_old_acc",
            "c_old_floor",
            "seen_new_acc",
            "average_forgetting",
        ):
            _require_finite(row.get(metric), field=f"D92.{metric}")
    return rows


def _load_r5(path: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise D92E0DSelectionAuditError("R5 score JSON cannot be read") from error
    rows = payload.get("state_rows")
    if not isinstance(rows, list):
        raise D92E0DSelectionAuditError("R5 state_rows drift")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("state") == "DA0_REG1"]
    if len(selected) != 375:
        raise D92E0DSelectionAuditError(f"R5 DA0_REG1 scene row count drift: {len(selected)}")
    grouped: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[_parse_key(row)].append(row)
    if len(grouped) != 125 or any(len(group) != 3 for group in grouped.values()):
        raise D92E0DSelectionAuditError("R5 DA0_REG1 outer/scene grouping drift")
    result: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for key, group in grouped.items():
        scene_names = {str(row.get("scene")) for row in group}
        if scene_names != set(SCENES):
            raise D92E0DSelectionAuditError(f"R5 scene coverage drift for {key}")
        result[key] = {
            metric: sum(
                _require_finite(row.get(metric), field=f"R5.{metric}") for row in group
            )
            / 3.0
            for metric in (
                "H_old_new",
                "old_balanced_accuracy",
                "old_floor",
                "seen_new_acc",
            )
        }
    return result


def _rank_pi(values: Iterable[float], *, high_is_hard: bool) -> np.ndarray:
    ranks = rankdata(np.asarray(list(values), dtype=np.float64), method="average")
    if high_is_hard:
        return (ranks - 1.0) / 124.0
    return (125.0 - ranks) / 124.0


def _hard_scores(
    d92_rows: list[dict[str, Any]],
    r5_rows: Mapping[tuple[str, int, int, int], Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    d92_low = ("h_old_new", "c_old_acc", "c_old_floor", "seen_new_acc")
    d92_high = ("average_forgetting",)
    r5_low = ("H_old_new", "old_balanced_accuracy", "old_floor", "seen_new_acc")
    d92_pi: dict[str, dict[str, float]] = {}
    for metric in d92_low:
        pi = _rank_pi((_require_finite(row[metric], field=f"D92.{metric}") for row in d92_rows), high_is_hard=False)
        d92_pi[metric] = {str(row["job_id"]): float(value) for row, value in zip(d92_rows, pi)}
    for metric in d92_high:
        pi = _rank_pi((_require_finite(row[metric], field=f"D92.{metric}") for row in d92_rows), high_is_hard=True)
        d92_pi[metric] = {str(row["job_id"]): float(value) for row, value in zip(d92_rows, pi)}
    r5_pi: dict[str, dict[tuple[str, int, int, int], float]] = {}
    for metric in r5_low:
        pi = _rank_pi((float(r5_rows[key][metric]) for key in r5_rows), high_is_hard=False)
        r5_pi[metric] = {key: float(value) for key, value in zip(r5_rows, pi)}
    result: dict[str, dict[str, Any]] = {}
    for row in d92_rows:
        job_id = str(row["job_id"])
        key = _parse_key(row)
        d92_component = sum(d92_pi[metric][job_id] for metric in d92_low + d92_high) / 5.0
        r5_component = sum(r5_pi[metric][key] for metric in r5_low) / 4.0
        hard_score = 0.5 * d92_component + 0.5 * r5_component
        result[job_id] = {
            "outer_key": job_id,
            "key": key,
            "d92_component": d92_component,
            "r5_component": r5_component,
            "hard_score": hard_score,
            "hard_score_fixed": f"{hard_score:.12f}",
        }
    return result


def _constraint_matrix(rows: list[dict[str, Any]], hard_scores: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(rows)
    matrix: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(indices: Iterable[int], lo: float, hi: float) -> None:
        vector = np.zeros(n, dtype=np.float64)
        for index in indices:
            vector[int(index)] = 1.0
        matrix.append(vector)
        lower.append(float(lo))
        upper.append(float(hi))

    add(range(n), 12, 12)
    receivers = sorted({str(row["receiver"]) for row in rows})
    for receiver in receivers:
        add((i for i, row in enumerate(rows) if str(row["receiver"]) == receiver), 2, 3)
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        add((i for i, row in enumerate(rows) if int(row["seed"]) == seed), 2, 3)
    liveness = [
        i
        for i, row in enumerate(rows)
        if int(row["k_shot"]) == 1 and int(row["new_class_count"]) == 20
    ]
    add(liveness, 2, 2)
    for receiver in receivers:
        add(
            (i for i in liveness if str(rows[i]["receiver"]) == receiver),
            0,
            1,
        )
    performance_slices = (
        (5, 20, 3),
        (10, 5, 2),
        (10, 10, 2),
        (10, 20, 3),
    )
    for k_shot, new_count, count in performance_slices:
        add(
            (
                i
                for i, row in enumerate(rows)
                if int(row["k_shot"]) == k_shot
                and int(row["new_class_count"]) == new_count
            ),
            count,
            count,
        )
    excluded = set(SELECTION_PAYLOAD["excluded_outer_keys"])
    allowed_indices = {
        i
        for i, row in enumerate(rows)
        if (
            int(row["k_shot"]) == 1
            and int(row["new_class_count"]) == 20
        )
        or (int(row["k_shot"]), int(row["new_class_count"]))
        in {(5, 20), (10, 5), (10, 10), (10, 20)}
    }
    excluded_indices = {
        i for i, row in enumerate(rows) if str(row["job_id"]) in excluded
    }
    fixed_zero = set(range(n)) - allowed_indices
    fixed_zero.update(excluded_indices)
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(sorted(fixed_zero), dtype=np.int64),
    )


def _solve_selection(rows: list[dict[str, Any]], hard_scores: Mapping[str, Mapping[str, Any]]) -> tuple[list[int], dict[str, Any]]:
    n = len(rows)
    objective = np.asarray(
        [
            -(
                float(hard_scores[str(row["job_id"])]["hard_score"])
                + 1e-9 * float(n - index)
            )
            for index, row in enumerate(sorted(rows, key=lambda value: str(value["job_id"])))
        ],
        dtype=np.float64,
    )
    ordered_rows = sorted(rows, key=lambda value: str(value["job_id"]))
    matrix, lower, upper, fixed_zero = _constraint_matrix(ordered_rows, hard_scores)
    bounds_lower = np.zeros(n, dtype=np.float64)
    bounds_upper = np.ones(n, dtype=np.float64)
    bounds_upper[fixed_zero] = 0.0
    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(bounds_lower, bounds_upper),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise D92E0DSelectionAuditError(f"Hard12-v2 MILP failed: {result.message}")
    selected = [index for index, value in enumerate(result.x) if float(value) > 0.5]
    selected_ids = [str(ordered_rows[index]["job_id"]) for index in selected]
    return selected, {
        "solver": "scipy.optimize.milp",
        "backend": "HiGHS",
        "status": int(result.status),
        "message": str(result.message),
        "tie_perturbation": "1e-9*descending_lexicographic_index",
        "selected_job_ids_sorted": sorted(selected_ids),
    }


def _v1_outer_keys() -> set[str]:
    return {str(row["outer_key"]) for row in HARD12_V1_ROWS}


def audit_selection(output_path: str | Path) -> dict[str, Any]:
    """Recompute all 125 Hard scores and verify the frozen 12-row optimum."""

    if canonical_selection_sha256() != CANONICAL_SELECTION_SHA256:
        raise D92E0DSelectionAuditError("selection payload SHA drift")
    d92_path, r5_path = _source_paths()
    expected_inputs = SELECTION_PAYLOAD["inputs"]
    d92_sha = _sha256_file(d92_path)
    r5_sha = _sha256_file(r5_path)
    if d92_sha != expected_inputs[0]["sha256"] or r5_sha != expected_inputs[1]["sha256"]:
        raise D92E0DSelectionAuditError("historical input SHA drift")
    d92_rows = _load_d92(d92_path)
    r5_rows = _load_r5(r5_path)
    hard_scores = _hard_scores(d92_rows, r5_rows)
    expected_rows = list(HARD12_ROWS)
    expected_ids = [str(row["outer_key"]) for row in expected_rows]
    errors = [
        abs(float(hard_scores[job_id]["hard_score"]) - float(row["hard_score"]))
        for job_id, row in ((str(row["outer_key"]), row) for row in expected_rows)
    ]
    max_error = max(errors)
    if max_error >= 1e-12:
        raise D92E0DSelectionAuditError(f"frozen Hard score drift: {max_error}")
    selected_indices, solver = _solve_selection(d92_rows, hard_scores)
    ordered_ids = list(solver["selected_job_ids_sorted"])
    if ordered_ids != sorted(expected_ids):
        raise D92E0DSelectionAuditError("MILP optimum does not match frozen Hard12-v2 rows")
    selected_rows = [
        {
            "outer_key": str(row["outer_key"]),
            "role": str(row["role"]),
            "hard_score": str(row["hard_score"]),
        }
        for row in expected_rows
    ]
    selected_d92 = [next(row for row in d92_rows if str(row["job_id"]) == key) for key in expected_ids]
    receiver_counts = Counter(str(row["receiver"]) for row in selected_d92)
    seed_counts = Counter(str(row["seed"]) for row in selected_d92)
    slice_counts = Counter(
        f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}" for row in selected_d92
    )
    v1_keys = _v1_outer_keys()
    payload_excluded = {str(value) for value in SELECTION_PAYLOAD["excluded_outer_keys"]}
    if payload_excluded != v1_keys:
        raise D92E0DSelectionAuditError("Hard12-v1 exclusion payload drift")
    coverage = {
        "outer_count": len(selected_rows),
        "scene_count": len(selected_rows) * len(SCENES),
        "liveness_outer_count": sum(row["role"] == "liveness" for row in selected_rows),
        "performance_outer_count": sum(row["role"] == "performance" for row in selected_rows),
        "v1_intersection_count": len(set(expected_ids) & v1_keys),
        "receiver_counts": dict(sorted(receiver_counts.items())),
        "seed_counts": dict(sorted(seed_counts.items())),
        "slice_counts": dict(sorted(slice_counts.items())),
    }
    expected_coverage = {
        key: value for key, value in SELECTION_PAYLOAD["coverage"].items() if key != "historical_hard_sum"
    }
    if coverage != expected_coverage:
        raise D92E0DSelectionAuditError("frozen Hard12-v2 coverage drift")
    historical_sum = sum(float(hard_scores[key]["hard_score"]) for key in expected_ids)
    historical_sum_text = f"{historical_sum:.15f}"
    if historical_sum_text != str(SELECTION_PAYLOAD["coverage"]["historical_hard_sum"]):
        raise D92E0DSelectionAuditError("historical Hard12-v2 objective sum drift")
    receipt: dict[str, Any] = {
        "schema": "cvs.phase2.d92_e0d_hard12v2.selection_audit.v1",
        "status": "SELECTION_AUDIT_PASS",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "input_sha256": {
            "d92_retry2_row_metrics": d92_sha,
            "next_r5_r11_score": r5_sha,
        },
        "d92_outer_count": len(d92_rows),
        "r5_outer_count": len(r5_rows),
        "r5_da0_reg1_scene_count": len(r5_rows) * len(SCENES),
        "hard_score_max_abs_error": max_error,
        "historical_hard_sum": historical_sum_text,
        "selected_outer_count": len(selected_rows),
        "selected_outer_keys": expected_ids,
        "selected_rows": selected_rows,
        "v1_intersection_count": coverage["v1_intersection_count"],
        "coverage": coverage,
        "objective": {
            **solver,
            "historical_hard_sum": historical_sum_text,
            "selected_count": len(selected_indices),
        },
    }
    _write_json_new(Path(output_path), receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-path", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    receipt = audit_selection(args.output_path)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["D92E0DSelectionAuditError", "audit_selection", "main", "parser"]
