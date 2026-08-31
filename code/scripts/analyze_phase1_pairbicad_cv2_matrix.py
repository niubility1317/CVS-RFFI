#!/usr/bin/env python3
"""Analyze the frozen PairBiCAD-CV2 source-only matrix.

This wrapper reuses the existing row-level closure validator and adds only the
CV2-specific static candidate schedule, S_DG aggregation, and same-row gate
analysis.  A complete but weak row is a negative scientific result, not a
technical failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
for import_path in (str(CODE_ROOT), str(SCRIPT_DIR)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import analyze_phase1_pairbicad_matrix as base_analyzer
from cvsrffi.phase1_bicad_xr.config import (
    CV2_CANDIDATE_IDS,
    candidate_config,
    method_lock_payload,
)
from cvsrffi.phase1_bicad_xr.metrics import (
    aggregate_s_dg,
    compute_s_dg,
    evaluate_candidate_gate,
    validate_four_scenario_closure,
)


MatrixAnalysisError = base_analyzer.MatrixAnalysisError
SCENARIOS = base_analyzer.SCENARIOS
DEFAULT_FOLDS: tuple[int, int] = (1, 8)
DEFAULT_SEEDS: tuple[int] = (392002,)

CONTROL_FOR: dict[str, str] = {
    "CV2-B2": "CV2-B1",
    "CV2-B3": "CV2-B1",
    "CV2-D1": "CV2-D0",
    "CV2-D2": "CV2-D0",
    "CV2-D3": "CV2-D0",
    "CV2-T1": "CV2-T0",
    "CV2-T2": "CV2-T0",
    "CV2-T3": "CV2-T0",
}
GATE_KIND: dict[str, str] = {
    "CV2-T2": "tailguard",
    "CV2-T3": "tailguard",
}


def _row_error(row_id: str, field: str, message: str) -> MatrixAnalysisError:
    return MatrixAnalysisError(f"row={row_id} field={field}: {message}")


def _read_json(path: Path, *, row_id: str, field: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise _row_error(row_id, field, f"missing file: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise _row_error(row_id, field, f"invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise _row_error(row_id, field, "must be a JSON object")
    return payload


def _finite(value: Any, *, row_id: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _row_error(row_id, field, "must be a finite number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise _row_error(row_id, field, "must be a finite number")
    return result


def _diagnostic_value(
    diagnostics: Mapping[str, Any], *, row_id: str, name: str
) -> float:
    sources: list[Mapping[str, Any]] = [diagnostics]
    for nested_name in ("s_dg_inputs", "S_DG_inputs", "s_dg"):
        nested = diagnostics.get(nested_name)
        if isinstance(nested, Mapping):
            sources.append(nested)
    aliases = {
        "receiver_floor": ("receiver_floor", "receiver_floor_accuracy"),
        "receiver_std": ("receiver_std", "receiver_population_std"),
        "negative_margin_rate": ("negative_margin_rate", "negative_margin"),
    }[name]
    for source in sources:
        for alias in aliases:
            if alias in source:
                return _finite(source[alias], row_id=row_id, field=f"diagnostics.{alias}")
    raise _row_error(row_id, f"diagnostics.{name}", "missing S_DG input")


def _expected_update_map(
    candidates: Sequence[str],
    expected_updates: int | Mapping[str, int] | None,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for candidate in candidates:
        if candidate not in CV2_CANDIDATE_IDS:
            raise MatrixAnalysisError(
                f"row=<run-root> field=expected_candidates: unknown CV2 candidate {candidate}"
            )
        if expected_updates is None:
            result[candidate] = candidate_config(candidate).optimizer_updates
        elif isinstance(expected_updates, Mapping):
            value = expected_updates.get(candidate)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MatrixAnalysisError(
                    f"row=<run-root> field=expected_updates.{candidate}: must be positive"
                )
            result[candidate] = value
        else:
            if isinstance(expected_updates, bool) or not isinstance(expected_updates, int):
                raise MatrixAnalysisError(
                    "row=<run-root> field=expected_updates: must be an integer"
                )
            if expected_updates <= 0:
                raise MatrixAnalysisError(
                    "row=<run-root> field=expected_updates: must be positive"
                )
            result[candidate] = expected_updates
    return result


def _attach_s_dg(row: dict[str, Any], row_root: Path) -> dict[str, Any]:
    row_id = str(row["row_id"])
    diagnostics = _read_json(
        row_root / "diagnostics.json", row_id=row_id, field="diagnostics"
    )
    receiver_floor = _diagnostic_value(
        diagnostics, row_id=row_id, name="receiver_floor"
    )
    receiver_std = _diagnostic_value(diagnostics, row_id=row_id, name="receiver_std")
    negative_margin_rate = _diagnostic_value(
        diagnostics, row_id=row_id, name="negative_margin_rate"
    )
    score = compute_s_dg(
        row["clean_accuracy"],
        row["leo_scenario_floor"],
        receiver_floor,
        receiver_std,
        negative_margin_rate,
    )
    row.update(
        {
            "receiver_floor": receiver_floor,
            "receiver_std": receiver_std,
            "negative_margin_rate": negative_margin_rate,
            "s_dg": score,
            "S_DG": score,
            "method_lock": method_lock_payload(str(row["candidate_id"])),
            "four_scenario_complete": True,
            "technical_status": "ARTIFACTS_COMPLETE",
            "technical_failure": False,
        }
    )
    return row


def _expected_row_names(
    candidates: Sequence[str], folds: Sequence[int], seeds: Sequence[int]
) -> list[tuple[str, int, int, str]]:
    return [
        (candidate, fold, seed, f"{candidate}-F{fold}-S{seed}")
        for candidate in candidates
        for fold in folds
        for seed in seeds
    ]


def _validate_expected_sets(
    run_root: Path,
    expected: Sequence[tuple[str, int, int, str]],
) -> None:
    expected_names = {item[3] for item in expected}
    try:
        actual_names = {path.name for path in run_root.iterdir() if path.is_dir()}
    except OSError as exc:
        raise MatrixAnalysisError(
            f"row=<run-root> field=row_directories: cannot enumerate: {exc}"
        ) from exc
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        first = missing[0] if missing else unexpected[0]
        raise _row_error(first, "row_directories", "; ".join(details))


def _control_gate(candidate: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gate": "control",
        "same_row": True,
        "candidate_id": candidate,
        "control_candidate_id": None,
        "passed": None,
        "status": "CONTROL",
        "scientific_result": "CONTROL_BASELINE",
        "technical_failure": False,
        "failed_conditions": [],
        "deltas": {},
    }


def _summarize_candidate(
    candidate: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    score_summary = aggregate_s_dg(rows)
    metrics = {
        "s_dg": score_summary,
        "leo_mean": {
            "mean": sum(float(row["leo_mean"]) for row in rows) / len(rows),
            "minimum": min(float(row["leo_mean"]) for row in rows),
        },
        "leo_class_floor": {
            "mean": sum(float(row["leo_class_floor"]) for row in rows) / len(rows),
            "minimum": min(float(row["leo_class_floor"]) for row in rows),
        },
        "clean_accuracy": {
            "mean": sum(float(row["clean_accuracy"]) for row in rows) / len(rows),
            "minimum": min(float(row["clean_accuracy"]) for row in rows),
        },
        "receiver_floor": {
            "mean": sum(float(row["receiver_floor"]) for row in rows) / len(rows),
            "minimum": min(float(row["receiver_floor"]) for row in rows),
        },
    }
    gates = [row["gate"] for row in rows]
    is_control = candidate not in CONTROL_FOR
    passed = None if is_control else all(gate["passed"] for gate in gates)
    scientific_result = (
        "CONTROL_BASELINE"
        if is_control
        else ("SCIENTIFIC_GATE_PASS" if passed else "NEGATIVE_SCIENTIFIC_RESULT")
    )
    return {
        "candidate_id": candidate,
        "row_count": len(rows),
        "metrics": metrics,
        "s_dg_mean": score_summary["mean"],
        "s_dg_population_std": score_summary["population_std"],
        "s_dg_minimum": score_summary["minimum"],
        "gate_kind": "control" if is_control else GATE_KIND.get(candidate, "mainline"),
        "gate_passed": passed,
        "scientific_result": scientific_result,
        "technical_status": "ARTIFACTS_COMPLETE",
        "technical_failure": False,
        "failed_rows": [
            str(row["row_id"])
            for row in rows
            if row["gate"]["passed"] is False
        ],
    }


def analyze_cv2_matrix(
    run_root: str | Path,
    *,
    expected_candidates: Sequence[str] = CV2_CANDIDATE_IDS,
    expected_folds: Sequence[int] = DEFAULT_FOLDS,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_updates: int | Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate, score, and same-row analyze a frozen CV2 matrix."""

    root = Path(run_root)
    if not root.is_dir():
        raise MatrixAnalysisError(
            f"row=<run-root> field=run_root: not a directory: {root}"
        )
    candidates = tuple(expected_candidates)
    folds = tuple(expected_folds)
    seeds = tuple(expected_seeds)
    if not candidates or len(set(candidates)) != len(candidates):
        raise MatrixAnalysisError(
            "row=<run-root> field=expected_candidates: must be unique and non-empty"
        )
    if not folds or len(set(folds)) != len(folds):
        raise MatrixAnalysisError(
            "row=<run-root> field=expected_folds: must be unique and non-empty"
        )
    if not seeds or len(set(seeds)) != len(seeds):
        raise MatrixAnalysisError(
            "row=<run-root> field=expected_seeds: must be unique and non-empty"
        )

    update_map = _expected_update_map(candidates, expected_updates)
    expected = _expected_row_names(candidates, folds, seeds)
    _validate_expected_sets(root, expected)

    analyzed_rows: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in candidates}
    for candidate, fold, seed, row_id in expected:
        row_root = root / row_id
        closure = validate_four_scenario_closure(row_root)
        if not closure["complete"]:
            missing = closure.get("missing", [])
            field = str(missing[0]) if missing else "four_scenario_closure"
            raise _row_error(row_id, field, "four-scenario artifact closure failed")
        row = base_analyzer._analyze_row(
            row_root,
            candidate=candidate,
            fold=fold,
            seed=seed,
            expected_updates=update_map[candidate],
        )
        row = _attach_s_dg(row, row_root)
        by_candidate[candidate].append(row)
        analyzed_rows.append(row)

    lookup = {
        (str(row["candidate_id"]), int(row["fold"]), int(row["seed"])): row
        for row in analyzed_rows
    }
    for row in analyzed_rows:
        candidate = str(row["candidate_id"])
        control_candidate = CONTROL_FOR.get(candidate)
        if control_candidate is None:
            row["gate"] = _control_gate(candidate, row)
            row["scientific_result"] = "CONTROL_BASELINE"
            continue
        key = (control_candidate, int(row["fold"]), int(row["seed"]))
        control = lookup.get(key)
        if control is None:
            raise _row_error(
                str(row["row_id"]),
                "same_row_control",
                f"missing control row {control_candidate}-F{row['fold']}-S{row['seed']}",
            )
        try:
            gate = evaluate_candidate_gate(
                row,
                control,
                kind=GATE_KIND.get(candidate, "mainline"),
            )
        except ValueError as exc:
            raise _row_error(str(row["row_id"]), "same_row_gate", str(exc)) from exc
        row["gate"] = gate
        row["scientific_result"] = gate["scientific_result"]

    summaries = [_summarize_candidate(candidate, by_candidate[candidate]) for candidate in candidates]
    ranking = [
        summary["candidate_id"]
        for summary in sorted(
            summaries,
            key=lambda summary: (-float(summary["s_dg_mean"]), summary["candidate_id"]),
        )
    ]
    return {
        "schema": "pairbicad_cv2_matrix_analysis_v1",
        "run_root": str(root.resolve()),
        "expected": {
            "candidates": list(candidates),
            "folds": list(folds),
            "seeds": list(seeds),
            "optimizer_updates": update_map,
        },
        "scenarios": list(SCENARIOS),
        "row_count": len(analyzed_rows),
        "rows": analyzed_rows,
        "ranking": ranking,
        "candidates": [next(summary for summary in summaries if summary["candidate_id"] == candidate) for candidate in ranking],
        "candidate_summaries": summaries,
        "technical_status": "ARTIFACTS_COMPLETE",
        "technical_failure": False,
    }


analyze_matrix = analyze_cv2_matrix


CSV_FIELDS: tuple[str, ...] = (
    "row_id",
    "candidate_id",
    "fold",
    "seed",
    "optimizer_updates",
    "clean_accuracy",
    "leo_mean",
    "leo_class_floor",
    "receiver_floor",
    "receiver_std",
    "negative_margin_rate",
    "s_dg",
    "gate_kind",
    "gate_passed",
    "scientific_result",
    "technical_failure",
    "failed_conditions",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            gate = row["gate"]
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "candidate_id": row["candidate_id"],
                    "fold": row["fold"],
                    "seed": row["seed"],
                    "optimizer_updates": row["optimizer_updates"],
                    "clean_accuracy": row["clean_accuracy"],
                    "leo_mean": row["leo_mean"],
                    "leo_class_floor": row["leo_class_floor"],
                    "receiver_floor": row["receiver_floor"],
                    "receiver_std": row["receiver_std"],
                    "negative_margin_rate": row["negative_margin_rate"],
                    "s_dg": row["s_dg"],
                    "gate_kind": gate["gate"],
                    "gate_passed": gate["passed"],
                    "scientific_result": row["scientific_result"],
                    "technical_failure": row["technical_failure"],
                    "failed_conditions": json.dumps(
                        gate["failed_conditions"], ensure_ascii=False
                    ),
                }
            )


def _parse_csv_text(raw: str, *, name: str, integer: bool) -> tuple[Any, ...]:
    values = tuple(part.strip() for part in raw.split(","))
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} must be a non-empty comma-separated unique list")
    if not integer:
        return values
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except ValueError as exc:
            raise ValueError(f"{name} must contain integers") from exc
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--expected-candidates", default=",".join(CV2_CANDIDATE_IDS)
    )
    parser.add_argument("--expected-folds", default=",".join(map(str, DEFAULT_FOLDS)))
    parser.add_argument("--expected-seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--expected-updates", type=int, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidates = _parse_csv_text(
            args.expected_candidates, name="expected_candidates", integer=False
        )
        folds = _parse_csv_text(args.expected_folds, name="expected_folds", integer=True)
        seeds = _parse_csv_text(args.expected_seeds, name="expected_seeds", integer=True)
        analysis = analyze_cv2_matrix(
            args.run_root,
            expected_candidates=candidates,
            expected_folds=folds,
            expected_seeds=seeds,
            expected_updates=args.expected_updates,
        )
        _write_json(args.output_json, analysis)
        _write_csv(args.output_csv, analysis["rows"])
    except (MatrixAnalysisError, OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
