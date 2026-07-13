from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from paper_reproduction.scripts.run_cvs_publication_matrix import (
    DEFAULT_K,
    DEFAULT_RECEIVERS,
    DEFAULT_SEEDS,
    PHASE_METHODS,
    _artifact_status,
    build_rows,
)


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
DETAIL_LEVELS = {
    "overall",
    "per_split",
    "per_receiver",
    "per_transmitter",
    "per_receiver_transmitter",
    "per_receiver_transmitter_day",
}


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def validate_phase1(method: str, run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "metrics.json",
        "split_manifest.json",
        "resolved_config.json",
        "score_table.csv",
        "detailed_metrics.json",
        "detailed_metrics.csv",
    )
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing_or_empty:{name}")
    if errors:
        return {"method": method, "complete": False, "errors": errors}
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    details = json.loads((run_dir / "detailed_metrics.json").read_text(encoding="utf-8"))
    score_count = _csv_rows(run_dir / "score_table.csv")
    detail_csv_count = _csv_rows(run_dir / "detailed_metrics.csv")
    if score_count != 612000 or int(metrics.get("score_row_count", -1)) != 612000:
        errors.append(f"score_row_count:{score_count}:{metrics.get('score_row_count')}")
    if len(details) != 894 or detail_csv_count != 894 or int(metrics.get("detailed_row_count", -1)) != 894:
        errors.append(f"detail_row_count:{len(details)}:{detail_csv_count}:{metrics.get('detailed_row_count')}")
    if tuple(metrics.get("scenarios", {})) != SCENARIOS:
        errors.append(f"scenarios:{tuple(metrics.get('scenarios', {}))}")
    if tuple(manifest.get("formal_satellite_scenarios", ())) != SCENARIOS:
        errors.append(f"manifest_scenarios:{manifest.get('formal_satellite_scenarios')}")
    if manifest.get("all_tests_satellite_augmented") is not True:
        errors.append("all_tests_satellite_augmented_not_true")
    if manifest.get("clean_control_in_formal_result") is not False:
        errors.append("clean_control_in_formal_result_not_false")
    levels = {str(row.get("group_type")) for row in details}
    if levels != DETAIL_LEVELS:
        errors.append(f"detail_levels:{sorted(levels)}")
    for scenario, values in metrics.get("scenarios", {}).items():
        for key in ("accuracy", "correct_count", "sample_count"):
            if key not in values or not math.isfinite(float(values[key])):
                errors.append(f"nonfinite_or_missing:{scenario}:{key}")
        if int(values.get("sample_count", -1)) != 204000:
            errors.append(f"scenario_sample_count:{scenario}:{values.get('sample_count')}")
    return {
        "method": method,
        "complete": not errors,
        "errors": errors,
        "score_row_count": score_count,
        "detailed_row_count": len(details),
        "scenarios": metrics.get("scenarios", {}),
    }


def validate_stage2(phase: str, run_root: Path, log_root: Path) -> dict[str, Any]:
    rows = build_rows(
        phase=phase,
        methods=PHASE_METHODS[phase],
        receivers=DEFAULT_RECEIVERS,
        k_grid=DEFAULT_K,
        seeds=DEFAULT_SEEDS,
        output_root=run_root,
        log_root=log_root,
    )
    incomplete = []
    by_method = {method: 0 for method in PHASE_METHODS[phase]}
    for row in rows:
        status = _artifact_status(row)
        if status["complete"]:
            by_method[row.method] += 1
        else:
            incomplete.append({"experiment_id": row.experiment_id, "status": status})
    errors: list[str] = []
    canonical_path = run_root / "matrix_manifest.json"
    if not canonical_path.is_file():
        errors.append("missing_canonical_manifest")
    else:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        expected_ids = {row.experiment_id for row in rows}
        manifest_ids = {str(row.get("experiment_id")) for row in canonical.get("rows", [])}
        if int(canonical.get("row_count", -1)) != 500 or manifest_ids != expected_ids:
            errors.append(
                f"canonical_manifest_mismatch:row_count={canonical.get('row_count')}:ids={len(manifest_ids)}"
            )
    if incomplete:
        errors.append(f"incomplete_rows:{len(incomplete)}")
    if any(count != 125 for count in by_method.values()):
        errors.append(f"method_counts:{by_method}")
    return {
        "phase": phase,
        "complete": not errors,
        "errors": errors,
        "artifact_complete_count": len(rows) - len(incomplete),
        "expected_count": len(rows),
        "by_method": by_method,
        "incomplete_preview": incomplete[:10],
    }


def validate_summary(summary_dir: Path) -> dict[str, Any]:
    errors = []
    manifest_path = summary_dir / "summary_manifest.json"
    required = (
        "per_run_results.csv",
        "per_scenario_results.csv",
        "method_k_summary.csv",
        "receiver_k_summary.csv",
        "paired_deltas_vs_cvs.csv",
        "paired_delta_summary.csv",
        "incomplete_rows.json",
        "summary_manifest.json",
    )
    for name in required:
        path = summary_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing_or_empty:{name}")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("per_run_row_count", -1)) != 1000:
            errors.append(f"per_run_row_count:{manifest.get('per_run_row_count')}")
        if int(manifest.get("per_scenario_row_count", -1)) != 3000:
            errors.append(f"per_scenario_row_count:{manifest.get('per_scenario_row_count')}")
        if int(manifest.get("incomplete_row_count", -1)) != 0:
            errors.append(f"incomplete_row_count:{manifest.get('incomplete_row_count')}")
    return {"complete": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="Final audit for the CVS publication comparison")
    parser.add_argument("--phase1", action="append", default=[], help="METHOD=RUN_DIR")
    parser.add_argument("--stage2b-root", type=Path, required=True)
    parser.add_argument("--stage2c-root", type=Path, required=True)
    parser.add_argument("--stage2b-log-root", type=Path, required=True)
    parser.add_argument("--stage2c-log-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    phase1_mapping = {}
    for value in args.phase1:
        if "=" not in value:
            raise ValueError(f"--phase1 expects METHOD=RUN_DIR, got {value!r}")
        method, path = value.split("=", 1)
        phase1_mapping[method] = Path(path)
    expected_phase1 = {"cvs_jointp0_j5", "cvcnn_ce", "riei_fd", "drift"}
    phase1_results = [validate_phase1(method, path) for method, path in sorted(phase1_mapping.items())]
    errors = []
    if set(phase1_mapping) != expected_phase1:
        errors.append(f"phase1_methods:{sorted(phase1_mapping)}")
    if any(not result["complete"] for result in phase1_results):
        errors.append("phase1_incomplete")
    stage2b = validate_stage2("stage2b", args.stage2b_root, args.stage2b_log_root)
    stage2c = validate_stage2("stage2c", args.stage2c_root, args.stage2c_log_root)
    summary = validate_summary(args.summary_dir)
    if not stage2b["complete"]:
        errors.append("stage2b_incomplete")
    if not stage2c["complete"]:
        errors.append("stage2c_incomplete")
    if not summary["complete"]:
        errors.append("summary_incomplete")
    payload = {
        "schema": "cvs_publication_comparison_final_audit_v1",
        "complete": not errors,
        "errors": errors,
        "phase1": phase1_results,
        "stage2b": stage2b,
        "stage2c": stage2c,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
