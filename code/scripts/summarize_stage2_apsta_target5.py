"""Aggregate APSTA-P1 same-row scores and apply the preregistered joint gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def summarize(output_root: str | Path, *, expected_rows: int = 15) -> dict[str, Any]:
    root = Path(output_root)
    scores = sorted((root / "scores").glob("*_pair.json"))
    predictions = sorted((root / "predictions").glob("*_da1_reg0.json"))
    if len(scores) != int(expected_rows) or len(predictions) != int(expected_rows):
        raise ValueError("Target5 prediction/score closure is incomplete")
    row_results: list[dict[str, Any]] = []
    da0_means: list[float] = []
    da1_means: list[float] = []
    da0_floors: list[float] = []
    da1_floors: list[float] = []
    for path in scores:
        value = _read(path)
        if (
            value.get("status") != "ANALYZED"
            or value.get("protocol_schema") != "p2_min_v1"
            or value.get("phase2_data_status") != "VALIDATED_ONCE"
            or value.get("scorer_output_must_not_feed_predictor") is not True
        ):
            raise ValueError(f"invalid truth-last score boundary: {path}")
        da0 = value["DA0_REG0"]
        da1 = value["DA1_REG0"]
        mean0 = float(da0["old_class_mean_accuracy"])
        mean1 = float(da1["old_class_mean_accuracy"])
        floor0 = float(da0["old_class_floor_accuracy"])
        floor1 = float(da1["old_class_floor_accuracy"])
        da0_means.append(mean0); da1_means.append(mean1)
        da0_floors.append(floor0); da1_floors.append(floor1)
        row_results.append({
            "row_id": path.name.removesuffix("_pair.json"),
            "scenario": value["scenario"],
            "DA0_REG0_old_class_mean_accuracy": mean0,
            "DA1_REG0_old_class_mean_accuracy": mean1,
            "old_class_mean_delta_pp": 100.0 * (mean1 - mean0),
            "DA0_REG0_old_class_floor_accuracy": floor0,
            "DA1_REG0_old_class_floor_accuracy": floor1,
            "old_class_floor_delta_pp": 100.0 * (floor1 - floor0),
        })
    selected_steps: Counter[str] = Counter()
    fallback_count = 0
    for path in predictions:
        value = _read(path)
        if (
            value.get("status") != "PREDICTIONS_COMPLETE"
            or value.get("state") != "DA1_REG0"
            or value.get("query_truth_loaded") is not False
            or value.get("query_role_loaded") is not False
            or value.get("query_batch_state_updated") is not False
        ):
            raise ValueError(f"invalid prediction boundary: {path}")
        audit = value.get("audit") or {}
        if int(audit.get("backward_count", -1)) != 300:
            raise ValueError("formal APSTA row did not complete 300 support updates")
        selected_steps[str(int(audit["selected_step"]))] += 1
        fallback_count += int(bool(audit.get("fallback_to_teacher")))
    da0_mean = sum(da0_means) / len(da0_means)
    da1_mean = sum(da1_means) / len(da1_means)
    da0_floor = min(da0_floors)
    da1_floor = min(da1_floors)
    mean_delta = round(100.0 * (da1_mean - da0_mean), 10)
    floor_delta = round(100.0 * (da1_floor - da0_floor), 10)
    promote = mean_delta >= 1.0 and floor_delta >= 0.5
    return {
        "status": "ANALYZED",
        "row_count": len(scores),
        "DA0_REG0": {
            "old_class_mean_accuracy": da0_mean,
            "old_class_global_floor_accuracy": da0_floor,
        },
        "DA1_REG0": {
            "old_class_mean_accuracy": da1_mean,
            "old_class_global_floor_accuracy": da1_floor,
        },
        "DA1_REG0_minus_DA0_REG0": {
            "old_class_mean_delta_pp": mean_delta,
            "old_class_floor_delta_pp": floor_delta,
        },
        "mean_gate_pp": 1.0,
        "floor_gate_pp": 0.5,
        "verdict": "PROMOTE_TARGET25" if promote else "SCIENTIFIC_FAILURE_NO_PROMOTION",
        "support_selected_step_histogram": dict(sorted(selected_steps.items(), key=lambda item: int(item[0]))),
        "teacher_fallback_row_count": fallback_count,
        "rows": row_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-rows", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.output_root, expected_rows=args.expected_rows)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite summary: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
