#!/usr/bin/env python3
"""Build a compact, machine-readable summary for an M2.4 D1 matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = ("A_o_pre", "A_o_post", "A_n", "H", "F", "min_old", "min_new")


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _weighted(rows: Iterable[dict[str, Any]], key: str) -> float:
    rows = list(rows)
    total = sum(int(row["query_count"]) for row in rows)
    return sum(float(row[key]) * int(row["query_count"]) for row in rows) / total


def _stats(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {key: value for key, value in zip(keys, identity)}
        record["row_count"] = len(members)
        record["metrics"] = {
            metric: _stats(member["aggregate_metrics"][metric] for member in members)
            for metric in METRICS
        }
        output.append(record)
    return output


def build_summary(prediction_root: Path, score_root: Path) -> dict[str, Any]:
    matrix = _load(prediction_root / "matrix_index.json")
    scored = _load(score_root / "scored_matrix_index.json")
    identities = {entry["row_id"]: entry for entry in scored["entries"]}
    rows = []
    for path in sorted(score_root.rglob("same_row_score.json")):
        score = _load(path)
        row_id = score["logical_row_key"]
        identity = identities[row_id]
        scenario_rows = [
            {key: scenario[key] for key in ("scenario", "query_count", *METRICS)}
            for scenario in score["scenario_rows"]
        ]
        aggregate = {metric: _weighted(scenario_rows, metric) for metric in METRICS}
        parity = identity["d1_historical_parity"]
        resource = score["resource"]
        rows.append(
            {
                "row_id": row_id,
                "receiver": identity["receiver"],
                "method_seed": identity["method_seed"],
                "support_seed": identity["support_seed"],
                "query_seed": identity["query_seed"],
                "new_class_draw_seed": identity["new_class_draw_seed"],
                "split_id": identity["split_id"],
                "k_shot": identity["k_shot"],
                "new_class_count": identity["new_class_count"],
                "condition": f"K{identity['k_shot']}_new{identity['new_class_count']}",
                "status": score["status"],
                "truth_opened_after_prediction_commit": score[
                    "truth_opened_after_prediction_commit"
                ],
                "d1_historical_parity": parity,
                "aggregate_metrics": aggregate,
                "scenario_metrics": scenario_rows,
                "resource": {
                    key: resource.get(key)
                    for key in (
                        "state_bytes",
                        "deployment_state_bytes",
                        "registration_time_ms",
                        "candidate_head_batch_query_latency_ms_per_row",
                        "query_head_mac",
                        "mac_equivalent_upper_bound",
                        "closed_form_fit_count",
                    )
                },
            }
        )

    parity_disagreements = sum(
        row["d1_historical_parity"]["prediction_disagreements"] for row in rows
    )
    scenario_units = sum(len(row["scenario_metrics"]) for row in rows)
    return {
        "schema": "cvs.erbt_idr.m24.d1_expanded.results_summary.v1",
        "run_id": "erbt_idr_m24_d1_expanded_20260820_v1",
        "status": "ANALYZED / DEVELOPMENT_EXPANDED_EVIDENCE / D1_PARITY_PASS",
        "evidence_boundary": (
            "D1 stability and historical-F1 parity evidence; not M2.4 module gain, "
            "fresh confirmation, Phase3, or deployment evidence."
        ),
        "protocol": {
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "prediction_truth_opened": matrix["query_truth_opened"],
            "truth_last_rows": sum(
                bool(row["truth_opened_after_prediction_commit"]) for row in rows
            ),
        },
        "matrix": {
            "row_count": len(rows),
            "scenario_unit_count": scenario_units,
            "receivers": matrix["receivers"],
            "method_seeds": matrix["method_seeds"],
            "conditions": matrix["conditions"],
            "parity_prediction_disagreements": parity_disagreements,
            "pass_rows": sum(row["status"] == "PASS" for row in rows),
        },
        "metric_semantics": {
            "A_o_pre": "DA0_REG0 old-class accuracy",
            "A_o_post": "DA0_REG1 old-class accuracy",
            "A_n": "DA0_REG1 new-class accuracy",
            "H": "DA0_REG1 old/new harmonic mean",
            "F": "A_o_pre - A_o_post (positive means forgetting)",
            "min_old": "minimum registered old-class accuracy",
            "min_new": "minimum registered new-class accuracy",
            "aggregate": "query-count-weighted mean over three leo_*_weak scenarios",
        },
        "condition_summary": _group(rows, ("condition",)),
        "receiver_summary": _group(rows, ("receiver",)),
        "method_seed_summary": _group(rows, ("method_seed",)),
        "condition_receiver_summary": _group(rows, ("condition", "receiver")),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.prediction_root, args.score_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary["matrix"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
