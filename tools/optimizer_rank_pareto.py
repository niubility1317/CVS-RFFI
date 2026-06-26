#!/usr/bin/env python
"""Rank optimizer summaries with Pareto and no-regression gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from optimizer_workflow_lib import candidate_best_metrics, dominates, load_standard_summary, write_json


LANE_THRESHOLDS = {
    "centralized": {"strict_udu": 84.0, "sat_floor": 41.0, "receiver_floor": 73.0},
    "federated_vmb": {"strict_udu": 77.5, "sat_floor": 37.0, "receiver_floor": 58.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", nargs="+", type=Path, help="optimizer_batch_summary_v1 or legacy summary JSON.")
    parser.add_argument("--output", type=Path, help="Write ranking JSON here.")
    return parser.parse_args()


def row_for_item(summary: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = candidate_best_metrics(item)
    thresholds = LANE_THRESHOLDS.get(summary["lane"], {})
    no_regression = {
        key: (metrics.get(key) is not None and metrics[key] >= floor)
        for key, floor in thresholds.items()
    }
    views = item.get("views") if isinstance(item.get("views"), dict) else {}
    comparable_views = [view for view in views.values() if isinstance(view, dict)]
    if summary["lane"] == "centralized":
        diagnostic = any(
            view.get("sat_floor") is not None
            and view["sat_floor"] >= 41.0
            and (
                (view.get("strict_udu") is not None and view["strict_udu"] < 82.0)
                or (view.get("receiver_floor") is not None and view["receiver_floor"] < 65.0)
            )
            for view in comparable_views
        )
    else:
        diagnostic = any(
            view.get("sat_floor") is not None
            and view["sat_floor"] >= 37.5
            and view.get("receiver_floor") is not None
            and view["receiver_floor"] < 50.0
            for view in comparable_views
        )
    return {
        "batch": summary["batch"],
        "lane": summary["lane"],
        "candidate_id": item.get("candidate_id"),
        "run_name": item.get("run_name"),
        "status": item.get("status"),
        "metrics": metrics,
        "no_regression": no_regression,
        "eligible_anchor": bool(thresholds and all(no_regression.values())),
        "diagnostic_only": diagnostic,
        "collapse_flags": item.get("collapse_flags", []),
        "mechanism_activation": item.get("mechanism_activation", {}),
    }


def compute_pareto(rows: List[Dict[str, Any]]) -> List[str]:
    front: List[str] = []
    keys = ("strict_udu", "sat_floor", "receiver_floor")
    for row in rows:
        if row["diagnostic_only"] or row.get("collapse_flags"):
            continue
        metrics = row["metrics"]
        dominated = any(
            other is not row and dominates(other["metrics"], metrics, keys)
            for other in rows
            if other["lane"] == row["lane"] and not other["diagnostic_only"] and not other.get("collapse_flags")
        )
        if not dominated:
            front.append(str(row["candidate_id"]))
    return sorted(set(front))


def main() -> int:
    args = parse_args()
    summaries = [load_standard_summary(path) for path in args.summary_json]
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        for item in summary.get("items", []):
            if isinstance(item, dict):
                rows.append(row_for_item(summary, item))
    payload = {
        "schema": "optimizer_pareto_ranking_v1",
        "source_count": len(summaries),
        "candidate_count": len(rows),
        "pareto_front": compute_pareto(rows),
        "anchor_pool": [row for row in rows if row["eligible_anchor"] and not row["diagnostic_only"] and not row.get("collapse_flags")],
        "diagnostic_only": [row for row in rows if row["diagnostic_only"]],
        "rows": rows,
    }
    if args.output:
        write_json(args.output, payload)
    print(json.dumps({k: payload[k] for k in ("schema", "candidate_count", "pareto_front")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
