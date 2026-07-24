from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ARMS = ("M0", "M_DA", "M_OTHER", "M_JOINT")
GROUP_FIELDS = ("receiver", "scene", "k_shot", "seed", "new_class_count")


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return fmean(finite) if finite else None


def _minimum(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _row_identity(receipt: dict[str, Any], score: dict[str, Any], job_name: str) -> dict[str, Any]:
    def pick(*names: str) -> Any:
        for name in names:
            if name in receipt:
                return receipt[name]
            if name in score:
                return score[name]
        return None

    return {
        "job": job_name,
        "receiver": str(pick("receiver", "target_receiver")),
        "seed": int(pick("seed")),
        "k_shot": int(pick("k_shot", "k")),
        "new_class_count": int(pick("new_class_count", "new_count")),
    }


def _arm_row_metrics(score: dict[str, Any]) -> dict[str, float | None]:
    before = score["before"]
    after = score["after"]
    by_tx = list(after.get("by_tx", {}).values())
    old_class = [item.get("accuracy") for item in by_tx if item.get("role") == "target_old"]
    new_class = [item.get("accuracy") for item in by_tx if item.get("role") == "target_new"]
    all_class = [item.get("accuracy") for item in by_tx]
    by_scene = list(after.get("by_scenario", {}).values())
    old_before = float(before["old_acc"])
    old_after = float(after["old_acc"])
    return {
        "old_before": old_before,
        "old_after": old_after,
        "old_gain": old_after - old_before,
        "seen_new": float(after["seen_new_acc"]),
        "h": float(after["h_old_new"]),
        "ba": _mean(all_class),
        "floor": _minimum(all_class),
        "min_old": _minimum(old_class),
        "min_new": _minimum(new_class),
        "forgetting": old_before - old_after,
        "old_to_new": _mean(item.get("old_to_new_rate") for item in by_scene),
        "new_to_old": _mean(item.get("new_to_old_rate") for item in by_scene),
    }


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    metric_names = tuple(rows[0]["metrics"])
    result = {name: _mean(row["metrics"].get(name) for row in rows) for name in metric_names}
    result["row_count"] = len(rows)
    return result


def _group_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    output = []
    for value, items in sorted(grouped.items()):
        arm_summaries = {
            arm: _aggregate_metric_rows([item["arms"][arm] for item in items])
            for arm in ARMS
        }
        output.append(
            {
                field: value,
                "row_count": len(items),
                "arms": arm_summaries,
                "mean_i_syn": _mean(item["i_syn"] for item in items),
            }
        )
    return output


def _scene_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for scene in row["scenes"]:
            grouped[scene["scene"]].append(scene)
    output = []
    for scene_name, items in sorted(grouped.items()):
        arms = {
            arm: {"h": _mean(item["h"][arm] for item in items), "slice_count": len(items)}
            for arm in ARMS
        }
        output.append(
            {
                "scene": scene_name,
                "slice_count": len(items),
                "arms": arms,
                "mean_i_syn": _mean(item["i_syn"] for item in items),
                "positive_i_syn_slices": sum(item["i_syn"] > 0.0 for item in items),
                "zero_i_syn_slices": sum(item["i_syn"] == 0.0 for item in items),
                "negative_i_syn_slices": sum(item["i_syn"] < 0.0 for item in items),
            }
        )
    return output


def summarize_run(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for m0_path in sorted(run_dir.glob("artifacts/jobs/*/scorer/M0.score.json")):
        job_dir = m0_path.parents[1]
        receipt_path = job_dir / "row_receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(receipt_path)
        scores = {arm: _load_json(job_dir / "scorer" / f"{arm}.score.json") for arm in ARMS}
        receipt = _load_json(receipt_path)
        identity = _row_identity(receipt, scores["M0"], job_dir.name)
        arms = {
            arm: {"arm": arm, "metrics": _arm_row_metrics(scores[arm])}
            for arm in ARMS
        }
        i_syn = (
            arms["M_JOINT"]["metrics"]["h"]
            - arms["M_DA"]["metrics"]["h"]
            - arms["M_OTHER"]["metrics"]["h"]
            + arms["M0"]["metrics"]["h"]
        )
        scenes = []
        scene_names = sorted(scores["M0"]["after"]["by_scenario"])
        for scene_name in scene_names:
            h = {
                arm: float(scores[arm]["after"]["by_scenario"][scene_name]["h_old_new"])
                for arm in ARMS
            }
            scene_i_syn = h["M_JOINT"] - h["M_DA"] - h["M_OTHER"] + h["M0"]
            scenes.append({"scene": scene_name, "h": h, "i_syn": scene_i_syn})
        rows.append({**identity, "arms": arms, "i_syn": i_syn, "scenes": scenes})

    completion_path = run_dir / "artifacts" / "matrix_runtime_completion.json"
    completion = _load_json(completion_path) if completion_path.is_file() else {}
    succeeded = len(rows)
    summary = {
        "run_id": run_dir.name,
        "candidate": rows[0]["arms"]["M0"] and _load_json(
            next(run_dir.glob("artifacts/jobs/*/scorer/M0.score.json"))
        ).get("candidate") if rows else None,
        "diagnostic_status": (
            "COMPLETE_MATRIX_PERFORMANCE" if succeeded == 125 else
            "PARTIAL_MATRIX_DIAGNOSTIC" if succeeded else
            "NO_PERFORMANCE_ROWS"
        ),
        "successful_rows": succeeded,
        "expected_rows": 125,
        "row_coverage": succeeded / 125.0,
        "scene_slices": succeeded * 3,
        "arms": {
            arm: _aggregate_metric_rows([row["arms"][arm] for row in rows])
            for arm in ARMS
        },
        "mean_i_syn": _mean(row["i_syn"] for row in rows),
        "positive_i_syn_rows": sum(row["i_syn"] > 0.0 for row in rows),
        "zero_i_syn_rows": sum(row["i_syn"] == 0.0 for row in rows),
        "negative_i_syn_rows": sum(row["i_syn"] < 0.0 for row in rows),
        "positive_i_syn_slices": sum(
            scene["i_syn"] > 0.0 for row in rows for scene in row["scenes"]
        ),
        "zero_i_syn_slices": sum(
            scene["i_syn"] == 0.0 for row in rows for scene in row["scenes"]
        ),
        "negative_i_syn_slices": sum(
            scene["i_syn"] < 0.0 for row in rows for scene in row["scenes"]
        ),
        "by_receiver": _group_summary(rows, "receiver"),
        "by_k": _group_summary(rows, "k_shot"),
        "by_seed": _group_summary(rows, "seed"),
        "by_new_count": _group_summary(rows, "new_class_count"),
        "by_scene": _scene_summary(rows),
        "completion": completion,
        "rows": rows,
    }
    return summary


def _flat_summary(run: dict[str, Any], arm: str) -> dict[str, Any]:
    metrics = run["arms"].get(arm, {})
    return {
        "run_id": run["run_id"],
        "candidate": run.get("candidate"),
        "diagnostic_status": run["diagnostic_status"],
        "successful_rows": run["successful_rows"],
        "expected_rows": run["expected_rows"],
        "row_coverage": run["row_coverage"],
        "scene_slices": run["scene_slices"],
        "arm": arm,
        **{name: metrics.get(name) for name in (
            "old_before", "old_after", "old_gain", "seen_new", "h", "ba",
            "floor", "min_old", "min_new", "forgetting", "old_to_new", "new_to_old",
        )},
        "mean_i_syn": run["mean_i_syn"],
        "positive_i_syn_slices": run["positive_i_syn_slices"],
        "zero_i_syn_slices": run["zero_i_syn_slices"],
        "negative_i_syn_slices": run["negative_i_syn_slices"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    summaries = [summarize_run(path) for path in run_dirs]
    payload = {
        "schema": "cvs.stage2.adv3b02.partial_matrix_diagnostic.v1",
        "formal_claim_rule": "Only COMPLETE_MATRIX_PERFORMANCE may support the frozen full125 verdict.",
        "partial_claim_rule": (
            "PARTIAL_MATRIX_DIAGNOSTIC is descriptive for completed rows only; "
            "it must retain coverage and missing-cell disclosure."
        ),
        "runs": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    flat = [_flat_summary(run, arm) for run in summaries for arm in ARMS]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
