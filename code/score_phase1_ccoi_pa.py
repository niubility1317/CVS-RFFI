"""Independent truth-joining scorer for completed CCOI-PA predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


MAIN_LOADERS = {
    "test_unseen_day_seen_rx",
    "test_seen_day_unseen_rx",
    "test_unseen_day_unseen_rx",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _unique_by_id(rows: Iterable[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{name} row is missing sample_id")
        if sample_id in indexed:
            raise ValueError(f"duplicate {name} sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(int(row["predicted_class"]) == int(row["true_class"]) for row in rows)
    return {"accuracy": 100.0 * correct / max(1, total), "correct": correct, "total": total}


def score_streams(prediction_path: Path, truth_path: Path) -> Dict[str, Any]:
    predictions = _read_jsonl(prediction_path)
    truths = _read_jsonl(truth_path)
    if any("true_class" in row for row in predictions):
        raise ValueError("prediction stream must remain truth-blind")
    if any("predicted_class" in row for row in truths):
        raise ValueError("truth stream must not contain predictions")
    pred_by_id = _unique_by_id(predictions, "prediction")
    truth_by_id = _unique_by_id(truths, "truth")
    if pred_by_id.keys() != truth_by_id.keys():
        missing_truth = sorted(pred_by_id.keys() - truth_by_id.keys())[:5]
        missing_prediction = sorted(truth_by_id.keys() - pred_by_id.keys())[:5]
        raise ValueError(
            f"prediction/truth closure mismatch missing_truth={missing_truth} "
            f"missing_prediction={missing_prediction}"
        )
    joined = []
    for sample_id, prediction in pred_by_id.items():
        row = dict(prediction)
        row["true_class"] = int(truth_by_id[sample_id]["true_class"])
        joined.append(row)

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_loader: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_scenario[str(row["scenario"])].append(row)
        by_loader[f"{row['scenario']}::{row['loader']}"] .append(row)
    scenario_metrics = {}
    for scenario, rows in sorted(by_scenario.items()):
        main = [row for row in rows if str(row["loader"]) in MAIN_LOADERS]
        selected = main or rows
        if main:
            invalid_receivers = [
                row.get("receiver")
                for row in main
                if str(row.get("receiver", "")).strip().lower() in {"", "-1", "none", "unknown"}
            ]
            if invalid_receivers:
                raise ValueError(f"receiver identity is missing or invalid for {scenario} main predictions")
        receiver_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            receiver_groups[str(row.get("receiver", "unknown"))].append(row)
        receiver_metrics = {key: _accuracy(value) for key, value in receiver_groups.items()}
        finite = [value["accuracy"] for value in receiver_metrics.values() if math.isfinite(value["accuracy"])]
        scenario_metrics[scenario] = {
            "aggregate": _accuracy(selected),
            "receiver": receiver_metrics,
            "receiver_floor": min(finite) if finite else float("nan"),
        }
    leo_values = [
        scenario_metrics[name]["aggregate"]["accuracy"]
        for name in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
        if name in scenario_metrics
    ]
    return {
        "schema": "cvs.phase1.ccoi_pa_score.v2",
        "status": "ANALYZED" if len(leo_values) == 3 and "clean" in scenario_metrics else "PARTIAL",
        "prediction_count": len(joined),
        "scenario": scenario_metrics,
        "loader": {key: _accuracy(value) for key, value in sorted(by_loader.items())},
        "leo_mean_accuracy": sum(leo_values) / len(leo_values) if leo_values else float("nan"),
        "truth_joined_after_prediction": True,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score completed CCOI-PA prediction streams")
    parser.add_argument("--run_dir", required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREDICTIONS_COMPLETE_TRUTH_NOT_SCORED":
        raise RuntimeError("scorer requires completed prediction closure")
    for row, info in manifest.get("rows", {}).items():
        row_dir = run_dir / row
        metrics_path = row_dir / "metrics.json"
        if metrics_path.exists():
            raise FileExistsError(f"refusing to overwrite score artifact: {metrics_path}")
        metrics = score_streams(Path(info["prediction_path"]), Path(info["truth_path"]))
        metrics["row"] = row
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        info["metrics_path"] = str(metrics_path)
        info["score_status"] = metrics["status"]
    manifest["status"] = "ANALYZED"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[CCOI-SCORE] ANALYZED run_dir={run_dir}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
