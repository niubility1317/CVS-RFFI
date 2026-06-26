"""Summarize old-first Stage2 gate behavior from metrics and score tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOOL_TRUE = {"1", "1.0", "true", "yes", "y"}


def finite(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stats(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "min": min(vals),
        "median": statistics.median(vals),
        "max": max(vals),
    }


def metric_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
        adapter = (
            data.get("telemetry", {})
            .get("oa_mse_onboard_adaptation", {})
            .get("target_adapter", {})
        )
        if not isinstance(adapter, dict):
            adapter = {}
        cid = path.parent.name
        parts = cid.split("_")
        gpu = next((part for part in parts if part.startswith("GPU")), "?")
        gpu_index = parts.index(gpu) if gpu in parts else -1
        slot = parts[gpu_index + 1] if 0 <= gpu_index + 1 < len(parts) else "?"
        row = {
            "candidate_id": cid,
            "gpu": gpu,
            "slot": slot,
            "category": "aggressive" if slot in {"D", "E", "F"} else "conservative",
            "kold": 10 if "KOLD10" in cid else 5 if "KOLD5" in cid else None,
            "adapter_policy": adapter.get("adapter_selection_policy"),
            "selected_alpha": adapter.get("selected_alpha"),
            "old_acc": finite(metrics.get("old_class_accuracy")),
            "unknown_far": finite(metrics.get("unknown_false_accept_rate")),
            "unknown_rejection": finite(metrics.get("unknown_rejection_rate")),
            "full_accuracy": finite(metrics.get("full_accuracy")),
            "coverage": finite(metrics.get("coverage")),
            "auroc": finite(metrics.get("auroc")),
            "loss_initial": finite(data.get("loss_initial") or adapter.get("loss_initial")),
            "loss_final": finite(data.get("loss_final") or adapter.get("loss_final")),
        }
        if row["old_acc"] is not None and row["unknown_far"] is not None:
            known_reject = 1.0 - row["unknown_far"]
            denom = row["old_acc"] + known_reject
            row["old_unknown_hmean"] = 0.0 if denom <= 0 else 2.0 * row["old_acc"] * known_reject / denom
            row["old_far_balance"] = row["old_acc"] - row["unknown_far"]
        rows.append(row)
    return rows


def top(rows: list[dict[str, Any]], key: str, reverse: bool = True, n: int = 8) -> list[dict[str, Any]]:
    valid = [r for r in rows if isinstance(r.get(key), (int, float))]
    selected = sorted(valid, key=lambda r: r[key], reverse=reverse)[:n]
    fields = [
        "candidate_id",
        "slot",
        "category",
        "kold",
        key,
        "old_acc",
        "unknown_far",
        "unknown_rejection",
        "coverage",
        "full_accuracy",
        "old_unknown_hmean",
        "old_far_balance",
        "adapter_policy",
        "selected_alpha",
    ]
    return [{field: row.get(field) for field in fields if field in row} for row in selected]


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for key in (
            f"slot_{row.get('slot')}",
            f"kold_{row.get('kold')}",
            str(row.get("category")),
            str(row.get("adapter_policy")),
        ):
            grouped[key].append(row)
    output = {}
    for key, vals in grouped.items():
        output[key] = {
            metric: stats([row[metric] for row in vals if isinstance(row.get(metric), (int, float))])
            for metric in ("old_acc", "unknown_far", "unknown_rejection", "coverage", "old_unknown_hmean")
        }
    return output


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in BOOL_TRUE


def score_summary(run_root: Path) -> dict[str, Any]:
    by_true: dict[str, Counter] = defaultdict(Counter)
    by_candidate: dict[str, Counter] = defaultdict(Counter)
    numeric: dict[str, list[float]] = defaultdict(list)
    bool_columns = (
        "accepted",
        "old_primary_consistency_pass",
        "old_primary_unknown_veto",
        "old_primary_unknown_veto_applied",
        "old_primary_blocked_accept",
        "old_primary_evidence_pass",
        "old_primary_anchor_delta_pass",
        "old_primary_anchor_margin_pass",
        "old_primary_score_margin_pass",
        "old_primary_soft_mixture_pass",
        "old_primary_support_knn_pass",
        "old_primary_drift_pass",
        "old_primary_class_envelope_pass",
        "retention_rescue_accept",
        "class_envelope_reject",
        "two_branch_background_reject",
        "pre_reject_arbitration_reject",
        "pre_reject_arbitration_defer",
        "support_conformal_reject",
        "support_reconstruction_reject",
        "source_looo_risk_reject",
    )
    numeric_columns = (
        "old_primary_background_score",
        "old_primary_background_margin",
        "old_primary_drift_cos",
        "old_primary_drift_dist",
        "old_primary_score_margin",
        "old_primary_soft_mixture_cos",
        "old_primary_soft_mixture_residual",
        "old_support_evidence_delta",
        "support_knn_margin",
        "class_envelope_failure_count",
    )
    total_rows = 0
    for path in sorted(run_root.glob("*/score_table.csv")):
        cid = path.parent.name
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_rows += 1
                true_group = row.get("true_group") or "missing"
                pred_group = row.get("predicted_group") or "missing"
                decision = row.get("decision") or "missing"
                outcome = row.get("outcome_code") or "missing"
                gate_reason = row.get("gate_reason") or "missing"
                for bucket in (by_true[true_group], by_candidate[cid]):
                    bucket["rows"] += 1
                    bucket[f"pred_{pred_group}"] += 1
                    bucket[f"decision_{decision}"] += 1
                    bucket[f"outcome_{outcome}"] += 1
                    bucket[f"gate_{gate_reason}"] += 1
                    for col in bool_columns:
                        if is_true(row.get(col)):
                            bucket[f"{col}_true"] += 1
                for col in numeric_columns:
                    value = finite(row.get(col))
                    if value is not None:
                        numeric[f"{true_group}.{col}"].append(value)
    ratios = {}
    for true_group, counts in by_true.items():
        denominator = counts.get("rows", 0) or 1
        ratios[true_group] = {
            key.removesuffix("_true"): value / denominator
            for key, value in counts.items()
            if key.endswith("_true")
        }
    return {
        "score_table_rows": total_rows,
        "by_true_group": {key: dict(value) for key, value in sorted(by_true.items())},
        "gate_ratios_by_true_group": ratios,
        "numeric_gate_stats": {key: stats(value) for key, value in sorted(numeric.items())},
    }


def analyze(run_root: Path) -> dict[str, Any]:
    rows = metric_rows(run_root)
    output = {
        "schema": "oldfirst_gate_analysis_v1",
        "run_root": str(run_root),
        "candidate_count": len(rows),
        "metric_stats": {
            key: stats([row[key] for row in rows if isinstance(row.get(key), (int, float))])
            for key in (
                "old_acc",
                "unknown_far",
                "unknown_rejection",
                "full_accuracy",
                "coverage",
                "auroc",
                "loss_initial",
                "loss_final",
                "old_unknown_hmean",
                "old_far_balance",
            )
        },
        "group_metrics": group_metrics(rows),
        "top_old": top(rows, "old_acc"),
        "best_unknown_far": top(rows, "unknown_far", reverse=False),
        "best_hmean": top(rows, "old_unknown_hmean"),
        "best_balance": top(rows, "old_far_balance"),
        "score_summary": score_summary(run_root),
        "rows": rows,
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    out = analyze(args.run_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_count": out["candidate_count"],
        "old_acc": out["metric_stats"]["old_acc"],
        "unknown_far": out["metric_stats"]["unknown_far"],
        "coverage": out["metric_stats"]["coverage"],
        "best_hmean": out["best_hmean"][:3],
        "gate_ratios_by_true_group": out["score_summary"]["gate_ratios_by_true_group"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
