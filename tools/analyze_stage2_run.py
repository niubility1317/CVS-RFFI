"""Post-run evidence analysis for Stage2 CV-SincNet experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FATAL_MARKERS = (
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "Killed",
    "status 137",
    "nan",
    "inf",
)


def _finite(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return vals[idx]


def _trend(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"count": 0}
    window = max(1, len(vals) // 10)
    first = vals[:window]
    last = vals[-window:]
    return {
        "count": len(vals),
        "start": vals[0],
        "final": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "first10_mean": _mean(first),
        "last10_mean": _mean(last),
        "first_to_last10_delta": (_mean(last) - _mean(first)) if first and last else None,
        "last_step_delta": vals[-1] - vals[-2] if len(vals) > 1 else None,
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def _get_adapter(metrics: dict[str, Any]) -> dict[str, Any]:
    adapter = (
        metrics.get("telemetry", {})
        .get("oa_mse_onboard_adaptation", {})
        .get("target_adapter", {})
    )
    return adapter if isinstance(adapter, dict) else {}


def _read_loss_trace(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    adapter = _get_adapter(metrics)
    trace = adapter.get("loss_trace") or metrics.get("loss_trace") or []
    return trace if isinstance(trace, list) else []


def _row_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
    adapter = _get_adapter(data)
    trace = _read_loss_trace(data)
    trace_keys: set[str] = set()
    for step in trace:
        if isinstance(step, dict):
            trace_keys.update(step.keys())
    trend_by_key = {}
    for key in sorted(trace_keys):
        values = [_finite(step.get(key)) for step in trace if isinstance(step, dict)]
        numeric = [v for v in values if v is not None]
        if numeric and (key.startswith("loss") or key in {"grad_norm", "support_acc", "lr"}):
            trend_by_key[key] = _trend(numeric)
    return {
        "id": path.parent.name,
        "metrics_path": str(path),
        "old_acc": _finite(metrics.get("old_class_accuracy")),
        "seen_new_acc": _finite(metrics.get("new_class_accuracy")),
        "unknown_rejection": _finite(metrics.get("unknown_rejection_rate")),
        "unknown_far": _finite(metrics.get("unknown_false_accept_rate")),
        "full_accuracy": _finite(metrics.get("full_accuracy")),
        "coverage": _finite(metrics.get("coverage")),
        "old_reject_rate": _finite(metrics.get("old_reject_rate")),
        "unknown_to_old_rate": _finite(metrics.get("unknown_to_old_rate")),
        "unknown_to_seen_new_rate": _finite(metrics.get("unknown_to_seen_new_rate")),
        "selected_alpha": adapter.get("selected_alpha"),
        "adapter_selection_policy": adapter.get("adapter_selection_policy"),
        "loss_trace_status": data.get("loss_trace_status"),
        "loss_trace_schema": data.get("loss_trace_schema"),
        "loss_trace_len": len(trace),
        "loss_initial": _finite(data.get("loss_initial") or adapter.get("loss_initial")),
        "loss_final": _finite(data.get("loss_final") or adapter.get("loss_final")),
        "loss_trends": trend_by_key,
        "loss_terms": adapter.get("loss_terms") or data.get("loss_terms") or {},
        "adapter_selection": adapter.get("adapter_selection") or {},
    }


def _score_table_summary(path: Path) -> dict[str, Any]:
    by_group: dict[str, Counter] = defaultdict(Counter)
    by_group_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_scenario: dict[str, Counter] = defaultdict(Counter)
    gate_reasons: Counter = Counter()
    outcomes: Counter = Counter()
    rows = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            true_group = row.get("true_group") or "missing"
            pred_group = row.get("predicted_group") or "missing"
            outcome = row.get("outcome_code") or "missing"
            gate_reason = row.get("gate_reason") or "missing"
            scenario = row.get("query_sat_scenario") or "missing"
            by_group[true_group][pred_group] += 1
            by_group[true_group][outcome] += 1
            by_scenario[scenario][outcome] += 1
            gate_reasons[gate_reason] += 1
            outcomes[outcome] += 1
            for key in (
                "best_old_score",
                "best_seen_new_score",
                "seen_new_minus_old_score",
                "unknown_score",
                "margin",
                "mahalanobis",
                "openmax_distance",
                "subspace_residual",
                "min_accept_delta",
                "old_support_evidence_delta",
                "seen_new_evidence_delta",
            ):
                value = _finite(row.get(key))
                if value is not None:
                    by_group_values[true_group][key].append(value)
    numeric = {}
    for group, fields in by_group_values.items():
        numeric[group] = {}
        for key, values in fields.items():
            numeric[group][key] = {
                "mean": _mean(values),
                "p10": _quantile(values, 0.10),
                "p50": _quantile(values, 0.50),
                "p90": _quantile(values, 0.90),
                "count": len(values),
            }
    return {
        "id": path.parent.name,
        "rows": rows,
        "confusion_by_true_group": {k: dict(v) for k, v in sorted(by_group.items())},
        "outcomes": dict(outcomes.most_common()),
        "gate_reasons": dict(gate_reasons.most_common(12)),
        "outcomes_by_scenario": {k: dict(v) for k, v in sorted(by_scenario.items())},
        "numeric_by_true_group": numeric,
    }


def _scan_logs(log_root: Path) -> dict[str, Any]:
    if not log_root.exists():
        return {"status": "LOG_ROOT_MISSING", "path": str(log_root)}
    files = [p for p in sorted(log_root.rglob("*")) if p.is_file()]
    marker_hits = []
    total_bytes = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            marker_hits.append({"path": str(path), "error": str(exc)})
            continue
        total_bytes += len(text.encode("utf-8", errors="replace"))
        for marker in FATAL_MARKERS:
            count = text.count(marker)
            if count:
                marker_hits.append({"path": str(path), "marker": marker, "count": count})
    return {
        "status": "SCANNED",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "fatal_marker_hits": marker_hits,
    }


def analyze(run_root: Path, log_root: Path) -> dict[str, Any]:
    metric_paths = sorted(run_root.glob("*/metrics.json"))
    score_paths = sorted(run_root.glob("*/score_table.csv"))
    row_summaries = [_row_summary(path) for path in metric_paths]
    score_summaries = [_score_table_summary(path) for path in score_paths]
    losses = [row["loss_final"] for row in row_summaries if row["loss_final"] is not None]
    loss_initial = [row["loss_initial"] for row in row_summaries if row["loss_initial"] is not None]
    old = [row["old_acc"] for row in row_summaries if row["old_acc"] is not None]
    seen = [row["seen_new_acc"] for row in row_summaries if row["seen_new_acc"] is not None]
    rej = [row["unknown_rejection"] for row in row_summaries if row["unknown_rejection"] is not None]
    far = [row["unknown_far"] for row in row_summaries if row["unknown_far"] is not None]
    alpha_counts = Counter(str(row.get("selected_alpha")) for row in row_summaries)
    policy_counts = Counter(str(row.get("adapter_selection_policy")) for row in row_summaries)
    target_hits = [
        row["id"]
        for row in row_summaries
        if (row["old_acc"] or -1.0) >= 0.95
        and (row["seen_new_acc"] or -1.0) >= 0.80
        and (row["unknown_rejection"] or -1.0) >= 0.95
    ]
    return {
        "schema": "stage2_postrun_evidence_analysis_v1",
        "run_root": str(run_root),
        "log_root": str(log_root),
        "artifact_counts": {
            "metrics_json": len(metric_paths),
            "score_table_csv": len(score_paths),
        },
        "summary": {
            "old_mean": _mean(old),
            "old_max": max(old) if old else None,
            "seen_new_mean": _mean(seen),
            "seen_new_max": max(seen) if seen else None,
            "unknown_rejection_mean": _mean(rej),
            "unknown_rejection_max": max(rej) if rej else None,
            "unknown_far_mean": _mean(far),
            "unknown_far_min": min(far) if far else None,
            "loss_initial_mean": _mean(loss_initial),
            "loss_final_mean": _mean(losses),
            "alpha_counts": dict(alpha_counts),
            "policy_counts": dict(policy_counts),
            "target_hit_count": len(target_hits),
            "target_hits": target_hits,
        },
        "top_old": sorted(row_summaries, key=lambda r: r["old_acc"] or -1.0, reverse=True)[:8],
        "top_unknown_rejection": sorted(
            row_summaries, key=lambda r: r["unknown_rejection"] or -1.0, reverse=True
        )[:8],
        "rows": row_summaries,
        "score_tables": score_summaries,
        "log_scan": _scan_logs(log_root),
        "loss_verdict": "TRAINING_LOG_ANALYSIS_PASS" if row_summaries and all(
            row["loss_trace_len"] > 0 for row in row_summaries
        ) else "MISSING_LOSS_TELEMETRY",
        "optimization_verdict": "DIAGNOSTIC_NEGATIVE_NO_TARGET_HIT",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    out = analyze(args.run_root, args.log_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print("loss_verdict=" + out["loss_verdict"])
    print("optimization_verdict=" + out["optimization_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
