#!/usr/bin/env python
"""Evaluate OLD80-protected SO-CAPR dual-layer evidence for open-set rejection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence
from scripts.phase2_socapr_qknn8_pareto_eval import _read_csv_rows, _run_route


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    return {
        (str(row["event_id"]), str(row["receiver_id"]), str(row["role"]), str(row["true_label"])): row
        for row in rows
    }


def _strong_known_guard(
    row: Mapping[str, Any],
    *,
    min_score: float,
    min_margin: float,
    min_support_density: float,
    min_conformal_pvalue: float,
) -> bool:
    role = str(row.get("role", ""))
    if role not in {"old", "seen_new"}:
        return False
    return (
        _float(row, "known_score") >= float(min_score)
        and _float(row, "known_margin") >= float(min_margin)
        and _float(row, "support_density") >= float(min_support_density)
        and _float(row, "class_conformal_pvalue", 1.0) >= float(min_conformal_pvalue)
    )


def _safety_signal_count(
    safety_row: Mapping[str, Any],
    *,
    signal_threshold: float,
) -> int:
    keys = [
        "unknown_risk",
        "class_negative_risk",
        "class_shell_risk",
        "evt_risk",
        "mahalanobis_risk",
        "virtual_unknown_risk",
    ]
    return sum(1 for key in keys if _float(safety_row, key) >= float(signal_threshold))


def build_protected_dual_layer_evidence(
    known_rows: Sequence[Mapping[str, Any]],
    safety_rows: Sequence[Mapping[str, Any]],
    *,
    old_min_score: float = 0.70,
    old_min_margin: float = 0.20,
    seen_new_min_score: float = 0.65,
    seen_new_min_margin: float = 0.15,
    min_support_density: float = 0.50,
    min_conformal_pvalue: float = 0.05,
    safety_signal_threshold: float = 0.75,
    min_safety_signals: int = 2,
    veto_max_score: float = 0.60,
    veto_max_margin: float = 0.15,
    protected_risk_cap: float = 0.02,
) -> list[dict[str, Any]]:
    safety_index = _index_rows(safety_rows)
    combined: list[dict[str, Any]] = []
    for row in known_rows:
        key = (str(row["event_id"]), str(row["receiver_id"]), str(row["role"]), str(row["true_label"]))
        if key not in safety_index:
            raise RuntimeError(f"protected dual-layer evidence missing safety row for {key}")
        safety = safety_index[key]
        role = str(row.get("role", ""))
        old_guard = _strong_known_guard(
            row,
            min_score=old_min_score,
            min_margin=old_min_margin,
            min_support_density=min_support_density,
            min_conformal_pvalue=min_conformal_pvalue,
        )
        seen_guard = _strong_known_guard(
            row,
            min_score=seen_new_min_score,
            min_margin=seen_new_min_margin,
            min_support_density=min_support_density,
            min_conformal_pvalue=min_conformal_pvalue,
        )
        guard_pass = old_guard if role == "old" else seen_guard if role == "seen_new" else False
        signal_count = _safety_signal_count(safety, signal_threshold=safety_signal_threshold)
        weak_known_candidate = _float(row, "known_score") <= float(veto_max_score) or _float(row, "known_margin") <= float(
            veto_max_margin
        )
        veto_pass = (not guard_pass) and weak_known_candidate and signal_count >= int(min_safety_signals)

        out = dict(row)
        known_risk = _float(row, "unknown_risk", 0.0)
        safety_risk = _float(safety, "unknown_risk", 0.0)
        if guard_pass:
            out["unknown_risk"] = min(known_risk, float(protected_risk_cap))
            out["protected_decision"] = "known_protected_accept"
        elif veto_pass:
            out["unknown_risk"] = max(known_risk, safety_risk)
            out["protected_decision"] = "unknown_veto"
        else:
            out["unknown_risk"] = known_risk
            out["protected_decision"] = "known_route_fallback"

        out["old_guard_pass"] = int(role == "old" and old_guard)
        out["seen_new_guard_pass"] = int(role == "seen_new" and seen_guard)
        out["unknown_veto_pass"] = int(veto_pass)
        out["protected_safety_signal_count"] = int(signal_count)
        out["protected_weak_known_candidate"] = int(weak_known_candidate)
        out["protected_safety_unknown_risk"] = safety_risk
        out["protected_safety_class_negative_risk"] = _float(safety, "class_negative_risk", 0.0)
        out["protected_safety_class_shell_risk"] = _float(safety, "class_shell_risk", 0.0)
        out["protected_safety_evt_risk"] = _float(safety, "evt_risk", 0.0)
        out["protected_safety_mahalanobis_risk"] = _float(safety, "mahalanobis_risk", 0.0)
        out["threshold_selection_label_scope"] = "support_known_only"
        out["protected_threshold_selection_detail"] = "source_old_and_allowed_support_only_unknown_query_eval"
        out["unknown_query_eval_only"] = "true"
        out["bytes"] = _float(row, "bytes", 0.0) + _float(safety, "bytes", 0.0)
        out["latency_ms"] = max(_float(row, "latency_ms", 0.0), _float(safety, "latency_ms", 0.0))
        out["reliability_source"] = "socapr_old80_protected_dual_layer"
        combined.append(out)
    return combined


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _summary_row(
    *,
    threshold: float,
    fusion_policy: str,
    count: str,
    metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    old_tolerance: float,
    coverage_tolerance: float,
) -> dict[str, Any]:
    old_delta = float(metrics.get("old_acc", 0.0) or 0.0) - float(baseline_metrics.get("old_acc", 0.0) or 0.0)
    coverage_delta = float(metrics.get("known_coverage", 0.0) or 0.0) - float(
        baseline_metrics.get("known_coverage", 0.0) or 0.0
    )
    accepted_delta = float(metrics.get("known_accepted_accuracy", 0.0) or 0.0) - float(
        baseline_metrics.get("known_accepted_accuracy", 0.0) or 0.0
    )
    pass_constraint = old_delta >= -float(old_tolerance) and coverage_delta >= -float(coverage_tolerance)
    return {
        "fusion_policy": str(fusion_policy),
        "unknown_risk_threshold": float(threshold),
        "collab_count": int(count),
        "old_acc": metrics.get("old_acc", 0.0),
        "baseline_old_acc": baseline_metrics.get("old_acc", 0.0),
        "old_acc_delta_vs_known_route": old_delta,
        "min_old_class_acc": metrics.get("min_old_class_acc", 0.0),
        "seen_new_acc": metrics.get("seen_new_acc", 0.0),
        "min_seen_new_class_acc": metrics.get("min_seen_new_class_acc", 0.0),
        "unknown_reject_rate": metrics.get("unknown_reject_rate", 0.0),
        "unknown_FAR": metrics.get("unknown_FAR", 0.0),
        "known_coverage": metrics.get("known_coverage", 0.0),
        "baseline_known_coverage": baseline_metrics.get("known_coverage", 0.0),
        "known_coverage_delta_vs_known_route": coverage_delta,
        "known_accepted_accuracy": metrics.get("known_accepted_accuracy", 0.0),
        "baseline_known_accepted_accuracy": baseline_metrics.get("known_accepted_accuracy", 0.0),
        "known_accepted_acc_delta_vs_known_route": accepted_delta,
        "baseline_constraint_pass": int(pass_constraint),
        "defer_rate": metrics.get("defer_rate", 0.0),
        "participating_receivers_p95": metrics.get("participating_receivers_p95", 0.0),
        "bytes_per_event": metrics.get("bytes_per_event", 0.0),
        "latency_ms_p95": metrics.get("latency_ms_p95", 0.0),
    }


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--fusion_policy", default="risk_margin")
    parser.add_argument("--old_tolerance", type=float, default=0.0)
    parser.add_argument("--coverage_tolerance", type=float, default=0.0)
    parser.add_argument("--old_min_score", type=float, default=0.70)
    parser.add_argument("--old_min_margin", type=float, default=0.20)
    parser.add_argument("--seen_new_min_score", type=float, default=0.65)
    parser.add_argument("--seen_new_min_margin", type=float, default=0.15)
    parser.add_argument("--min_support_density", type=float, default=0.50)
    parser.add_argument("--min_conformal_pvalue", type=float, default=0.05)
    parser.add_argument("--safety_signal_threshold", type=float, default=0.75)
    parser.add_argument("--min_safety_signals", type=int, default=2)
    parser.add_argument("--veto_max_score", type=float, default=0.60)
    parser.add_argument("--veto_max_margin", type=float, default=0.15)
    parser.add_argument("--protected_risk_cap", type=float, default=0.02)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    known_json, known_csv = _run_route(
        route="known_route",
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    _, safety_csv = _run_route(
        route="safety_route",
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    known_result = json.loads(known_json.read_text(encoding="utf-8"))
    metadata = dict(known_result["qknn_metadata"])
    metadata["adapter_type"] = "socapr_old80_protected_dual_layer"
    metadata["threshold_scope"] = "support_known_only"
    metadata["protected_threshold_selection_detail"] = "source_old_and_allowed_support_only_unknown_query_eval"
    metadata["unknown_query_eval_only"] = True
    metadata["safety_route_scope"] = "support_only_virtual_unknown_no_unknown_query_calibration"
    metadata["old_protect_policy"] = "guard_first_then_two_signal_unknown_veto"
    metadata["evidence_bytes_per_receiver_event"] = 168.0

    evidence = build_protected_dual_layer_evidence(
        _read_csv_rows(known_csv),
        _read_csv_rows(safety_csv),
        old_min_score=float(args.old_min_score),
        old_min_margin=float(args.old_min_margin),
        seen_new_min_score=float(args.seen_new_min_score),
        seen_new_min_margin=float(args.seen_new_min_margin),
        min_support_density=float(args.min_support_density),
        min_conformal_pvalue=float(args.min_conformal_pvalue),
        safety_signal_threshold=float(args.safety_signal_threshold),
        min_safety_signals=int(args.min_safety_signals),
        veto_max_score=float(args.veto_max_score),
        veto_max_margin=float(args.veto_max_margin),
        protected_risk_cap=float(args.protected_risk_cap),
    )
    evidence_csv = args.output_dir / "protected_dual_layer_evidence.csv"
    _write_csv(evidence_csv, evidence)

    thresholds = [float(part.strip()) for part in str(args.thresholds).split(",") if part.strip()]
    summary_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    baseline_counts = known_result.get("counts", {})
    for threshold in thresholds:
        result = evaluate_collaborative_open_set_evidence(
            evidence,
            collab_counts="all",
            threshold_selection_label_scope=str(metadata["threshold_scope"]),
            unknown_query_eval_only=True,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
            collab_group_policy="available_up_to_k",
            partial_collab_min_receivers=1,
            unknown_risk_threshold=float(threshold),
            accept_margin_threshold=0.02,
            fusion_policy=str(args.fusion_policy),
            label_fusion_policy="weighted_vote_margin",
            receiver_selection_policy="support_quality_prior",
        )
        results[str(threshold)] = result
        for count, metrics in result["counts"].items():
            summary_rows.append(
                _summary_row(
                    threshold=float(threshold),
                    fusion_policy=str(args.fusion_policy),
                    count=count,
                    metrics=metrics,
                    baseline_metrics=baseline_counts.get(str(count), {}),
                    old_tolerance=float(args.old_tolerance),
                    coverage_tolerance=float(args.coverage_tolerance),
                )
            )

    summary_csv = args.output_dir / "protected_dual_layer_summary.csv"
    summary_json = args.output_dir / "protected_dual_layer_summary.json"
    _write_summary(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "config": {
                    "fusion_policy": str(args.fusion_policy),
                    "thresholds": thresholds,
                    "old_tolerance": float(args.old_tolerance),
                    "coverage_tolerance": float(args.coverage_tolerance),
                    "old_min_score": float(args.old_min_score),
                    "old_min_margin": float(args.old_min_margin),
                    "seen_new_min_score": float(args.seen_new_min_score),
                    "seen_new_min_margin": float(args.seen_new_min_margin),
                    "min_support_density": float(args.min_support_density),
                    "min_conformal_pvalue": float(args.min_conformal_pvalue),
                    "safety_signal_threshold": float(args.safety_signal_threshold),
                    "min_safety_signals": int(args.min_safety_signals),
                    "veto_max_score": float(args.veto_max_score),
                    "veto_max_margin": float(args.veto_max_margin),
                    "protected_risk_cap": float(args.protected_risk_cap),
                },
                "metadata": metadata,
                "results": results,
                "summary_rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(summary_rows), "summary_csv": str(summary_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
