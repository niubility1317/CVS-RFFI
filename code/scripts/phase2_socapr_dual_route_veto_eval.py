#!/usr/bin/env python
"""Evaluate SO-CAPR dual-route known-candidate plus safety-veto evidence."""

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


def _discounted_safety_risk(
    known_row: Mapping[str, Any],
    safety_row: Mapping[str, Any],
    *,
    score_anchor: float,
    margin_anchor: float,
    safety_weight: float,
    discount_mode: str,
) -> float:
    known_score = _float(known_row, "known_score", 0.0)
    known_margin = _float(known_row, "known_margin", 0.0)
    known_risk = _float(known_row, "unknown_risk", 0.0)
    safety_risk = _float(safety_row, "unknown_risk", 0.0)
    score_discount = max(0.0, 1.0 - known_score / max(float(score_anchor), 1e-6))
    margin_discount = max(0.0, 1.0 - known_margin / max(float(margin_anchor), 1e-6))
    mode = str(discount_mode or "prod").strip().lower()
    if mode == "prod":
        discount = score_discount * margin_discount
    elif mode == "mean":
        discount = 0.5 * (score_discount + margin_discount)
    elif mode == "max":
        discount = max(score_discount, margin_discount)
    else:
        raise ValueError("discount_mode must be prod, mean, or max")
    return float(max(0.0, min(1.0, max(known_risk, float(safety_weight) * safety_risk * discount))))


def build_dual_route_evidence(
    known_rows: Sequence[Mapping[str, Any]],
    safety_rows: Sequence[Mapping[str, Any]],
    *,
    score_anchor: float = 0.70,
    margin_anchor: float = 0.40,
    safety_weight: float = 0.20,
    discount_mode: str = "prod",
) -> list[dict[str, Any]]:
    safety_index = _index_rows(safety_rows)
    combined: list[dict[str, Any]] = []
    for row in known_rows:
        key = (str(row["event_id"]), str(row["receiver_id"]), str(row["role"]), str(row["true_label"]))
        if key not in safety_index:
            raise RuntimeError(f"dual-route evidence missing safety row for {key}")
        safety = safety_index[key]
        out = dict(row)
        out["unknown_risk"] = _discounted_safety_risk(
            row,
            safety,
            score_anchor=score_anchor,
            margin_anchor=margin_anchor,
            safety_weight=safety_weight,
            discount_mode=discount_mode,
        )
        out["socapr_known_route_unknown_risk"] = _float(row, "unknown_risk", 0.0)
        out["socapr_safety_route_unknown_risk"] = _float(safety, "unknown_risk", 0.0)
        out["socapr_safety_mahalanobis_risk"] = _float(safety, "mahalanobis_risk", 0.0)
        out["socapr_safety_evt_risk"] = _float(safety, "evt_risk", 0.0)
        out["socapr_safety_class_shell_risk"] = _float(safety, "class_shell_risk", 0.0)
        out["socapr_safety_class_negative_risk"] = _float(safety, "class_negative_risk", 0.0)
        out["socapr_score_anchor"] = float(score_anchor)
        out["socapr_margin_anchor"] = float(margin_anchor)
        out["socapr_safety_weight"] = float(safety_weight)
        out["socapr_discount_mode"] = str(discount_mode)
        out["bytes"] = _float(row, "bytes", 0.0) + _float(safety, "bytes", 0.0)
        out["latency_ms"] = max(_float(row, "latency_ms", 0.0), _float(safety, "latency_ms", 0.0))
        out["reliability_source"] = "socapr_dual_route_known_candidate_safety_veto"
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
) -> dict[str, Any]:
    return {
        "fusion_policy": str(fusion_policy),
        "unknown_risk_threshold": float(threshold),
        "collab_count": int(count),
        "old_acc": metrics.get("old_acc", 0.0),
        "min_old_class_acc": metrics.get("min_old_class_acc", 0.0),
        "seen_new_acc": metrics.get("seen_new_acc", 0.0),
        "min_seen_new_class_acc": metrics.get("min_seen_new_class_acc", 0.0),
        "unknown_reject_rate": metrics.get("unknown_reject_rate", 0.0),
        "unknown_FAR": metrics.get("unknown_FAR", 0.0),
        "known_coverage": metrics.get("known_coverage", 0.0),
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
    parser.add_argument("--score_anchor", type=float, default=0.70)
    parser.add_argument("--margin_anchor", type=float, default=0.40)
    parser.add_argument("--safety_weight", type=float, default=0.20)
    parser.add_argument("--discount_mode", choices=["prod", "mean", "max"], default="prod")
    parser.add_argument("--thresholds", default="0.4,0.6,0.8")
    parser.add_argument("--fusion_policy", default="risk_margin")
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
    metadata["adapter_type"] = "socapr_dual_route_known_candidate_safety_veto"
    metadata["safety_route_scope"] = "support_only_virtual_unknown_no_unknown_query_calibration"
    metadata["evidence_bytes_per_receiver_event"] = 168.0

    evidence = build_dual_route_evidence(
        _read_csv_rows(known_csv),
        _read_csv_rows(safety_csv),
        score_anchor=float(args.score_anchor),
        margin_anchor=float(args.margin_anchor),
        safety_weight=float(args.safety_weight),
        discount_mode=str(args.discount_mode),
    )
    evidence_csv = args.output_dir / "dual_route_veto_evidence.csv"
    _write_csv(evidence_csv, evidence)

    thresholds = [float(part.strip()) for part in str(args.thresholds).split(",") if part.strip()]
    summary_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
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
                )
            )

    summary_csv = args.output_dir / "dual_route_veto_summary.csv"
    summary_json = args.output_dir / "dual_route_veto_summary.json"
    _write_summary(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "config": {
                    "score_anchor": float(args.score_anchor),
                    "margin_anchor": float(args.margin_anchor),
                    "safety_weight": float(args.safety_weight),
                    "discount_mode": str(args.discount_mode),
                    "fusion_policy": str(args.fusion_policy),
                    "thresholds": thresholds,
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
