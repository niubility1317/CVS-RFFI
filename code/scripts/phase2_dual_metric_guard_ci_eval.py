#!/usr/bin/env python
"""Dual-metric guarded collaborative inference for Stage2-C evidence.

DMG-CI combines two already-sealed evidence streams:

* base ADV3B02/qknn8 evidence for old-class retention;
* source-heldout hard-negative metric evidence for seen-new rescue and reject
  risk.

It does not fit thresholds or profiles from target_unknown. target_unknown rows
are consumed only by the final evaluator.
"""

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

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence  # noqa: E402


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _str(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return str(default if value is None else value)


def _clip01(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _items(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.replace(";", ",").split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _str(row, "event_id"),
        _str(row, "receiver_id"),
        _str(row, "role"),
        _str(row, "true_label"),
    )


def merge_dual_metric_rows(
    base_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    old_labels: set[str],
    seen_new_labels: set[str],
    metric_rescue_min_score: float = 0.78,
    metric_rescue_min_margin: float = 0.12,
    metric_reject_risk: float = 0.86,
    base_old_core_min_score: float = 0.88,
    base_old_core_min_margin: float = 0.18,
    old_core_risk_cap: float = 0.32,
    metric_risk_weight: float = 1.0,
    base_risk_weight: float = 0.35,
    evidence_packet_bytes: float = 160.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_by_key = {_row_key(row): row for row in metric_rows}
    missing = sorted(set(_row_key(row) for row in base_rows) - set(metric_by_key))
    if missing:
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: metric evidence missing rows for keys={missing[:3]}")

    merged: list[dict[str, Any]] = []
    route_counts: dict[str, int] = {}
    for base in base_rows:
        metric = metric_by_key[_row_key(base)]
        row = dict(base)
        base_label = _str(base, "predicted_label")
        metric_label = _str(metric, "predicted_label")
        base_score = _clip01(_float(base, "known_score", 0.0))
        metric_score = _clip01(_float(metric, "known_score", 0.0))
        base_margin = _float(base, "known_margin", _float(base, "label_score_gap", 0.0))
        metric_margin = _float(metric, "known_margin", _float(metric, "label_score_gap", 0.0))
        base_risk = _clip01(_float(base, "unknown_risk", 0.0))
        metric_risk = _clip01(_float(metric, "unknown_risk", 0.0))

        base_old_core = (
            base_label in old_labels
            and base_score >= float(base_old_core_min_score)
            and base_margin >= float(base_old_core_min_margin)
        )
        metric_seen_rescue = (
            metric_label in seen_new_labels
            and metric_score >= float(metric_rescue_min_score)
            and metric_margin >= float(metric_rescue_min_margin)
            and not base_old_core
        )
        metric_reject_guard = metric_risk >= float(metric_reject_risk) and not base_old_core

        if base_old_core and metric_label == base_label:
            route = "base_old_core"
            predicted = base_label
            known_score = max(base_score, metric_score)
            margin = max(base_margin, metric_margin)
            risk = min(max(base_risk, 0.5 * metric_risk), float(old_core_risk_cap))
        elif metric_seen_rescue:
            route = "metric_seen_new_rescue"
            predicted = metric_label
            known_score = max(metric_score, base_score)
            margin = max(metric_margin, base_margin)
            risk = min(metric_risk, max(0.05, 1.0 - metric_score))
        elif metric_reject_guard:
            route = "metric_reject_guard"
            predicted = base_label
            known_score = min(base_score, metric_score)
            margin = min(base_margin, metric_margin)
            risk = max(base_risk * float(base_risk_weight), metric_risk * float(metric_risk_weight))
        else:
            route = "base_guarded"
            predicted = base_label
            known_score = max(base_score, 0.5 * metric_score)
            margin = max(base_margin, 0.5 * metric_margin)
            risk = max(base_risk, metric_risk * 0.65)

        route_counts[route] = route_counts.get(route, 0) + 1
        row.update(
            {
                "predicted_label": predicted,
                "known_score": float(_clip01(known_score)),
                "known_margin": float(margin),
                "label_score_gap": float(margin),
                "unknown_risk": float(_clip01(risk)),
                "score_risk": float(_clip01(risk)),
                "radius_risk": float(_clip01(risk)),
                "class_negative_risk": float(metric_risk),
                "dual_metric_route": route,
                "dual_metric_base_label": base_label,
                "dual_metric_metric_label": metric_label,
                "dual_metric_base_known_score": float(base_score),
                "dual_metric_metric_known_score": float(metric_score),
                "dual_metric_base_unknown_risk": float(base_risk),
                "dual_metric_metric_unknown_risk": float(metric_risk),
                "threshold_selection_label_scope": "source_only",
                "calibration_role": "query",
                "bytes": float(evidence_packet_bytes),
                "latency_ms": max(_float(base, "latency_ms", 0.0), _float(metric, "latency_ms", 0.0)) + 0.05,
            }
        )
        merged.append(row)

    metadata = {
        "algorithm": "DMG-CI",
        "base_evidence_rows": len(base_rows),
        "metric_evidence_rows": len(metric_rows),
        "merged_evidence_rows": len(merged),
        "old_tx_ids": sorted(old_labels),
        "seen_new_tx_ids": sorted(seen_new_labels),
        "target_unknown_eval_only": True,
        "target_unknown_training_count": 0,
        "threshold_uses_target_unknown": False,
        "profile_selection_uses_target_unknown": False,
        "training_roles": ["base_evidence_eval", "source_proxy_metric_evidence_eval"],
        "threshold_scope": "source_only",
        "qknn_k": 8,
        "route_counts": dict(sorted(route_counts.items())),
        "state_size_bytes": 0,
        "evidence_bytes_per_receiver_event": float(evidence_packet_bytes),
    }
    return merged, metadata


def _best_joint_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def score(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row.get("target_pass", False) is True),
            _float(row, "old_acc", 0.0),
            _float(row, "seen_new_acc", 0.0),
            _float(row, "unknown_reject_rate", 0.0),
            -_float(row, "unknown_FAR", 1.0),
        )

    return dict(max(rows, key=score))


def _summary_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("summary_rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows]
    counts = result.get("counts")
    if isinstance(counts, Mapping):
        return [
            dict(row)
            for _, row in sorted(
                counts.items(),
                key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
            )
        ]
    return []


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_evidence_csv", type=Path, required=True)
    p.add_argument("--metric_evidence_csv", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_evidence_csv", type=Path, default=None)
    p.add_argument("--old_labels", required=True)
    p.add_argument("--seen_new_labels", required=True)
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--partial_collab_min_receivers", type=int, default=1)
    p.add_argument("--unknown_risk_threshold", type=float, default=0.86)
    p.add_argument("--accept_margin_threshold", type=float, default=0.08)
    p.add_argument("--metric_rescue_min_score", type=float, default=0.78)
    p.add_argument("--metric_rescue_min_margin", type=float, default=0.12)
    p.add_argument("--metric_reject_risk", type=float, default=0.86)
    p.add_argument("--base_old_core_min_score", type=float, default=0.88)
    p.add_argument("--base_old_core_min_margin", type=float, default=0.18)
    p.add_argument("--old_core_risk_cap", type=float, default=0.32)
    p.add_argument("--evidence_packet_bytes", type=float, default=160.0)
    p.add_argument("--max_event_bytes", type=float, default=1152.0)
    p.add_argument("--max_event_latency_ms", type=float, default=20.0)
    p.add_argument("--allow_target_pass", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    merged, metadata = merge_dual_metric_rows(
        _read_csv(args.base_evidence_csv),
        _read_csv(args.metric_evidence_csv),
        old_labels=_items(args.old_labels),
        seen_new_labels=_items(args.seen_new_labels),
        metric_rescue_min_score=float(args.metric_rescue_min_score),
        metric_rescue_min_margin=float(args.metric_rescue_min_margin),
        metric_reject_risk=float(args.metric_reject_risk),
        base_old_core_min_score=float(args.base_old_core_min_score),
        base_old_core_min_margin=float(args.base_old_core_min_margin),
        old_core_risk_cap=float(args.old_core_risk_cap),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
    )
    result = evaluate_collaborative_open_set_evidence(
        merged,
        collab_counts=args.collab_counts,
        threshold_selection_label_scope="source_only",
        unknown_query_eval_only=True,
        protocol_metadata=metadata,
        strict_protocol_metadata=False,
        unknown_risk_threshold=float(args.unknown_risk_threshold),
        accept_margin_threshold=float(args.accept_margin_threshold),
        fusion_policy="risk_margin",
        label_fusion_policy="score_sum",
        collab_group_policy=str(args.collab_group_policy),
        partial_collab_min_receivers=int(args.partial_collab_min_receivers),
        max_event_bytes=float(args.max_event_bytes),
        max_event_latency_ms=float(args.max_event_latency_ms),
    )
    result["summary_rows"] = _summary_rows(result)
    for row in result.get("summary_rows", []):
        row["metric_target_pass"] = bool(row.get("target_pass", False))
        if not bool(args.allow_target_pass):
            row["target_pass"] = False
    result["best_joint_row"] = _best_joint_row(result.get("summary_rows", []))
    result["dual_metric_guard_metadata"] = metadata | {
        "allow_target_pass": bool(args.allow_target_pass),
        "target_pass_is_diagnostic_gated": not bool(args.allow_target_pass),
    }
    result["base_evidence_csv"] = str(args.base_evidence_csv)
    result["metric_evidence_csv"] = str(args.metric_evidence_csv)
    result["output_json"] = str(args.output_json)
    if args.output_evidence_csv is not None:
        _write_csv(args.output_evidence_csv, merged)
        result["output_evidence_csv"] = str(args.output_evidence_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
