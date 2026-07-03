#!/usr/bin/env python
"""Fuse qknn8 safety decisions with a second known-label evidence route.

This diagnostic evaluator keeps one decision rule for all roles:

1. A qknn8 safety route decides accept / unknown_reject / request_more / defer.
2. Only when the safety route accepts, the known route may replace the accepted
   label. Rejected/deferred/request-more events stay safety-controlled.

The script does not fit thresholds and does not use unknown query labels for
calibration. True labels are used only for final metric accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

UNKNOWN_LABEL = "__unknown__"


def _read_evidence_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def _rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if float(den) else 0.0


def _event_map(result_count: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(event["event_id"]): event for event in result_count.get("event_results", [])}


def _min_class_acc(counter: Mapping[str, Sequence[int]]) -> float:
    values = [correct / total for correct, total in counter.values() if total]
    return min(values) if values else 0.0


def fuse_dual_evidence_event_results(
    safety_count: Mapping[str, Any],
    known_count: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse one collaboration-count bucket from safety and known routes."""

    safety_events = _event_map(safety_count)
    known_events = _event_map(known_count)
    old_total = old_correct = 0
    seen_total = seen_correct = 0
    unknown_total = unknown_rejected = unknown_false_accept = unknown_defer = unknown_request_more = 0
    defer_total = request_more_total = 0
    old_classes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    seen_classes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    open_set_confusion: Counter[str] = Counter()
    fused_events: list[dict[str, Any]] = []

    for event_id, safety in sorted(safety_events.items()):
        known = known_events.get(event_id)
        role = str(safety.get("role", ""))
        true_label = str(safety.get("true_label", UNKNOWN_LABEL if role == "unknown" else ""))
        safety_decision = str(safety.get("decision", "defer"))
        if safety_decision == "unknown_reject":
            decision = "unknown_reject"
            output_label = UNKNOWN_LABEL
            label_source = "qknn8_safety_reject"
        elif safety_decision in {"defer", "request_more"}:
            decision = safety_decision
            output_label = ""
            label_source = "qknn8_safety_unresolved"
        elif safety_decision == "accept":
            if known is not None and str(known.get("decision", "")) == "accept" and str(known.get("output_label", "")):
                decision = "accept"
                output_label = str(known.get("output_label", ""))
                label_source = "known_route_safe_accept"
            else:
                decision = "accept"
                output_label = str(safety.get("output_label", ""))
                label_source = "qknn8_safety_accept"
        else:
            decision = "defer"
            output_label = ""
            label_source = "unknown_safety_decision"

        if decision == "defer":
            defer_total += 1
        if decision == "request_more":
            request_more_total += 1

        if role == "old":
            old_total += 1
            old_classes[true_label][1] += 1
            if decision == "accept" and output_label == true_label:
                old_correct += 1
                old_classes[true_label][0] += 1
        elif role == "seen_new":
            seen_total += 1
            seen_classes[true_label][1] += 1
            if decision == "accept" and output_label == true_label:
                seen_correct += 1
                seen_classes[true_label][0] += 1
        elif role == "unknown":
            unknown_total += 1
            if decision == "unknown_reject":
                unknown_rejected += 1
                open_set_confusion["unknown->unknown_reject"] += 1
            elif decision == "accept":
                unknown_false_accept += 1
                open_set_confusion[f"unknown->{output_label}"] += 1
            elif decision == "request_more":
                unknown_request_more += 1
                open_set_confusion["unknown->request_more"] += 1
            else:
                unknown_defer += 1
                open_set_confusion["unknown->defer"] += 1

        fused_events.append(
            {
                "event_id": event_id,
                "role": role,
                "true_label": true_label,
                "decision": decision,
                "output_label": output_label,
                "safety_decision": safety_decision,
                "safety_output_label": str(safety.get("output_label", "")),
                "known_output_label": str(known.get("output_label", "")) if known is not None else "",
                "label_source": label_source,
            }
        )

    total = len(fused_events)
    known_total = old_total + seen_total
    known_correct = old_correct + seen_correct
    known_accepted = sum(1 for event in fused_events if event["role"] in {"old", "seen_new"} and event["decision"] == "accept")
    return {
        "total": total,
        "old_total": old_total,
        "old_correct": old_correct,
        "old_acc": _rate(old_correct, old_total),
        "per_old_class_acc": {label: _rate(v[0], v[1]) for label, v in sorted(old_classes.items())},
        "min_old_class_acc": _min_class_acc(old_classes),
        "seen_new_total": seen_total,
        "seen_new_correct": seen_correct,
        "seen_new_acc": _rate(seen_correct, seen_total),
        "per_seen_new_class_acc": {label: _rate(v[0], v[1]) for label, v in sorted(seen_classes.items())},
        "min_seen_new_class_acc": _min_class_acc(seen_classes),
        "unknown_total": unknown_total,
        "unknown_rejected": unknown_rejected,
        "unknown_reject_rate": _rate(unknown_rejected, unknown_total),
        "unknown_FAR": _rate(unknown_false_accept, unknown_total),
        "unknown_defer": unknown_defer,
        "unknown_defer_rate": _rate(unknown_defer, unknown_total),
        "unknown_request_more": unknown_request_more,
        "unknown_request_more_rate": _rate(unknown_request_more, unknown_total),
        "known_coverage": _rate(known_accepted, known_total),
        "known_full_accuracy": _rate(known_correct, known_total),
        "defer_rate": _rate(defer_total, total),
        "request_more_rate": _rate(request_more_total, total),
        "open_set_confusion": dict(sorted(open_set_confusion.items())),
        "bytes_per_event": safety_count.get("bytes_per_event", 0.0),
        "latency_ms_p95": safety_count.get("latency_ms_p95", 0.0),
        "event_results": fused_events,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--safety_evidence_csv", type=Path, required=True)
    p.add_argument("--known_evidence_csv", type=Path, required=True)
    p.add_argument("--protocol_metadata_json", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="available_up_to_k")
    p.add_argument("--partial_collab_min_receivers", type=int, default=3)
    p.add_argument("--unknown_quantile", type=float, default=0.75)
    p.add_argument("--safety_fusion_policy", default="candidate_set_cvs")
    p.add_argument("--safety_label_fusion_policy", default="weighted_vote_margin")
    p.add_argument("--safety_unknown_risk_threshold", type=float, default=0.8)
    p.add_argument("--safety_accept_margin_threshold", type=float, default=0.1)
    p.add_argument("--candidate_set_unknown_reject_risk", type=float, default=0.8)
    p.add_argument("--candidate_set_min_receivers", type=int, default=2)
    p.add_argument("--candidate_set_min_top1_receivers", type=int, default=0)
    p.add_argument("--candidate_set_min_conformal_pvalue", type=float, default=0.5)
    p.add_argument("--candidate_set_max_label_unknown_risk", type=float, default=1.0)
    p.add_argument("--candidate_set_max_event_unknown_risk", type=float, default=1.0)
    p.add_argument("--candidate_set_max_label_risk_component_agreement", type=float, default=0.33)
    p.add_argument("--candidate_set_max_receiver_pair_label_disagreement", type=float, default=0.5)
    p.add_argument("--candidate_set_max_receiver_pair_unknown_risk_range", type=float, default=0.7)
    p.add_argument("--candidate_set_pairguard_mode", default="support_calibrated")
    p.add_argument("--candidate_set_pairguard_action", default="soft_penalty")
    p.add_argument("--candidate_set_pairguard_soft_penalty", type=float, default=0.2)
    p.add_argument("--candidate_set_pairguard_soft_floor", type=float, default=0.03)
    p.add_argument("--candidate_set_pairguard_soft_min_margin", type=float, default=0.18)
    p.add_argument("--candidate_set_pairguard_soft_min_pvalue", type=float, default=0.6)
    p.add_argument("--candidate_set_pairguard_soft_min_reliability", type=float, default=0.8)
    p.add_argument("--candidate_set_pairguard_min_event_unknown_risk", type=float, default=0.98)
    p.add_argument("--candidate_set_pairguard_min_label_unknown_risk", type=float, default=1.0)
    p.add_argument("--candidate_set_pairguard_min_shell_risk", type=float, default=1.0)
    p.add_argument("--known_fusion_policy", default="risk_margin")
    p.add_argument("--known_label_fusion_policy", default="weighted_vote_margin")
    p.add_argument("--known_unknown_risk_threshold", type=float, default=0.65)
    p.add_argument("--known_accept_margin_threshold", type=float, default=0.02)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    safety_rows = _read_evidence_csv(args.safety_evidence_csv)
    known_rows = _read_evidence_csv(args.known_evidence_csv)
    metadata_payload = json.loads(args.protocol_metadata_json.read_text(encoding="utf-8"))
    protocol_metadata = metadata_payload.get("qknn_metadata") or metadata_payload.get("support_ridge_metadata") or metadata_payload
    safety = evaluate_collaborative_open_set_evidence(
        safety_rows,
        collab_counts=args.collab_counts,
        collab_group_policy=args.collab_group_policy,
        partial_collab_min_receivers=int(args.partial_collab_min_receivers),
        threshold_selection_label_scope="support_known_only",
        unknown_query_eval_only=True,
        protocol_metadata=protocol_metadata,
        strict_protocol_metadata=True,
        include_event_results=True,
        fusion_policy=str(args.safety_fusion_policy),
        label_fusion_policy=str(args.safety_label_fusion_policy),
        unknown_risk_threshold=float(args.safety_unknown_risk_threshold),
        accept_margin_threshold=float(args.safety_accept_margin_threshold),
        unknown_quantile=float(args.unknown_quantile),
        class_reliability_policy="conformal_margin_risk",
        receiver_class_reliability_policy="support_calibrated",
        candidate_set_unknown_reject_risk=float(args.candidate_set_unknown_reject_risk),
        candidate_set_min_receivers=int(args.candidate_set_min_receivers),
        candidate_set_min_top1_receivers=int(args.candidate_set_min_top1_receivers),
        candidate_set_min_conformal_pvalue=float(args.candidate_set_min_conformal_pvalue),
        candidate_set_max_label_unknown_risk=float(args.candidate_set_max_label_unknown_risk),
        candidate_set_max_event_unknown_risk=float(args.candidate_set_max_event_unknown_risk),
        candidate_set_max_label_risk_component_agreement=float(args.candidate_set_max_label_risk_component_agreement),
        candidate_set_max_receiver_pair_label_disagreement=float(args.candidate_set_max_receiver_pair_label_disagreement),
        candidate_set_max_receiver_pair_unknown_risk_range=float(args.candidate_set_max_receiver_pair_unknown_risk_range),
        candidate_set_pairguard_mode=str(args.candidate_set_pairguard_mode),
        candidate_set_pairguard_action=str(args.candidate_set_pairguard_action),
        candidate_set_pairguard_soft_penalty=float(args.candidate_set_pairguard_soft_penalty),
        candidate_set_pairguard_soft_floor=float(args.candidate_set_pairguard_soft_floor),
        candidate_set_pairguard_soft_min_margin=float(args.candidate_set_pairguard_soft_min_margin),
        candidate_set_pairguard_soft_min_pvalue=float(args.candidate_set_pairguard_soft_min_pvalue),
        candidate_set_pairguard_soft_min_reliability=float(args.candidate_set_pairguard_soft_min_reliability),
        candidate_set_pairguard_min_event_unknown_risk=float(args.candidate_set_pairguard_min_event_unknown_risk),
        candidate_set_pairguard_min_label_unknown_risk=float(args.candidate_set_pairguard_min_label_unknown_risk),
        candidate_set_pairguard_min_shell_risk=float(args.candidate_set_pairguard_min_shell_risk),
    )
    known = evaluate_collaborative_open_set_evidence(
        known_rows,
        collab_counts=args.collab_counts,
        collab_group_policy=args.collab_group_policy,
        partial_collab_min_receivers=int(args.partial_collab_min_receivers),
        threshold_selection_label_scope="support_known_only",
        unknown_query_eval_only=True,
        protocol_metadata=protocol_metadata,
        strict_protocol_metadata=True,
        include_event_results=True,
        fusion_policy=str(args.known_fusion_policy),
        label_fusion_policy=str(args.known_label_fusion_policy),
        unknown_risk_threshold=float(args.known_unknown_risk_threshold),
        accept_margin_threshold=float(args.known_accept_margin_threshold),
        unknown_quantile=float(args.unknown_quantile),
    )
    counts = {
        str(k): fuse_dual_evidence_event_results(safety["counts"][str(k)], known["counts"][str(k)])
        for k in sorted(safety["counts"], key=lambda value: int(value))
    }
    result = {
        "algorithm": "dual_evidence_safety_rescue",
        "receiver_count": safety.get("receiver_count"),
        "counts": counts,
        "safety_source": str(args.safety_evidence_csv),
        "known_source": str(args.known_evidence_csv),
        "protocol_metadata": protocol_metadata,
        "unknown_query_eval_only": True,
        "threshold_selection_label_scope": "support_known_only",
        "diagnostic_only": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
