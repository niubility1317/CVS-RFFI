#!/usr/bin/env python
"""Evaluate a minimal SCORPION-CVS collaborative open-set policy.

This offline diagnostic consumes qknn8 evidence rows generated under the CVS
Stage2-C protocol. It does not fit thresholds from unknown query labels. The
old-label set is recovered from the known Stage2-C old transmitter identities
recorded in the evidence table unless explicitly provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


UNKNOWN_LABEL = "__unknown__"
ROLE_ALIASES = {
    "old": "old",
    "target_old": "old",
    "seen_new": "seen_new",
    "target_new": "seen_new",
    "new": "seen_new",
    "unknown": "unknown",
    "target_unknown": "unknown",
}


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    return value if math.isfinite(value) else float(default)


def _clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _role(row: Mapping[str, Any]) -> str:
    raw = str(row.get("role", row.get("raw_role", ""))).strip().lower().replace("-", "_")
    if raw not in ROLE_ALIASES:
        raise ValueError(f"unknown role {raw!r}")
    return ROLE_ALIASES[raw]


def _parse_weighted_components(spec: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for part in str(spec).replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        if ":" in text:
            key, weight = text.split(":", 1)
            pairs.append((key.strip(), float(weight)))
        else:
            pairs.append((text, 1.0))
    if not pairs:
        raise ValueError("at least one risk component is required")
    return pairs


def read_evidence_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def scorpion_risk(row: Mapping[str, Any], components: Sequence[tuple[str, float]]) -> float:
    total = 0.0
    weight_total = 0.0
    for key, weight in components:
        if key not in row:
            continue
        total += max(0.0, float(weight)) * _clamp01(_float(row, key, 0.0))
        weight_total += max(0.0, float(weight))
    if weight_total <= 0.0:
        return _clamp01(_float(row, "unknown_risk", 0.0))
    return _clamp01(total / weight_total)


def receiver_quality(row: Mapping[str, Any], risk: float) -> float:
    reliability = _clamp01(_float(row, "reliability", 1.0))
    class_reliability = _clamp01(_float(row, "receiver_class_reliability", 1.0))
    deployment_prior = _clamp01(_float(row, "receiver_deployment_prior", 1.0))
    density = _clamp01(_float(row, "support_density", 0.5))
    return max(1.0e-6, reliability * class_reliability * deployment_prior * (0.5 + 0.5 * density) * (1.0 - 0.5 * _clamp01(risk)))


def _label_support(row: Mapping[str, Any]) -> float:
    score = _clamp01(_float(row, "known_score", _float(row, "class_evidence_top1_score", 0.0)))
    margin = max(0.0, _float(row, "known_margin", _float(row, "label_score_gap", 0.0)))
    pvalue = _clamp01(_float(row, "class_evidence_top1_conformal_pvalue", 0.0))
    threshold = max(0.0, _float(row, "effective_score_threshold", _float(row, "receiver_score_threshold", 0.0)))
    score_excess = max(0.0, score - threshold)
    return score + 0.35 * margin + 0.25 * pvalue + 0.5 * score_excess


def _event_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        event_id = str(row.get("event_id", "")).strip() or f"row-{index}"
        grouped[event_id].append(row)
    return dict(grouped)


def infer_old_labels(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row.get("true_label", "")).strip()
        for row in rows
        if _role(row) == "old" and str(row.get("true_label", "")).strip()
    }


def _min_class_accuracy(correct: Mapping[str, int], total: Mapping[str, int]) -> float:
    values = [correct.get(label, 0) / count for label, count in total.items() if count > 0]
    return min(values) if values else 0.0


def evaluate_scorpion(
    rows: Sequence[Mapping[str, Any]],
    *,
    collab_counts: Sequence[int] | None = None,
    old_labels: set[str] | None = None,
    risk_components: Sequence[tuple[str, float]],
    unknown_gate: float = 0.52,
    old_shield_gate: float = 0.68,
    min_margin: float = 0.02,
    min_pvalue: float = 0.0,
    min_quality: float = 0.0,
    evidence_packet_bytes: float | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no evidence rows")
    old_label_set = set(old_labels or infer_old_labels(rows))
    receiver_ids = sorted({str(row.get("receiver_id", "")) for row in rows if str(row.get("receiver_id", ""))})
    receiver_count = len(receiver_ids)
    if receiver_count <= 0:
        raise ValueError("evidence rows must include receiver_id")
    counts = list(collab_counts or range(1, receiver_count + 1))
    for count in counts:
        if int(count) < 1 or int(count) > receiver_count:
            raise ValueError(f"collab count {count} outside 1..{receiver_count}")

    groups = _event_groups(rows)
    out: dict[str, Any] = {
        "algorithm": "SCORPION-CVS-minimal-evidence-diagnostic",
        "diagnostic_only": True,
        "threshold_selection_label_scope": "support_known_only_or_fixed_prior",
        "unknown_query_used_for_threshold_fit": False,
        "receiver_count": receiver_count,
        "receiver_ids": receiver_ids,
        "old_labels": sorted(old_label_set),
        "risk_components": [{"field": field, "weight": weight} for field, weight in risk_components],
        "counts": {},
        "rows": [],
    }

    for count in counts:
        total_by_role = Counter()
        correct_by_role = Counter()
        rejected_unknown = 0
        false_accept_unknown = 0
        defer_known = 0
        old_class_total: Counter[str] = Counter()
        old_class_correct: Counter[str] = Counter()
        seen_class_total: Counter[str] = Counter()
        seen_class_correct: Counter[str] = Counter()
        bytes_values = []
        latency_values = []
        row_outputs = []

        for event_id, event_rows in sorted(groups.items()):
            enriched = []
            for row in event_rows:
                risk = scorpion_risk(row, risk_components)
                quality = receiver_quality(row, risk)
                enriched.append((quality, risk, row))
            enriched.sort(key=lambda item: (-item[0], str(item[2].get("receiver_id", ""))))
            selected = enriched[: int(count)]
            if not selected:
                continue

            role = _role(selected[0][2])
            true_label = str(selected[0][2].get("true_label", UNKNOWN_LABEL if role == "unknown" else "")).strip()
            total_by_role[role] += 1
            if role == "old":
                old_class_total[true_label] += 1
            elif role == "seen_new":
                seen_class_total[true_label] += 1

            label_scores: dict[str, float] = defaultdict(float)
            label_votes = Counter()
            risk_weight_total = 0.0
            weighted_risk = 0.0
            high_risk = 0
            local_accepts = 0
            selected_receivers = []
            selected_bytes = 0.0
            selected_latency = 0.0
            for quality, risk, row in selected:
                label = str(row.get("predicted_label", "")).strip()
                selected_receivers.append(str(row.get("receiver_id", "")))
                if risk >= 0.80:
                    high_risk += 1
                weighted_risk += quality * risk
                risk_weight_total += quality
                margin = _float(row, "known_margin", _float(row, "label_score_gap", 0.0))
                pvalue = _float(row, "class_evidence_top1_conformal_pvalue", 0.0)
                if (
                    risk <= float(old_shield_gate)
                    and margin >= float(min_margin)
                    and pvalue >= float(min_pvalue)
                    and quality >= float(min_quality)
                ):
                    local_accepts += 1
                    if label:
                        label_scores[label] += quality * _label_support(row)
                        label_votes[label] += 1
                selected_bytes += float(evidence_packet_bytes) if evidence_packet_bytes is not None else _float(row, "bytes", 0.0)
                selected_latency = max(selected_latency, _float(row, "latency_ms", 0.0))

            event_unknown_risk = weighted_risk / risk_weight_total if risk_weight_total > 0.0 else 1.0
            high_risk_fraction = high_risk / max(1, len(selected))
            if label_scores:
                pred_label, pred_score = max(label_scores.items(), key=lambda item: (item[1], item[0]))
                vote_fraction = label_votes[pred_label] / max(1, len(selected))
            else:
                pred_label = ""
                pred_score = 0.0
                vote_fraction = 0.0
            disagreement = 1.0 - vote_fraction
            event_unknown_score = _clamp01(0.72 * event_unknown_risk + 0.18 * high_risk_fraction + 0.10 * disagreement)
            old_shield = bool(
                pred_label in old_label_set
                and event_unknown_score < float(old_shield_gate)
                and vote_fraction >= 0.50
                and local_accepts >= max(1, math.ceil(len(selected) / 2))
            )
            accept = bool(
                pred_label
                and local_accepts >= max(1, math.ceil(len(selected) / 2))
                and (event_unknown_score < float(unknown_gate) or old_shield)
            )
            rejected = not accept
            if role == "unknown":
                if rejected:
                    rejected_unknown += 1
                else:
                    false_accept_unknown += 1
            else:
                if rejected:
                    defer_known += 1
                elif pred_label == true_label:
                    correct_by_role[role] += 1
                    if role == "old":
                        old_class_correct[true_label] += 1
                    elif role == "seen_new":
                        seen_class_correct[true_label] += 1

            row_outputs.append(
                {
                    "event_id": event_id,
                    "role": role,
                    "true_label": true_label,
                    "predicted_label": pred_label if accept else UNKNOWN_LABEL,
                    "accepted_label": pred_label,
                    "reject": bool(rejected),
                    "old_shield": bool(old_shield),
                    "event_unknown_score": float(event_unknown_score),
                    "event_unknown_risk": float(event_unknown_risk),
                    "high_risk_fraction": float(high_risk_fraction),
                    "vote_fraction": float(vote_fraction),
                    "local_accepts": int(local_accepts),
                    "receiver_count": int(count),
                    "selected_receivers": selected_receivers,
                    "bytes_per_event": float(selected_bytes),
                    "latency_ms": float(selected_latency),
                }
            )
            bytes_values.append(selected_bytes)
            latency_values.append(selected_latency)

        old_total = total_by_role["old"]
        seen_total = total_by_role["seen_new"]
        unknown_total = total_by_role["unknown"]
        metrics = {
            "event_count": int(sum(total_by_role.values())),
            "role_counts": dict(sorted(total_by_role.items())),
            "old_acc": correct_by_role["old"] / old_total if old_total else 0.0,
            "min_old_class_acc": _min_class_accuracy(old_class_correct, old_class_total),
            "seen_new_acc": correct_by_role["seen_new"] / seen_total if seen_total else 0.0,
            "min_seen_new_class_acc": _min_class_accuracy(seen_class_correct, seen_class_total),
            "unknown_reject_rate": rejected_unknown / unknown_total if unknown_total else 0.0,
            "unknown_FAR": false_accept_unknown / unknown_total if unknown_total else 0.0,
            "known_defer_rate": defer_known / max(1, old_total + seen_total),
            "bytes_per_event_mean": sum(bytes_values) / len(bytes_values) if bytes_values else 0.0,
            "latency_ms_pessimistic": max(latency_values) if latency_values else 0.0,
            "participants_per_event": int(count),
        }
        out["counts"][str(int(count))] = metrics
        for item in row_outputs:
            item["count_metrics"] = metrics
            out["rows"].append(item)
    return out


def _parse_ints(spec: str | None, receiver_count: int | None = None) -> list[int] | None:
    if spec is None or str(spec).strip().lower() in {"", "all", "1..n", "*"}:
        return None if receiver_count is None else list(range(1, receiver_count + 1))
    out = []
    for part in str(spec).replace(";", ",").split(","):
        if part.strip():
            out.append(int(part))
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_rows_csv", type=Path)
    parser.add_argument("--collab_counts", default="all")
    parser.add_argument("--old_labels", default="")
    parser.add_argument(
        "--risk_components",
        default="mahalanobis_risk:0.30,evt_risk:0.20,margin_risk:0.20,oldness_risk:0.20,score_risk:0.10",
    )
    parser.add_argument("--unknown_gate", type=float, default=0.52)
    parser.add_argument("--old_shield_gate", type=float, default=0.68)
    parser.add_argument("--min_margin", type=float, default=0.02)
    parser.add_argument("--min_pvalue", type=float, default=0.0)
    parser.add_argument("--min_quality", type=float, default=0.0)
    parser.add_argument("--evidence_packet_bytes", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_evidence_csv(args.evidence_csv)
    old_labels = {part.strip() for part in str(args.old_labels).split(",") if part.strip()} or None
    result = evaluate_scorpion(
        rows,
        collab_counts=_parse_ints(args.collab_counts),
        old_labels=old_labels,
        risk_components=_parse_weighted_components(args.risk_components),
        unknown_gate=float(args.unknown_gate),
        old_shield_gate=float(args.old_shield_gate),
        min_margin=float(args.min_margin),
        min_pvalue=float(args.min_pvalue),
        min_quality=float(args.min_quality),
        evidence_packet_bytes=args.evidence_packet_bytes,
    )
    result["evidence_csv"] = str(args.evidence_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_rows_csv:
        args.output_rows_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "receiver_count",
            "event_id",
            "role",
            "true_label",
            "predicted_label",
            "accepted_label",
            "reject",
            "old_shield",
            "event_unknown_score",
            "event_unknown_risk",
            "high_risk_fraction",
            "vote_fraction",
            "local_accepts",
            "selected_receivers",
            "bytes_per_event",
            "latency_ms",
        ]
        with args.output_rows_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in result["rows"]:
                out_row = dict(row)
                out_row["selected_receivers"] = "|".join(out_row["selected_receivers"])
                writer.writerow({key: out_row.get(key, "") for key in fieldnames})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
