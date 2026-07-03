from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from typing import Any, Mapping, Sequence


UNKNOWN_LABEL = "__unknown__"


def parse_collab_counts(spec: str | Sequence[int] | None, *, receiver_count: int) -> list[int]:
    receiver_count = int(receiver_count)
    if receiver_count < 1:
        raise ValueError(f"receiver_count must be >= 1, got {receiver_count}")
    if spec is None or str(spec).strip().lower() in {"", "all", "*", "1..n"}:
        return list(range(1, receiver_count + 1))
    if isinstance(spec, str):
        items = [part.strip() for part in spec.replace(";", ",").split(",") if part.strip()]
    else:
        items = [str(part) for part in spec]
    counts: list[int] = []
    for item in items:
        k = int(item)
        if k < 1 or k > receiver_count:
            raise ValueError(f"collaborative receiver count {k} is outside valid range 1..{receiver_count}")
        if k not in counts:
            counts.append(k)
    if not counts:
        raise ValueError("no collaborative receiver counts were requested")
    return counts


def _role(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"new", "seennew", "seen_new", "target_new"}:
        return "seen_new"
    if text in {"unk", "unknown", "target_unknown"}:
        return "unknown"
    return "old"


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _str(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return str(default if value is None else value)


def _percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    pairs = sorted(
        (float(v), max(0.0, float(w)))
        for v, w in zip(values, weights)
        if math.isfinite(float(v)) and math.isfinite(float(w))
    )
    if not pairs:
        return 0.0
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return _percentile([value for value, _ in pairs], q)
    cutoff = max(0.0, min(1.0, float(q))) * total
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= cutoff:
            return value
    return pairs[-1][0]


def _safe_rate(num: int, den: int) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if _role(row.get("role")) == "unknown" and str(row.get("calibration_role", "")).lower() in {
            "threshold_fit",
            "calibration",
            "fit",
            "train",
        }:
            raise ValueError("unknown query rows cannot be used for threshold fitting or calibration")


def _select_receivers(rows: Sequence[Mapping[str, Any]], k: int) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -_float(row, "reliability", 1.0),
            _float(row, "latency_ms", 0.0),
            _str(row, "receiver_id"),
        ),
    )[: int(k)]


def _fuse_event(
    selected: Sequence[Mapping[str, Any]],
    *,
    unknown_risk_threshold: float,
    accept_margin_threshold: float,
    unknown_quantile: float,
) -> dict[str, Any]:
    label_scores: defaultdict[str, float] = defaultdict(float)
    weights = []
    risks = []
    margins = []
    for row in selected:
        weight = max(0.0, _float(row, "reliability", 1.0))
        label = _str(row, "predicted_label", "")
        if label and label != UNKNOWN_LABEL:
            label_scores[label] += weight * max(0.0, _float(row, "known_score", 1.0))
        weights.append(weight)
        risks.append(_float(row, "unknown_risk", 0.0))
        margins.append(_float(row, "known_margin", 0.0))

    unknown_risk = _weighted_quantile(risks, weights, unknown_quantile)
    mean_margin = 0.0
    if weights and sum(weights) > 0:
        mean_margin = sum(m * max(0.0, w) for m, w in zip(margins, weights)) / max(sum(weights), 1e-12)
    elif margins:
        mean_margin = sum(margins) / len(margins)

    if label_scores:
        label, score = max(label_scores.items(), key=lambda item: (item[1], item[0]))
    else:
        label, score = "", 0.0

    if unknown_risk >= float(unknown_risk_threshold) and mean_margin < float(accept_margin_threshold):
        decision = "unknown_reject"
        output_label = UNKNOWN_LABEL
    elif label:
        decision = "accept"
        output_label = label
    else:
        decision = "defer"
        output_label = ""

    return {
        "decision": decision,
        "output_label": output_label,
        "unknown_risk": float(unknown_risk),
        "known_margin": float(mean_margin),
        "known_score": float(score),
        "bytes": float(sum(_float(row, "bytes", 0.0) for row in selected)),
        "latency_ms": float(max((_float(row, "latency_ms", 0.0) for row in selected), default=0.0)),
    }


def _finalize_metrics(event_results: Sequence[dict[str, Any]], *, k: int, excluded_incomplete: int) -> dict[str, Any]:
    role_totals = {"old": 0, "seen_new": 0, "unknown": 0}
    role_correct = {"old": 0, "seen_new": 0}
    role_accepted = {"old": 0, "seen_new": 0}
    per_class_total: dict[str, defaultdict[str, int]] = {
        "old": defaultdict(int),
        "seen_new": defaultdict(int),
    }
    per_class_correct: dict[str, defaultdict[str, int]] = {
        "old": defaultdict(int),
        "seen_new": defaultdict(int),
    }
    unknown_rejected = 0
    unknown_false_accept = 0
    defer_total = 0

    for item in event_results:
        role = item["role"]
        truth = item["true_label"]
        output = item["output_label"]
        decision = item["decision"]
        accepted = decision == "accept"
        deferred = decision == "defer"
        if deferred:
            defer_total += 1
        role_totals[role] += 1
        if role in {"old", "seen_new"}:
            role_accepted[role] += int(accepted)
            role_correct[role] += int(accepted and output == truth)
            per_class_total[role][str(truth)] += 1
            per_class_correct[role][str(truth)] += int(accepted and output == truth)
        elif role == "unknown":
            unknown_rejected += int(decision == "unknown_reject")
            unknown_false_accept += int(accepted)

    old_class_rates = {
        label: _safe_rate(per_class_correct["old"][label], total)
        for label, total in sorted(per_class_total["old"].items())
        if label
    }
    new_class_rates = {
        label: _safe_rate(per_class_correct["seen_new"][label], total)
        for label, total in sorted(per_class_total["seen_new"].items())
        if label
    }

    bytes_values = [float(item["bytes"]) for item in event_results]
    latency_values = [float(item["latency_ms"]) for item in event_results]
    known_total = role_totals["old"] + role_totals["seen_new"]
    known_accepted = role_accepted["old"] + role_accepted["seen_new"]
    known_correct = role_correct["old"] + role_correct["seen_new"]
    total_events = len(event_results)

    return {
        "participating_receivers": int(k),
        "total": int(total_events),
        "excluded_incomplete_groups": int(excluded_incomplete),
        "old_total": int(role_totals["old"]),
        "old_correct": int(role_correct["old"]),
        "old_acc": _safe_rate(role_correct["old"], role_totals["old"]),
        "per_old_class_acc": old_class_rates,
        "min_old_class_acc": min(old_class_rates.values()) if old_class_rates else 0.0,
        "seen_new_total": int(role_totals["seen_new"]),
        "seen_new_correct": int(role_correct["seen_new"]),
        "seen_new_acc": _safe_rate(role_correct["seen_new"], role_totals["seen_new"]),
        "per_seen_new_class_acc": new_class_rates,
        "min_seen_new_class_acc": min(new_class_rates.values()) if new_class_rates else 0.0,
        "unknown_total": int(role_totals["unknown"]),
        "unknown_rejected": int(unknown_rejected),
        "unknown_reject_rate": _safe_rate(unknown_rejected, role_totals["unknown"]),
        "unknown_FAR": _safe_rate(unknown_false_accept, role_totals["unknown"]),
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy": _safe_rate(known_correct, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct, known_accepted),
        "defer_rate": _safe_rate(defer_total, total_events),
        "bytes_per_event": sum(bytes_values) / max(len(bytes_values), 1),
        "total_bytes": float(sum(bytes_values)),
        "latency_ms_p50": _percentile(latency_values, 0.50),
        "latency_ms_p95": _percentile(latency_values, 0.95),
    }


def evaluate_collaborative_open_set_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    collab_counts: str | Sequence[int] | None = None,
    unknown_risk_threshold: float = 0.80,
    accept_margin_threshold: float = 0.10,
    unknown_quantile: float = 0.75,
) -> dict[str, Any]:
    """Evaluate offline collaborative open-set qknn-style evidence.

    Each row is one receiver observation for one event. The function does not
    fit thresholds and does not use unknown query rows for calibration.
    """

    rows = list(rows)
    _validate_rows(rows)
    groups: "OrderedDict[str, list[Mapping[str, Any]]]" = OrderedDict()
    receivers: set[str] = set()
    for row in rows:
        event_id = _str(row, "event_id")
        if not event_id:
            raise ValueError("each evidence row must include event_id")
        receiver = _str(row, "receiver_id")
        if not receiver:
            raise ValueError("each evidence row must include receiver_id")
        groups.setdefault(event_id, []).append(row)
        receivers.add(receiver)
    if not groups:
        raise ValueError("no evidence rows were provided")

    receiver_count = len(receivers)
    counts = parse_collab_counts(collab_counts, receiver_count=receiver_count)
    max_requested = max(counts)
    eligible = [(event_id, group) for event_id, group in groups.items() if len({ _str(row, "receiver_id") for row in group }) >= max_requested]
    excluded = len(groups) - len(eligible)
    if not eligible:
        raise ValueError(f"no evidence groups contain {max_requested} receiver observations")

    out_counts: dict[str, Any] = {}
    for k in counts:
        event_results: list[dict[str, Any]] = []
        for _, group in eligible:
            selected = _select_receivers(group, int(k))
            fused = _fuse_event(
                selected,
                unknown_risk_threshold=unknown_risk_threshold,
                accept_margin_threshold=accept_margin_threshold,
                unknown_quantile=unknown_quantile,
            )
            first = selected[0]
            fused["role"] = _role(first.get("role"))
            fused["true_label"] = _str(first, "true_label", UNKNOWN_LABEL if fused["role"] == "unknown" else "")
            event_results.append(fused)
        out_counts[str(k)] = _finalize_metrics(event_results, k=int(k), excluded_incomplete=excluded)

    return {
        "enabled": True,
        "protocol": "collaborative_open_set_qknn_evidence",
        "receiver_count": int(receiver_count),
        "observed_receiver_ids": sorted(receivers),
        "group_count": int(len(groups)),
        "eligible_group_count": int(len(eligible)),
        "excluded_incomplete_groups": int(excluded),
        "unknown_risk_threshold": float(unknown_risk_threshold),
        "accept_margin_threshold": float(accept_margin_threshold),
        "unknown_quantile": float(unknown_quantile),
        "counts": out_counts,
    }
