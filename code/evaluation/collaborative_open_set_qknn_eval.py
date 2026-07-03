from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from typing import Any, Mapping, Sequence


UNKNOWN_LABEL = "__unknown__"
ROLE_ALIASES = {
    "old": "old",
    "target_old": "old",
    "seen_new": "seen_new",
    "seennew": "seen_new",
    "target_new": "seen_new",
    "new": "seen_new",
    "unknown": "unknown",
    "target_unknown": "unknown",
    "unk": "unknown",
}
CALIBRATION_ROLES = {"threshold_fit", "calibration", "fit", "train"}
SAFE_THRESHOLD_SCOPES = {
    "fixed_prior",
    "source_only",
    "support_only",
    "known_support",
    "support_known_only",
    "old_new_support",
    "deployment_prior",
}
PRIOR_RELIABILITY_SOURCES = {
    "receiver_prior",
    "link_budget_prior",
    "pre_query_prior",
    "deployment_prior",
}


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
    if text not in ROLE_ALIASES:
        raise ValueError(f"unknown evidence role {value!r}; expected one of {sorted(ROLE_ALIASES)}")
    return ROLE_ALIASES[text]


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


def _normalize_scope(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _validate_threshold_scope(scope: str, *, unknown_query_eval_only: bool) -> None:
    scope = _normalize_scope(scope)
    if not scope:
        raise ValueError("threshold_selection_label_scope must be recorded")
    if scope not in SAFE_THRESHOLD_SCOPES:
        raise ValueError(
            "threshold_selection_label_scope must not use unknown query labels; "
            f"got {scope!r}"
        )
    if not bool(unknown_query_eval_only):
        raise ValueError("unknown_query_eval_only must be true for deployment-style open-set evaluation")


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_selection_label_scope: str,
    unknown_query_eval_only: bool,
    receiver_selection_policy: str,
) -> None:
    _validate_threshold_scope(
        threshold_selection_label_scope,
        unknown_query_eval_only=unknown_query_eval_only,
    )
    policy = _normalize_scope(receiver_selection_policy)
    if policy not in {"fixed_receiver_order", "reliability_prior"}:
        raise ValueError("receiver_selection_policy must be fixed_receiver_order or reliability_prior")
    for row in rows:
        role = _role(row.get("role"))
        if role in {"old", "seen_new"} and not _str(row, "true_label").strip():
            raise ValueError("known evidence rows must include true_label")
        row_scope = _normalize_scope(row.get("threshold_selection_label_scope", threshold_selection_label_scope))
        _validate_threshold_scope(row_scope, unknown_query_eval_only=unknown_query_eval_only)
        calibration_role = _normalize_scope(row.get("calibration_role", "query"))
        if role == "unknown" and calibration_role in CALIBRATION_ROLES:
            raise ValueError("unknown query rows cannot be used for threshold fitting or calibration")
        if policy == "reliability_prior":
            source = _normalize_scope(row.get("reliability_source", ""))
            if source not in PRIOR_RELIABILITY_SOURCES:
                raise ValueError(
                    "reliability_prior receiver selection requires query-independent reliability_source"
                )


def _select_receivers(
    rows: Sequence[Mapping[str, Any]],
    k: int,
    *,
    receiver_selection_policy: str,
) -> list[Mapping[str, Any]]:
    policy = _normalize_scope(receiver_selection_policy)
    if policy == "reliability_prior":
        ordered = sorted(
            rows,
            key=lambda row: (
                -_float(row, "reliability", 1.0),
                _float(row, "latency_ms", 0.0),
                _str(row, "receiver_id"),
            ),
        )
    else:
        ordered = sorted(rows, key=lambda row: _str(row, "receiver_id"))
    return ordered[: int(k)]


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

    if unknown_risk >= float(unknown_risk_threshold):
        if mean_margin < float(accept_margin_threshold):
            decision = "unknown_reject"
            output_label = UNKNOWN_LABEL
        else:
            decision = "defer"
            output_label = ""
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


def _finalize_metrics(
    event_results: Sequence[dict[str, Any]],
    *,
    k: int,
    excluded_incomplete: int,
    expected_old_labels: set[str] | None = None,
    expected_seen_new_labels: set[str] | None = None,
) -> dict[str, Any]:
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

    old_labels = set(expected_old_labels or set())
    seen_new_labels = set(expected_seen_new_labels or set())
    old_labels.update(str(item["true_label"]) for item in event_results if item["role"] == "old")
    seen_new_labels.update(str(item["true_label"]) for item in event_results if item["role"] == "seen_new")
    confusion: defaultdict[str, int] = defaultdict(int)

    for item in event_results:
        role = item["role"]
        truth = item["true_label"]
        output = item["output_label"]
        decision = item["decision"]
        accepted = decision == "accept"
        deferred = decision == "defer"
        if decision == "unknown_reject":
            predicted_bucket = "unknown_reject"
        elif deferred:
            predicted_bucket = "defer"
        elif output in old_labels:
            predicted_bucket = "old"
        elif output in seen_new_labels:
            predicted_bucket = "seen_new"
        else:
            predicted_bucket = "other_accept"
        confusion[f"{role}->{predicted_bucket}"] += 1
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

    missing_old = sorted(label for label in old_labels if per_class_total["old"].get(label, 0) <= 0)
    missing_seen_new = sorted(label for label in seen_new_labels if per_class_total["seen_new"].get(label, 0) <= 0)
    old_class_rates = {
        label: _safe_rate(per_class_correct["old"][label], per_class_total["old"].get(label, 0))
        for label in sorted(old_labels)
        if label
    }
    new_class_rates = {
        label: _safe_rate(per_class_correct["seen_new"][label], per_class_total["seen_new"].get(label, 0))
        for label in sorted(seen_new_labels)
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
        "missing_old_classes": missing_old,
        "min_old_class_acc": min(old_class_rates.values()) if old_class_rates else 0.0,
        "seen_new_total": int(role_totals["seen_new"]),
        "seen_new_correct": int(role_correct["seen_new"]),
        "seen_new_acc": _safe_rate(role_correct["seen_new"], role_totals["seen_new"]),
        "per_seen_new_class_acc": new_class_rates,
        "missing_seen_new_classes": missing_seen_new,
        "min_seen_new_class_acc": min(new_class_rates.values()) if new_class_rates else 0.0,
        "unknown_total": int(role_totals["unknown"]),
        "unknown_rejected": int(unknown_rejected),
        "unknown_reject_rate": _safe_rate(unknown_rejected, role_totals["unknown"]),
        "unknown_FAR": _safe_rate(unknown_false_accept, role_totals["unknown"]),
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy": _safe_rate(known_correct, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct, known_accepted),
        "defer_rate": _safe_rate(defer_total, total_events),
        "open_set_confusion": dict(sorted(confusion.items())),
        "bytes_per_event": sum(bytes_values) / max(len(bytes_values), 1),
        "total_bytes": float(sum(bytes_values)),
        "latency_ms_p50": _percentile(latency_values, 0.50),
        "latency_ms_p95": _percentile(latency_values, 0.95),
    }


def _items(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.replace(";", ",").split(",") if part.strip()}
    try:
        return {str(item).strip() for item in value if str(item).strip()}
    except TypeError:
        text = str(value).strip()
        return {text} if text else set()


def _validate_protocol_metadata(metadata: Mapping[str, Any] | None, *, strict: bool) -> dict[str, Any]:
    if not metadata:
        if strict:
            raise ValueError("protocol_metadata is required when strict_protocol_metadata=True")
        return {"validated": False, "reason": "protocol_metadata_missing"}
    source_receivers = _items(metadata.get("source_receiver_ids"))
    target_receivers = _items(metadata.get("target_receiver_ids"))
    old_tx = _items(metadata.get("old_tx_ids"))
    seen_new_tx = _items(metadata.get("seen_new_tx_ids", metadata.get("target_new_tx_ids")))
    unknown_tx = _items(metadata.get("unknown_tx_ids"))
    missing = [
        name
        for name, values in {
            "source_receiver_ids": source_receivers,
            "target_receiver_ids": target_receivers,
            "old_tx_ids": old_tx,
            "seen_new_tx_ids": seen_new_tx,
            "unknown_tx_ids": unknown_tx,
        }.items()
        if not values
    ]
    if missing:
        if strict:
            raise ValueError(f"protocol_metadata missing required fields: {', '.join(missing)}")
        return {"validated": False, "reason": "protocol_metadata_incomplete", "missing": missing}
    if source_receivers & target_receivers:
        raise ValueError("source_receiver_ids and target_receiver_ids must be disjoint")
    if old_tx & seen_new_tx or old_tx & unknown_tx or seen_new_tx & unknown_tx:
        raise ValueError("old_tx_ids, seen_new_tx_ids, and unknown_tx_ids must be mutually disjoint")
    channel_view = str(metadata.get("target_channel_view", "") or "").strip()
    if not channel_view:
        raise ValueError("protocol_metadata must include target_channel_view")
    return {
        "validated": True,
        "source_receiver_count": len(source_receivers),
        "target_receiver_count": len(target_receivers),
        "old_tx_count": len(old_tx),
        "seen_new_tx_count": len(seen_new_tx),
        "unknown_tx_count": len(unknown_tx),
        "target_channel_view": channel_view,
    }


def _validate_event_groups(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for event_id, group in groups.items():
        receivers: set[str] = set()
        roles: set[str] = set()
        labels: set[str] = set()
        for row in group:
            receiver = _str(row, "receiver_id")
            if receiver in receivers:
                raise ValueError(f"duplicate receiver_id {receiver!r} in event_id {event_id!r}")
            receivers.add(receiver)
            role = _role(row.get("role"))
            roles.add(role)
            labels.add(_str(row, "true_label", UNKNOWN_LABEL if role == "unknown" else "").strip())
        if len(roles) != 1:
            raise ValueError(f"inconsistent role values in event_id {event_id!r}")
        if len(labels) != 1:
            raise ValueError(f"inconsistent true_label values in event_id {event_id!r}")


def evaluate_collaborative_open_set_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    collab_counts: str | Sequence[int] | None = None,
    unknown_risk_threshold: float = 0.80,
    accept_margin_threshold: float = 0.10,
    unknown_quantile: float = 0.75,
    threshold_selection_label_scope: str = "support_known_only",
    unknown_query_eval_only: bool = True,
    receiver_selection_policy: str = "fixed_receiver_order",
    protocol_metadata: Mapping[str, Any] | None = None,
    strict_protocol_metadata: bool = False,
) -> dict[str, Any]:
    """Evaluate offline collaborative open-set qknn-style evidence.

    Each row is one receiver observation for one event. The function does not
    fit thresholds and does not use unknown query rows for calibration.
    """

    rows = list(rows)
    _validate_rows(
        rows,
        threshold_selection_label_scope=threshold_selection_label_scope,
        unknown_query_eval_only=unknown_query_eval_only,
        receiver_selection_policy=receiver_selection_policy,
    )
    protocol_report = _validate_protocol_metadata(protocol_metadata, strict=strict_protocol_metadata)
    expected_old_labels = _items(protocol_metadata.get("old_tx_ids") if protocol_metadata else None)
    expected_seen_new_labels = _items(
        protocol_metadata.get("seen_new_tx_ids", protocol_metadata.get("target_new_tx_ids"))
        if protocol_metadata
        else None
    )
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
    _validate_event_groups(groups)

    receiver_count = len(receivers)
    counts = parse_collab_counts(collab_counts, receiver_count=receiver_count)
    max_requested = max(counts)
    matched_max = [
        (event_id, group)
        for event_id, group in groups.items()
        if len({_str(row, "receiver_id") for row in group}) >= max_requested
    ]

    out_counts: dict[str, Any] = {}
    for k in counts:
        eligible = [
            (event_id, group)
            for event_id, group in groups.items()
            if len({_str(row, "receiver_id") for row in group}) >= int(k)
        ]
        excluded = len(groups) - len(eligible)
        if not eligible:
            raise ValueError(f"no evidence groups contain {k} receiver observations")
        event_results: list[dict[str, Any]] = []
        for _, group in eligible:
            selected = _select_receivers(
                group,
                int(k),
                receiver_selection_policy=receiver_selection_policy,
            )
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
        out_counts[str(k)] = _finalize_metrics(
            event_results,
            k=int(k),
            excluded_incomplete=excluded,
            expected_old_labels=expected_old_labels,
            expected_seen_new_labels=expected_seen_new_labels,
        )

    return {
        "enabled": True,
        "protocol": "collaborative_open_set_qknn_evidence",
        "receiver_count": int(receiver_count),
        "observed_receiver_ids": sorted(receivers),
        "group_count": int(len(groups)),
        "eligible_group_count": int(len(matched_max)),
        "excluded_incomplete_groups": int(len(groups) - len(matched_max)),
        "denominator_policy": "per_k_available_receivers",
        "matched_max_requested_group_count": int(len(matched_max)),
        "receiver_selection_policy": _normalize_scope(receiver_selection_policy),
        "threshold_selection_label_scope": _normalize_scope(threshold_selection_label_scope),
        "unknown_query_eval_only": bool(unknown_query_eval_only),
        "stage2_protocol": protocol_report,
        "evidence_scope": "offline_evidence_metrics_only",
        "unknown_risk_threshold": float(unknown_risk_threshold),
        "accept_margin_threshold": float(accept_margin_threshold),
        "unknown_quantile": float(unknown_quantile),
        "counts": out_counts,
    }
