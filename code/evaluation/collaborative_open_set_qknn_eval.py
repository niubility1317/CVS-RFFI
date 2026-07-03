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
RISK_COMPONENT_KEYS = {
    "score": "score_risk",
    "radius": "radius_risk",
    "margin": "margin_risk",
    "mahalanobis": "mahalanobis_risk",
    "evt": "evt_risk",
    "oldness": "oldness_risk",
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


def _parse_risk_components(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [part.strip().lower() for part in text.replace(";", ",").split(",") if part.strip()]
    else:
        try:
            parts = [str(part).strip().lower() for part in value if str(part).strip()]
        except TypeError:
            parts = [str(value).strip().lower()]
    components: list[str] = []
    for part in parts:
        component = part.replace("-", "_")
        if component not in RISK_COMPONENT_KEYS:
            raise ValueError(f"unknown scorer risk component {part!r}")
        if component not in components:
            components.append(component)
    return components or None


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


def _validate_collaboration_policy(value: object) -> str:
    policy = _normalize_scope(value or "fixed_k")
    if policy not in {"fixed_k", "progressive_budget", "adaptive_gain"}:
        raise ValueError("collaboration_policy must be fixed_k, progressive_budget, or adaptive_gain")
    return policy


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
    fusion_policy: str,
    consensus_gap_threshold: float,
    consensus_score_threshold: float,
    scorer_component_vote_threshold: float,
    scorer_risk_components: Sequence[str] | str | None = None,
    can_request_more: bool = False,
    latency_budget_ms: float = 0.0,
) -> dict[str, Any]:
    active_components = _parse_risk_components(scorer_risk_components)
    label_scores: defaultdict[str, float] = defaultdict(float)
    weights = []
    risks = []
    margins = []
    scores = []
    score_risks = []
    radius_risks = []
    margin_risks = []
    mahalanobis_risks = []
    evt_risks = []
    oldness_risks = []
    component_votes = []
    predicted_labels = []
    for row in selected:
        weight = max(0.0, _float(row, "reliability", 1.0))
        label = _str(row, "predicted_label", "")
        if label and label != UNKNOWN_LABEL:
            score_value = max(0.0, _float(row, "known_score", 1.0))
            label_scores[label] += weight * score_value
            predicted_labels.append(label)
            scores.append(score_value)
        weights.append(weight)
        risks.append(_float(row, "unknown_risk", 0.0))
        margins.append(_float(row, "known_margin", 0.0))
        score_risk_value = _float(row, "score_risk", _float(row, "unknown_risk", 0.0))
        radius_risk_value = _float(row, "radius_risk", _float(row, "unknown_risk", 0.0))
        margin_risk_value = _float(row, "margin_risk", _float(row, "unknown_risk", 0.0))
        has_mahalanobis = "mahalanobis_risk" in row
        has_evt = "evt_risk" in row
        has_oldness = "oldness_risk" in row
        mahalanobis_risk_value = _float(row, "mahalanobis_risk", _float(row, "unknown_risk", 0.0))
        evt_risk_value = _float(row, "evt_risk", _float(row, "unknown_risk", 0.0))
        oldness_risk_value = _float(row, "oldness_risk", _float(row, "unknown_risk", 0.0))
        score_risks.append(score_risk_value)
        radius_risks.append(radius_risk_value)
        margin_risks.append(margin_risk_value)
        mahalanobis_risks.append(mahalanobis_risk_value)
        evt_risks.append(evt_risk_value)
        oldness_risks.append(oldness_risk_value)
        if active_components is None:
            component_values = [score_risk_value, radius_risk_value, margin_risk_value]
            if has_mahalanobis:
                component_values.append(mahalanobis_risk_value)
            if has_evt:
                component_values.append(evt_risk_value)
            if has_oldness:
                component_values.append(oldness_risk_value)
        else:
            row_values = {
                "score": score_risk_value,
                "radius": radius_risk_value,
                "margin": margin_risk_value,
                "mahalanobis": mahalanobis_risk_value,
                "evt": evt_risk_value,
                "oldness": oldness_risk_value,
            }
            component_values = [row_values[component] for component in active_components]
        component_votes.append(
            sum(value >= float(unknown_risk_threshold) for value in component_values)
            / float(max(len(component_values), 1))
        )

    unknown_risk = _weighted_quantile(risks, weights, unknown_quantile)
    score_risk = _weighted_quantile(score_risks, weights, unknown_quantile)
    radius_risk = _weighted_quantile(radius_risks, weights, unknown_quantile)
    margin_risk = _weighted_quantile(margin_risks, weights, unknown_quantile)
    mahalanobis_risk = _weighted_quantile(mahalanobis_risks, weights, unknown_quantile)
    evt_risk = _weighted_quantile(evt_risks, weights, unknown_quantile)
    oldness_risk = _weighted_quantile(oldness_risks, weights, unknown_quantile)
    if weights and sum(weights) > 0:
        risk_component_agreement = sum(v * max(0.0, w) for v, w in zip(component_votes, weights)) / max(
            sum(max(0.0, w) for w in weights),
            1e-12,
        )
    elif component_votes:
        risk_component_agreement = sum(component_votes) / len(component_votes)
    else:
        risk_component_agreement = 0.0
    mean_margin = 0.0
    if weights and sum(weights) > 0:
        mean_margin = sum(m * max(0.0, w) for m, w in zip(margins, weights)) / max(sum(weights), 1e-12)
    elif margins:
        mean_margin = sum(margins) / len(margins)
    mean_score = 0.0
    if weights and sum(weights) > 0 and scores:
        usable_weights = [max(0.0, w) for w, label in zip(weights, [_str(row, "predicted_label", "") for row in selected]) if label and label != UNKNOWN_LABEL]
        mean_score = sum(s * w for s, w in zip(scores, usable_weights)) / max(sum(usable_weights), 1e-12)
    elif scores:
        mean_score = sum(scores) / len(scores)

    if label_scores:
        label, score = max(label_scores.items(), key=lambda item: (item[1], item[0]))
    else:
        label, score = "", 0.0
    ranked_label_scores = sorted(label_scores.values(), reverse=True)
    top_label_score = ranked_label_scores[0] if ranked_label_scores else 0.0
    second_label_score = ranked_label_scores[1] if len(ranked_label_scores) > 1 else 0.0
    label_score_total = sum(max(0.0, value) for value in label_scores.values())
    score_gap_ratio = (top_label_score - second_label_score) / max(label_score_total, 1e-12)
    label_count = defaultdict(int)
    for item in predicted_labels:
        label_count[item] += 1
    ranked_counts = sorted(label_count.values(), reverse=True)
    top_count = ranked_counts[0] if ranked_counts else 0
    second_count = ranked_counts[1] if len(ranked_counts) > 1 else 0
    receiver_n = max(len(selected), 1)
    agreement = float(top_count) / float(receiver_n)
    vote_gap = float(top_count - second_count) / float(receiver_n)
    latency_ms = float(max((_float(row, "latency_ms", 0.0) for row in selected), default=0.0))
    within_request_budget = bool(
        can_request_more
        and float(latency_budget_ms) > 0.0
        and latency_ms < float(latency_budget_ms)
    )

    policy = _normalize_scope(fusion_policy)
    if policy == "consensus_veto":
        low_consensus = vote_gap <= float(consensus_gap_threshold)
        low_score = mean_score < float(consensus_score_threshold)
        low_margin = mean_margin < float(accept_margin_threshold)
        if unknown_risk >= float(unknown_risk_threshold) and (low_consensus or low_score or low_margin):
            decision = "unknown_reject"
            output_label = UNKNOWN_LABEL
        elif unknown_risk >= float(unknown_risk_threshold):
            decision = "defer"
            output_label = ""
        elif label:
            decision = "accept"
            output_label = label
        else:
            decision = "defer"
            output_label = ""
    elif policy == "scorer_cvs":
        strong_consensus = (
            vote_gap > float(consensus_gap_threshold) or score_gap_ratio > float(consensus_gap_threshold)
        ) and agreement >= 0.5
        strong_known = (
            bool(label)
            and strong_consensus
            and mean_margin >= float(accept_margin_threshold)
            and mean_score >= float(consensus_score_threshold)
        )
        high_risk = unknown_risk >= float(unknown_risk_threshold)
        multi_channel_risk = risk_component_agreement >= float(scorer_component_vote_threshold)
        if high_risk and multi_channel_risk and not strong_known:
            decision = "unknown_reject"
            output_label = UNKNOWN_LABEL
        elif high_risk:
            decision = "defer"
            output_label = ""
        elif strong_known:
            decision = "accept"
            output_label = label
        elif within_request_budget:
            decision = "request_more"
            output_label = ""
        else:
            decision = "defer"
            output_label = ""
    elif policy == "risk_margin":
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
    elif label:
        raise ValueError("fusion_policy must be risk_margin, consensus_veto, or scorer_cvs")
    else:
        raise ValueError("fusion_policy must be risk_margin, consensus_veto, or scorer_cvs")

    return {
        "decision": decision,
        "output_label": output_label,
        "unknown_risk": float(unknown_risk),
        "score_risk": float(score_risk),
        "radius_risk": float(radius_risk),
        "margin_risk": float(margin_risk),
        "mahalanobis_risk": float(mahalanobis_risk),
        "evt_risk": float(evt_risk),
        "oldness_risk": float(oldness_risk),
        "risk_component_agreement": float(risk_component_agreement),
        "known_margin": float(mean_margin),
        "mean_known_score": float(mean_score),
        "known_score": float(score),
        "agreement": float(agreement),
        "vote_gap": float(vote_gap),
        "score_gap_ratio": float(score_gap_ratio),
        "bytes": float(sum(_float(row, "bytes", 0.0) for row in selected)),
        "latency_ms": latency_ms,
        "can_request_more": bool(can_request_more),
        "participating_receivers_used": int(len(selected)),
    }


def _fuse_progressive_event(
    ordered: Sequence[Mapping[str, Any]],
    *,
    max_receivers: int,
    unknown_risk_threshold: float,
    accept_margin_threshold: float,
    unknown_quantile: float,
    fusion_policy: str,
    consensus_gap_threshold: float,
    consensus_score_threshold: float,
    scorer_component_vote_threshold: float,
    scorer_risk_components: Sequence[str] | str | None = None,
    latency_budget_ms: float = 0.0,
) -> dict[str, Any]:
    max_receivers = max(1, int(max_receivers))
    last: dict[str, Any] | None = None
    for used in range(1, min(max_receivers, len(ordered)) + 1):
        fused = _fuse_event(
            ordered[:used],
            unknown_risk_threshold=unknown_risk_threshold,
            accept_margin_threshold=accept_margin_threshold,
            unknown_quantile=unknown_quantile,
            fusion_policy=fusion_policy,
            consensus_gap_threshold=consensus_gap_threshold,
            consensus_score_threshold=consensus_score_threshold,
            scorer_component_vote_threshold=scorer_component_vote_threshold,
            scorer_risk_components=scorer_risk_components,
            can_request_more=used < min(max_receivers, len(ordered)),
            latency_budget_ms=latency_budget_ms,
        )
        fused["participating_receiver_budget"] = int(max_receivers)
        fused["participating_receivers_used"] = int(used)
        fused["progressive_stop_reason"] = str(fused["decision"])
        last = fused
        if fused["decision"] != "request_more":
            return fused
    if last is None:
        raise ValueError("progressive collaboration requires at least one receiver observation")
    last["progressive_stop_reason"] = "budget_exhausted"
    return last


def _adaptive_receiver_gain(
    row: Mapping[str, Any],
    *,
    current_label: str,
    unknown_risk_threshold: float,
    accept_margin_threshold: float,
    consensus_score_threshold: float,
    latency_weight: float,
    bytes_weight: float,
    disagreement_weight: float,
) -> tuple[float, dict[str, float]]:
    reliability = max(0.0, _float(row, "reliability", 1.0))
    known_score = max(0.0, _float(row, "known_score", 0.0))
    known_margin = max(0.0, _float(row, "known_margin", 0.0))
    unknown_risk = max(0.0, min(1.0, _float(row, "unknown_risk", 0.0)))
    score_floor = max(float(consensus_score_threshold), 1e-6)
    margin_floor = max(float(accept_margin_threshold), 1e-6)
    ambiguity_gain = max(0.0, 1.0 - min(known_score / score_floor, 1.0))
    margin_gain = max(0.0, 1.0 - min(known_margin / margin_floor, 1.0))
    threshold = max(1e-6, min(1.0, float(unknown_risk_threshold)))
    unknown_boundary_gain = max(0.0, 1.0 - min(abs(unknown_risk - threshold) / threshold, 1.0))
    candidate_label = _str(row, "predicted_label", "")
    disagreement_gain = 1.0 if current_label and candidate_label and candidate_label != current_label else 0.0
    latency_cost = max(0.0, _float(row, "latency_ms", 0.0)) * max(0.0, float(latency_weight))
    bytes_cost = max(0.0, _float(row, "bytes", 0.0)) * max(0.0, float(bytes_weight))
    cost = 1.0 + latency_cost + bytes_cost
    raw_gain = reliability * (
        ambiguity_gain
        + margin_gain
        + unknown_boundary_gain
        + max(0.0, float(disagreement_weight)) * disagreement_gain
    )
    return raw_gain / cost, {
        "reliability": float(reliability),
        "ambiguity": float(ambiguity_gain),
        "margin": float(margin_gain),
        "unknown_boundary": float(unknown_boundary_gain),
        "disagreement": float(disagreement_gain),
        "cost": float(cost),
        "gain": float(raw_gain / cost),
    }


def _adaptive_should_request_more(
    fused: Mapping[str, Any],
    *,
    unknown_risk_threshold: float,
    accept_margin_threshold: float,
    consensus_gap_threshold: float,
    consensus_score_threshold: float,
    adaptive_gain_min_risk: float,
) -> bool:
    if str(fused.get("decision")) == "request_more":
        return True
    mean_score = float(fused.get("mean_known_score", 0.0))
    mean_margin = float(fused.get("known_margin", 0.0))
    vote_gap = float(fused.get("vote_gap", 0.0))
    unknown_risk = float(fused.get("unknown_risk", 0.0))
    risks = [
        float(fused.get("score_risk", 0.0)),
        float(fused.get("radius_risk", 0.0)),
        float(fused.get("margin_risk", 0.0)),
        float(fused.get("mahalanobis_risk", 0.0)),
        float(fused.get("evt_risk", 0.0)),
        float(fused.get("oldness_risk", 0.0)),
    ]
    low_score = float(consensus_score_threshold) > 0.0 and mean_score < float(consensus_score_threshold)
    low_margin = mean_margin < float(accept_margin_threshold)
    low_consensus = vote_gap <= float(consensus_gap_threshold)
    high_but_not_terminal_risk = max(risks + [unknown_risk]) >= float(adaptive_gain_min_risk)
    terminal_unknown = unknown_risk >= float(unknown_risk_threshold) and float(
        fused.get("risk_component_agreement", 0.0)
    ) >= 0.5
    if terminal_unknown:
        return False
    return bool(low_score or low_margin or low_consensus or high_but_not_terminal_risk)


def _fuse_adaptive_gain_event(
    ordered: Sequence[Mapping[str, Any]],
    *,
    max_receivers: int,
    unknown_risk_threshold: float,
    accept_margin_threshold: float,
    unknown_quantile: float,
    fusion_policy: str,
    consensus_gap_threshold: float,
    consensus_score_threshold: float,
    scorer_component_vote_threshold: float,
    scorer_risk_components: Sequence[str] | str | None = None,
    latency_budget_ms: float = 0.0,
    adaptive_gain_min_risk: float = 0.80,
    adaptive_gain_latency_weight: float = 0.0,
    adaptive_gain_bytes_weight: float = 0.0,
    adaptive_gain_disagreement_weight: float = 0.5,
) -> dict[str, Any]:
    max_receivers = max(1, int(max_receivers))
    if not ordered:
        raise ValueError("adaptive_gain collaboration requires at least one receiver observation")
    budget = min(max_receivers, len(ordered))
    selected = [ordered[0]]
    remaining = list(ordered[1:])
    gain_trace: list[str] = []
    last: dict[str, Any] | None = None
    while True:
        fused = _fuse_event(
            selected,
            unknown_risk_threshold=unknown_risk_threshold,
            accept_margin_threshold=accept_margin_threshold,
            unknown_quantile=unknown_quantile,
            fusion_policy=fusion_policy,
            consensus_gap_threshold=consensus_gap_threshold,
            consensus_score_threshold=consensus_score_threshold,
            scorer_component_vote_threshold=scorer_component_vote_threshold,
            scorer_risk_components=scorer_risk_components,
            can_request_more=len(selected) < budget,
            latency_budget_ms=latency_budget_ms,
        )
        fused["participating_receiver_budget"] = int(max_receivers)
        fused["participating_receivers_used"] = int(len(selected))
        fused["selected_receiver_order"] = ",".join(_str(row, "receiver_id") for row in selected)
        fused["adaptive_gain_trace"] = ";".join(gain_trace)
        last = fused
        if len(selected) >= budget or not remaining:
            fused["adaptive_stop_reason"] = (
                f"budget_exhausted_{fused['decision']}" if len(selected) >= budget else str(fused["decision"])
            )
            return fused
        if not _adaptive_should_request_more(
            fused,
            unknown_risk_threshold=unknown_risk_threshold,
            accept_margin_threshold=accept_margin_threshold,
            consensus_gap_threshold=consensus_gap_threshold,
            consensus_score_threshold=consensus_score_threshold,
            adaptive_gain_min_risk=adaptive_gain_min_risk,
        ):
            fused["adaptive_stop_reason"] = str(fused["decision"])
            return fused
        current_label = str(fused.get("output_label") or "")
        scored = []
        for row in remaining:
            gain, parts = _adaptive_receiver_gain(
                row,
                current_label=current_label,
                unknown_risk_threshold=unknown_risk_threshold,
                accept_margin_threshold=accept_margin_threshold,
                consensus_score_threshold=consensus_score_threshold,
                latency_weight=adaptive_gain_latency_weight,
                bytes_weight=adaptive_gain_bytes_weight,
                disagreement_weight=adaptive_gain_disagreement_weight,
            )
            scored.append((gain, row, parts))
        gain, row, parts = max(scored, key=lambda item: (item[0], _float(item[1], "reliability", 1.0), _str(item[1], "receiver_id")))
        remaining.remove(row)
        selected.append(row)
        gain_trace.append(
            f"{_str(row, 'receiver_id')}:{gain:.6f}:"
            f"a={parts['ambiguity']:.3f},m={parts['margin']:.3f},u={parts['unknown_boundary']:.3f},d={parts['disagreement']:.3f}"
        )
    if last is None:
        raise ValueError("adaptive_gain collaboration failed to fuse any receiver")
    return last


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
    unknown_defer = 0
    unknown_request_more = 0
    defer_total = 0
    request_more_total = 0
    adaptive_stop_reasons: defaultdict[str, int] = defaultdict(int)

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
        requested_more = decision == "request_more"
        if decision == "unknown_reject":
            predicted_bucket = "unknown_reject"
        elif requested_more:
            predicted_bucket = "request_more"
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
        if requested_more:
            request_more_total += 1
        stop_reason = str(item.get("adaptive_stop_reason") or item.get("progressive_stop_reason") or "")
        if stop_reason:
            adaptive_stop_reasons[stop_reason] += 1
        role_totals[role] += 1
        if role in {"old", "seen_new"}:
            role_accepted[role] += int(accepted)
            role_correct[role] += int(accepted and output == truth)
            per_class_total[role][str(truth)] += 1
            per_class_correct[role][str(truth)] += int(accepted and output == truth)
        elif role == "unknown":
            unknown_rejected += int(decision == "unknown_reject")
            unknown_false_accept += int(accepted)
            unknown_defer += int(deferred)
            unknown_request_more += int(requested_more)

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
    participating_values = [float(item.get("participating_receivers_used", k)) for item in event_results]
    known_total = role_totals["old"] + role_totals["seen_new"]
    known_accepted = role_accepted["old"] + role_accepted["seen_new"]
    known_correct = role_correct["old"] + role_correct["seen_new"]
    total_events = len(event_results)

    return {
        "participating_receivers": int(k),
        "participating_receivers_avg": sum(participating_values) / max(len(participating_values), 1),
        "participating_receivers_p95": _percentile(participating_values, 0.95),
        "participating_receivers_max": int(max(participating_values, default=float(k))),
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
        "unknown_defer": int(unknown_defer),
        "unknown_defer_rate": _safe_rate(unknown_defer, role_totals["unknown"]),
        "unknown_request_more": int(unknown_request_more),
        "unknown_request_more_rate": _safe_rate(unknown_request_more, role_totals["unknown"]),
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy": _safe_rate(known_correct, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct, known_accepted),
        "defer_rate": _safe_rate(defer_total, total_events),
        "request_more_rate": _safe_rate(request_more_total, total_events),
        "unresolved_rate": _safe_rate(defer_total + request_more_total, total_events),
        "open_set_confusion": dict(sorted(confusion.items())),
        "bytes_per_event": sum(bytes_values) / max(len(bytes_values), 1),
        "total_bytes": float(sum(bytes_values)),
        "latency_ms_p50": _percentile(latency_values, 0.50),
        "latency_ms_p95": _percentile(latency_values, 0.95),
        "collaboration_stop_reasons": dict(sorted(adaptive_stop_reasons.items())),
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
    fusion_policy: str = "risk_margin",
    consensus_gap_threshold: float = 0.0,
    consensus_score_threshold: float = 0.0,
    scorer_component_vote_threshold: float = 0.5,
    scorer_risk_components: Sequence[str] | str | None = None,
    collaboration_policy: str = "fixed_k",
    latency_budget_ms: float = 0.0,
    adaptive_gain_min_risk: float = 0.80,
    adaptive_gain_latency_weight: float = 0.0,
    adaptive_gain_bytes_weight: float = 0.0,
    adaptive_gain_disagreement_weight: float = 0.5,
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
    active_risk_components = _parse_risk_components(scorer_risk_components)
    collaboration_policy = _validate_collaboration_policy(collaboration_policy)
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
            if collaboration_policy == "progressive_budget":
                fused = _fuse_progressive_event(
                    selected,
                    max_receivers=int(k),
                    unknown_risk_threshold=unknown_risk_threshold,
                    accept_margin_threshold=accept_margin_threshold,
                    unknown_quantile=unknown_quantile,
                    fusion_policy=fusion_policy,
                    consensus_gap_threshold=consensus_gap_threshold,
                    consensus_score_threshold=consensus_score_threshold,
                    scorer_component_vote_threshold=scorer_component_vote_threshold,
                    scorer_risk_components=active_risk_components,
                    latency_budget_ms=latency_budget_ms,
                )
            elif collaboration_policy == "adaptive_gain":
                adaptive_ordered = _select_receivers(
                    group,
                    len(group),
                    receiver_selection_policy=receiver_selection_policy,
                )
                fused = _fuse_adaptive_gain_event(
                    adaptive_ordered,
                    max_receivers=int(k),
                    unknown_risk_threshold=unknown_risk_threshold,
                    accept_margin_threshold=accept_margin_threshold,
                    unknown_quantile=unknown_quantile,
                    fusion_policy=fusion_policy,
                    consensus_gap_threshold=consensus_gap_threshold,
                    consensus_score_threshold=consensus_score_threshold,
                    scorer_component_vote_threshold=scorer_component_vote_threshold,
                    scorer_risk_components=active_risk_components,
                    latency_budget_ms=latency_budget_ms,
                    adaptive_gain_min_risk=adaptive_gain_min_risk,
                    adaptive_gain_latency_weight=adaptive_gain_latency_weight,
                    adaptive_gain_bytes_weight=adaptive_gain_bytes_weight,
                    adaptive_gain_disagreement_weight=adaptive_gain_disagreement_weight,
                )
            else:
                fused = _fuse_event(
                    selected,
                    unknown_risk_threshold=unknown_risk_threshold,
                    accept_margin_threshold=accept_margin_threshold,
                    unknown_quantile=unknown_quantile,
                    fusion_policy=fusion_policy,
                    consensus_gap_threshold=consensus_gap_threshold,
                    consensus_score_threshold=consensus_score_threshold,
                    scorer_component_vote_threshold=scorer_component_vote_threshold,
                    scorer_risk_components=active_risk_components,
                    can_request_more=len({_str(row, "receiver_id") for row in group}) > int(k),
                    latency_budget_ms=latency_budget_ms,
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
        "collaboration_policy": collaboration_policy,
        "matched_max_requested_group_count": int(len(matched_max)),
        "receiver_selection_policy": _normalize_scope(receiver_selection_policy),
        "threshold_selection_label_scope": _normalize_scope(threshold_selection_label_scope),
        "unknown_query_eval_only": bool(unknown_query_eval_only),
        "stage2_protocol": protocol_report,
        "evidence_scope": "offline_evidence_metrics_only",
        "unknown_risk_threshold": float(unknown_risk_threshold),
        "accept_margin_threshold": float(accept_margin_threshold),
        "unknown_quantile": float(unknown_quantile),
        "fusion_policy": _normalize_scope(fusion_policy),
        "consensus_gap_threshold": float(consensus_gap_threshold),
        "consensus_score_threshold": float(consensus_score_threshold),
        "scorer_component_vote_threshold": float(scorer_component_vote_threshold),
        "active_risk_components": active_risk_components,
        "latency_budget_ms": float(latency_budget_ms),
        "adaptive_gain_min_risk": float(adaptive_gain_min_risk),
        "adaptive_gain_latency_weight": float(adaptive_gain_latency_weight),
        "adaptive_gain_bytes_weight": float(adaptive_gain_bytes_weight),
        "adaptive_gain_disagreement_weight": float(adaptive_gain_disagreement_weight),
        "counts": out_counts,
    }
