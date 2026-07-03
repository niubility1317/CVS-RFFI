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
    "support_virtual_unknown",
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
    "virtual_unknown": "virtual_unknown_risk",
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


def _has_finite_float(row: Mapping[str, Any], key: str) -> bool:
    if key not in row:
        return False
    try:
        return math.isfinite(float(row.get(key)))
    except (TypeError, ValueError):
        return False


def _resource_budget_reason(
    selected: Sequence[Mapping[str, Any]],
    *,
    max_event_bytes: float = 0.0,
    max_event_latency_ms: float = 0.0,
) -> str:
    reasons = []
    total_bytes = sum(_float(row, "bytes", 0.0) for row in selected)
    latency_ms = max((_float(row, "latency_ms", 0.0) for row in selected), default=0.0)
    if float(max_event_bytes) > 0.0 and total_bytes > float(max_event_bytes):
        reasons.append(f"bytes:{total_bytes:.6g}>{float(max_event_bytes):.6g}")
    if float(max_event_latency_ms) > 0.0 and latency_ms > float(max_event_latency_ms):
        reasons.append(f"latency_ms:{latency_ms:.6g}>{float(max_event_latency_ms):.6g}")
    return ",".join(reasons)


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
    label_fusion_policy: str = "score_sum",
    can_request_more: bool = False,
    latency_budget_ms: float = 0.0,
    max_event_bytes: float = 0.0,
    max_event_latency_ms: float = 0.0,
    old_labels: set[str] | None = None,
    seen_new_rescue_labels: set[str] | None = None,
    seen_new_rescue_enabled: bool = False,
    seen_new_rescue_risk_scale: float = 1.0,
    seen_new_rescue_min_score: float = 0.0,
    seen_new_rescue_min_margin: float = 0.0,
    seen_new_rescue_min_agreement: float = 0.5,
    conformal_rescue_enabled: bool = False,
    conformal_rescue_min_pvalue: float = 0.05,
    conformal_rescue_risk_scale: float = 0.5,
    conformal_rescue_min_agreement: float = 0.5,
    class_set_gate_enabled: bool = False,
    old_gate_min_receivers: int = 1,
    old_gate_max_effective_unknown_risk: float = 1.0,
    old_gate_max_component_agreement: float = 1.0,
    old_gate_min_support_density: float = 0.0,
    old_gate_max_radius_z: float = 1.0e12,
    seen_new_gate_min_receivers: int = 1,
    seen_new_gate_max_effective_unknown_risk: float = 1.0,
    seen_new_gate_max_component_agreement: float = 1.0,
    seen_new_gate_min_support_density: float = 0.0,
    seen_new_gate_max_radius_z: float = 1.0e12,
) -> dict[str, Any]:
    active_components = _parse_risk_components(scorer_risk_components)
    policy = _normalize_scope(fusion_policy)
    label_fusion_policy = _normalize_scope(label_fusion_policy)
    if label_fusion_policy not in {"score_sum", "vote_sum", "vote_margin", "max_score"}:
        raise ValueError("label_fusion_policy must be score_sum, vote_sum, vote_margin, or max_score")
    label_scores: defaultdict[str, float] = defaultdict(float)
    label_raw_scores: defaultdict[str, float] = defaultdict(float)
    label_weight_totals: defaultdict[str, float] = defaultdict(float)
    label_margins: defaultdict[str, list[float]] = defaultdict(list)
    label_max_scores: defaultdict[str, float] = defaultdict(float)
    label_support_density_values: defaultdict[str, list[float]] = defaultdict(list)
    label_radius_z_values: defaultdict[str, list[float]] = defaultdict(list)
    label_conformal_pvalues: defaultdict[str, list[float]] = defaultdict(list)
    label_conformal_support_counts: defaultdict[str, list[float]] = defaultdict(list)
    label_support_density_missing_values: defaultdict[str, list[bool]] = defaultdict(list)
    label_radius_z_missing_values: defaultdict[str, list[bool]] = defaultdict(list)
    label_unknown_risk_values: defaultdict[str, list[float]] = defaultdict(list)
    label_risk_component_votes: defaultdict[str, list[float]] = defaultdict(list)
    label_candidate_receiver_counts: defaultdict[str, int] = defaultdict(int)
    label_top1_receiver_counts: defaultdict[str, int] = defaultdict(int)
    label_min_evidence_rank: defaultdict[str, int] = defaultdict(lambda: 10**9)
    known_old_labels = set(str(item) for item in (old_labels or set()) if str(item))
    known_seen_new_labels = set(str(item) for item in (seen_new_rescue_labels or set()) if str(item))
    allowed_cp_set_labels = known_old_labels | known_seen_new_labels
    filtered_candidate_count = 0
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
    virtual_unknown_risks = []
    component_votes = []
    predicted_labels = []
    support_densities = []
    radius_z_values = []
    class_conformal_pvalues = []
    class_conformal_support_counts = []
    support_density_missing = []
    radius_z_missing = []

    def _add_label_candidate(
        *,
        row_seen_labels: set[str],
        label: str,
        weight: float,
        score_value: float,
        margin_value: float,
        support_density_value: float,
        radius_z_value: float,
        support_density_is_missing: bool,
        radius_z_is_missing: bool,
        conformal_pvalue: float,
        conformal_support_count: float,
        conformal_weighted: bool,
        source_rank: int,
        unknown_risk_value: float,
        score_risk_value: float,
        radius_risk_value: float,
        margin_risk_value: float,
        mahalanobis_risk_value: float,
        evt_risk_value: float,
        oldness_risk_value: float,
        virtual_unknown_risk_value: float,
    ) -> None:
        nonlocal filtered_candidate_count
        if not label or label == UNKNOWN_LABEL or label in row_seen_labels:
            return
        if policy == "cp_set_cvs" and allowed_cp_set_labels and label not in allowed_cp_set_labels:
            filtered_candidate_count += 1
            return
        row_seen_labels.add(label)
        pvalue = max(0.0, min(1.0, float(conformal_pvalue)))
        candidate_score = max(0.0, float(score_value))
        support_count = max(0.0, float(conformal_support_count))
        if policy == "cp_set_cvs" and int(source_rank) > 1 and (candidate_score <= 0.0 or support_count < 1.0):
            filtered_candidate_count += 1
            return
        if conformal_weighted:
            candidate_score *= 0.5 + 0.5 * pvalue
        label_scores[label] += weight * candidate_score
        label_raw_scores[label] += weight * max(0.0, float(score_value))
        label_weight_totals[label] += weight
        label_margins[label].append(max(0.0, float(margin_value)))
        label_max_scores[label] = max(label_max_scores[label], max(0.0, float(score_value)))
        label_support_density_values[label].append(support_density_value)
        label_radius_z_values[label].append(radius_z_value)
        label_conformal_pvalues[label].append(pvalue)
        label_conformal_support_counts[label].append(support_count)
        label_support_density_missing_values[label].append(bool(support_density_is_missing))
        label_radius_z_missing_values[label].append(bool(radius_z_is_missing))
        label_unknown_risk_values[label].append(max(0.0, min(1.0, float(unknown_risk_value))))
        row_component_values = {
            "score": float(score_risk_value),
            "radius": float(radius_risk_value),
            "margin": float(margin_risk_value),
            "mahalanobis": float(mahalanobis_risk_value),
            "evt": float(evt_risk_value),
            "oldness": float(oldness_risk_value),
            "virtual_unknown": float(virtual_unknown_risk_value),
        }
        if active_components is None:
            candidate_components = [row_component_values["score"], row_component_values["radius"], row_component_values["margin"]]
            if math.isfinite(row_component_values["mahalanobis"]):
                candidate_components.append(row_component_values["mahalanobis"])
            if math.isfinite(row_component_values["evt"]):
                candidate_components.append(row_component_values["evt"])
            if math.isfinite(row_component_values["oldness"]):
                candidate_components.append(row_component_values["oldness"])
            if math.isfinite(row_component_values["virtual_unknown"]):
                candidate_components.append(row_component_values["virtual_unknown"])
        else:
            candidate_components = [row_component_values[component] for component in active_components]
        label_risk_component_votes[label].append(
            sum(value >= float(unknown_risk_threshold) for value in candidate_components)
            / float(max(len(candidate_components), 1))
        )
        label_candidate_receiver_counts[label] += 1
        if int(source_rank) == 1:
            label_top1_receiver_counts[label] += 1
        label_min_evidence_rank[label] = min(label_min_evidence_rank[label], int(source_rank))

    for row in selected:
        weight = max(0.0, _float(row, "reliability", 1.0))
        label = _str(row, "predicted_label", "")
        weights.append(weight)
        risks.append(_float(row, "unknown_risk", 0.0))
        margin_value = max(0.0, _float(row, "known_margin", 0.0))
        margins.append(margin_value)
        support_density_is_missing = not _has_finite_float(row, "support_density")
        radius_z_is_missing = not _has_finite_float(row, "class_radius_z")
        support_density_value = _float(row, "support_density", 1.0)
        radius_z_value = _float(row, "class_radius_z", 0.0)
        class_conformal_pvalue = _float(row, "class_conformal_pvalue", 0.0)
        class_conformal_support_count = _float(row, "class_conformal_support_count", 0.0)
        support_density_missing.append(support_density_is_missing)
        radius_z_missing.append(radius_z_is_missing)
        support_densities.append(support_density_value)
        radius_z_values.append(radius_z_value)
        class_conformal_pvalues.append(class_conformal_pvalue)
        class_conformal_support_counts.append(class_conformal_support_count)
        row_seen_labels: set[str] = set()
        if label and label != UNKNOWN_LABEL:
            score_value = max(0.0, _float(row, "known_score", 1.0))
            scores.append(score_value)
            _add_label_candidate(
                row_seen_labels=row_seen_labels,
                label=label,
                weight=weight,
                score_value=score_value,
                margin_value=margin_value,
                support_density_value=support_density_value,
                radius_z_value=radius_z_value,
                support_density_is_missing=support_density_is_missing,
                radius_z_is_missing=radius_z_is_missing,
                conformal_pvalue=class_conformal_pvalue,
                conformal_support_count=class_conformal_support_count,
                conformal_weighted=policy == "cp_set_cvs",
                source_rank=1,
                unknown_risk_value=_float(row, "unknown_risk", 0.0),
                score_risk_value=_float(row, "score_risk", _float(row, "unknown_risk", 0.0)),
                radius_risk_value=_float(row, "radius_risk", _float(row, "unknown_risk", 0.0)),
                margin_risk_value=_float(row, "margin_risk", _float(row, "unknown_risk", 0.0)),
                mahalanobis_risk_value=_float(row, "mahalanobis_risk", _float(row, "unknown_risk", 0.0)),
                evt_risk_value=_float(row, "evt_risk", _float(row, "unknown_risk", 0.0)),
                oldness_risk_value=_float(row, "oldness_risk", _float(row, "unknown_risk", 0.0)),
                virtual_unknown_risk_value=_float(row, "virtual_unknown_risk", 0.0),
            )
        if policy == "cp_set_cvs":
            top_m = max(0, int(_float(row, "class_evidence_top_m", 0.0)))
            for rank in range(1, top_m + 1):
                top_label = _str(row, f"class_evidence_top{rank}_label", "")
                if not top_label or top_label == UNKNOWN_LABEL:
                    continue
                top_score = _float(
                    row,
                    f"class_evidence_top{rank}_score",
                    _float(row, "known_score", 0.0) if rank == 1 and top_label == label else 0.0,
                )
                top_pvalue = _float(row, f"class_evidence_top{rank}_conformal_pvalue", class_conformal_pvalue)
                top_support_count = _float(
                    row,
                    f"class_evidence_top{rank}_support_count",
                    class_conformal_support_count,
                )
                top_margin = _float(row, f"class_evidence_top{rank}_margin", margin_value)
                top_radius_z_is_missing = not _has_finite_float(row, f"class_evidence_top{rank}_class_radius_z")
                top_radius_z = _float(row, f"class_evidence_top{rank}_class_radius_z", radius_z_value)
                _add_label_candidate(
                    row_seen_labels=row_seen_labels,
                    label=top_label,
                    weight=weight,
                    score_value=top_score,
                    margin_value=top_margin,
                    support_density_value=support_density_value,
                    radius_z_value=top_radius_z,
                    support_density_is_missing=support_density_is_missing,
                    radius_z_is_missing=top_radius_z_is_missing,
                    conformal_pvalue=top_pvalue,
                    conformal_support_count=top_support_count,
                    conformal_weighted=True,
                    source_rank=rank,
                    unknown_risk_value=_float(row, f"class_evidence_top{rank}_unknown_risk", _float(row, "unknown_risk", 0.0)),
                    score_risk_value=_float(row, f"class_evidence_top{rank}_score_risk", _float(row, "score_risk", _float(row, "unknown_risk", 0.0))),
                    radius_risk_value=_float(row, f"class_evidence_top{rank}_radius_risk", _float(row, "radius_risk", _float(row, "unknown_risk", 0.0))),
                    margin_risk_value=_float(row, f"class_evidence_top{rank}_margin_risk", _float(row, "margin_risk", _float(row, "unknown_risk", 0.0))),
                    mahalanobis_risk_value=_float(row, f"class_evidence_top{rank}_mahalanobis_risk", _float(row, "mahalanobis_risk", _float(row, "unknown_risk", 0.0))),
                    evt_risk_value=_float(row, f"class_evidence_top{rank}_evt_risk", _float(row, "evt_risk", _float(row, "unknown_risk", 0.0))),
                    oldness_risk_value=_float(row, f"class_evidence_top{rank}_oldness_risk", _float(row, "oldness_risk", _float(row, "unknown_risk", 0.0))),
                    virtual_unknown_risk_value=_float(row, f"class_evidence_top{rank}_virtual_unknown_risk", _float(row, "virtual_unknown_risk", 0.0)),
                )
        score_risk_value = _float(row, "score_risk", _float(row, "unknown_risk", 0.0))
        radius_risk_value = _float(row, "radius_risk", _float(row, "unknown_risk", 0.0))
        margin_risk_value = _float(row, "margin_risk", _float(row, "unknown_risk", 0.0))
        has_mahalanobis = "mahalanobis_risk" in row
        has_evt = "evt_risk" in row
        has_oldness = "oldness_risk" in row
        has_virtual_unknown = "virtual_unknown_risk" in row
        mahalanobis_risk_value = _float(row, "mahalanobis_risk", _float(row, "unknown_risk", 0.0))
        evt_risk_value = _float(row, "evt_risk", _float(row, "unknown_risk", 0.0))
        oldness_risk_value = _float(row, "oldness_risk", _float(row, "unknown_risk", 0.0))
        virtual_unknown_risk_value = _float(row, "virtual_unknown_risk", 0.0)
        score_risks.append(score_risk_value)
        radius_risks.append(radius_risk_value)
        margin_risks.append(margin_risk_value)
        mahalanobis_risks.append(mahalanobis_risk_value)
        evt_risks.append(evt_risk_value)
        oldness_risks.append(oldness_risk_value)
        virtual_unknown_risks.append(virtual_unknown_risk_value)
        if active_components is None:
            component_values = [score_risk_value, radius_risk_value, margin_risk_value]
            if has_mahalanobis:
                component_values.append(mahalanobis_risk_value)
            if has_evt:
                component_values.append(evt_risk_value)
            if has_oldness:
                component_values.append(oldness_risk_value)
            if has_virtual_unknown:
                component_values.append(virtual_unknown_risk_value)
        else:
            row_values = {
                "score": score_risk_value,
                "radius": radius_risk_value,
                "margin": margin_risk_value,
                "mahalanobis": mahalanobis_risk_value,
                "evt": evt_risk_value,
                "oldness": oldness_risk_value,
                "virtual_unknown": virtual_unknown_risk_value,
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
    virtual_unknown_risk = _weighted_quantile(virtual_unknown_risks, weights, unknown_quantile)
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

    label_count = defaultdict(int, label_candidate_receiver_counts)
    label_rank_scores = {}
    for item, weighted_score in label_scores.items():
        count = float(label_count[item])
        mean_margin_for_label = sum(label_margins[item]) / max(len(label_margins[item]), 1)
        max_score_for_label = label_max_scores[item]
        if label_fusion_policy == "vote_sum":
            rank_score = count + 1e-3 * weighted_score
        elif label_fusion_policy == "vote_margin":
            rank_score = count + mean_margin_for_label + 1e-3 * weighted_score
        elif label_fusion_policy == "max_score":
            rank_score = max_score_for_label + 1e-3 * count
        else:
            rank_score = weighted_score
        label_rank_scores[item] = float(rank_score)
    if label_rank_scores:
        label = max(label_rank_scores.items(), key=lambda item: (item[1], item[0]))[0]
        score = label_scores[label]
    else:
        label, score = "", 0.0
    if label:
        label_weight_total = label_weight_totals[label]
        if label_weight_total > 0.0:
            mean_score = label_raw_scores[label] / max(label_weight_total, 1e-12)
        if label_margins[label]:
            mean_margin = sum(label_margins[label]) / len(label_margins[label])
    ranked_label_scores = sorted(label_rank_scores.values(), reverse=True)
    top_label_score = ranked_label_scores[0] if ranked_label_scores else 0.0
    second_label_score = ranked_label_scores[1] if len(ranked_label_scores) > 1 else 0.0
    label_score_total = sum(max(0.0, value) for value in label_scores.values())
    score_gap_ratio = (top_label_score - second_label_score) / max(label_score_total, 1e-12)
    ranked_counts = sorted(label_count.values(), reverse=True)
    top_count = ranked_counts[0] if ranked_counts else 0
    second_count = ranked_counts[1] if len(ranked_counts) > 1 else 0
    receiver_n = max(len(selected), 1)
    agreement = float(top_count) / float(receiver_n)
    vote_gap = float(top_count - second_count) / float(receiver_n)
    selected_label_top1_receivers = int(label_top1_receiver_counts[label]) if label else 0
    selected_label_candidate_receivers = int(label_candidate_receiver_counts[label]) if label else 0
    selected_label_min_evidence_rank = int(label_min_evidence_rank[label]) if label and label in label_min_evidence_rank else 0
    selected_label_support_density_values = label_support_density_values[label]
    selected_label_radius_z_values = label_radius_z_values[label]
    label_class_conformal_values = label_conformal_pvalues[label]
    label_class_conformal_count_values = label_conformal_support_counts[label]
    selected_label_unknown_risk_values = label_unknown_risk_values[label]
    selected_label_risk_component_votes = label_risk_component_votes[label]
    label_support_density_missing = any(label_support_density_missing_values[label])
    label_radius_z_missing = any(label_radius_z_missing_values[label])
    label_support_density = (
        sum(selected_label_support_density_values) / len(selected_label_support_density_values)
        if selected_label_support_density_values
        else 0.0
    )
    label_radius_z = max(selected_label_radius_z_values) if selected_label_radius_z_values else 0.0
    label_class_conformal_pvalue = (
        sum(label_class_conformal_values) / len(label_class_conformal_values)
        if label_class_conformal_values
        else 0.0
    )
    label_class_conformal_support_count = (
        min(label_class_conformal_count_values) if label_class_conformal_count_values else 0.0
    )
    label_unknown_risk = _percentile(selected_label_unknown_risk_values, unknown_quantile)
    label_risk_component_agreement = (
        sum(selected_label_risk_component_votes) / len(selected_label_risk_component_votes)
        if selected_label_risk_component_votes
        else risk_component_agreement
    )
    latency_ms = float(max((_float(row, "latency_ms", 0.0) for row in selected), default=0.0))
    within_request_budget = bool(
        can_request_more
        and float(latency_budget_ms) > 0.0
        and latency_ms < float(latency_budget_ms)
    )
    rescue_labels = set(str(item) for item in (seen_new_rescue_labels or set()) if str(item))
    rescue_enabled = bool(seen_new_rescue_enabled and rescue_labels)
    rescue_label_match = bool(label and label in rescue_labels)
    if label and label in rescue_labels:
        output_label_set = "seen_new"
    elif label and label in known_old_labels:
        output_label_set = "old"
    elif label:
        output_label_set = "other"
    else:
        output_label_set = ""

    def _class_set_gate_decision() -> tuple[bool, str]:
        if not class_set_gate_enabled or output_label_set not in {"old", "seen_new"}:
            return True, ""
        if output_label_set == "old":
            min_receivers = max(1, int(old_gate_min_receivers))
            max_risk = float(old_gate_max_effective_unknown_risk)
            max_agreement = float(old_gate_max_component_agreement)
            min_density = float(old_gate_min_support_density)
            max_radius_z = float(old_gate_max_radius_z)
        else:
            min_receivers = max(1, int(seen_new_gate_min_receivers))
            max_risk = float(seen_new_gate_max_effective_unknown_risk)
            max_agreement = float(seen_new_gate_max_component_agreement)
            min_density = float(seen_new_gate_min_support_density)
            max_radius_z = float(seen_new_gate_max_radius_z)
        reasons = []
        if len(selected) < min_receivers:
            reasons.append(f"min_receivers:{len(selected)}<{min_receivers}")
        if effective_unknown_risk > max_risk:
            reasons.append(f"effective_unknown_risk>{max_risk:.6g}")
        if decision_risk_component_agreement > max_agreement:
            reasons.append(f"risk_component_agreement>{max_agreement:.6g}")
        if min_density > 0.0 and label_support_density_missing:
            reasons.append("support_density:missing")
        elif label_support_density < min_density:
            reasons.append(f"support_density:{label_support_density:.6g}<{min_density:.6g}")
        if max_radius_z < 1.0e12 and label_radius_z_missing:
            reasons.append("radius_z:missing")
        elif label_radius_z > max_radius_z:
            reasons.append(f"radius_z:{label_radius_z:.6g}>{max_radius_z:.6g}")
        return not reasons, ",".join(reasons)

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
    elif policy in {"scorer_cvs", "cp_set_cvs"}:
        strong_consensus = (
            vote_gap > float(consensus_gap_threshold) or score_gap_ratio > float(consensus_gap_threshold)
        ) and agreement >= 0.5
        strong_known = (
            bool(label)
            and strong_consensus
            and mean_margin >= float(accept_margin_threshold)
            and mean_score >= float(consensus_score_threshold)
        )
        rescue_applied = bool(
            rescue_enabled
            and rescue_label_match
            and strong_known
            and agreement >= float(seen_new_rescue_min_agreement)
            and mean_score >= max(float(consensus_score_threshold), float(seen_new_rescue_min_score))
            and mean_margin >= max(float(accept_margin_threshold), float(seen_new_rescue_min_margin))
        )
        conformal_rescue_applied = bool(
            policy == "scorer_cvs"
            and conformal_rescue_enabled
            and output_label_set in {"old", "seen_new"}
            and strong_known
            and agreement >= float(conformal_rescue_min_agreement)
            and label_class_conformal_pvalue >= float(conformal_rescue_min_pvalue)
            and risk_component_agreement < float(scorer_component_vote_threshold)
        )
        cp_set_gate_passed = bool(
            policy != "cp_set_cvs"
            or (
                output_label_set in {"old", "seen_new"}
                and label_class_conformal_support_count >= 1.0
                and (selected_label_top1_receivers >= 1 or selected_label_candidate_receivers >= 2)
                and agreement >= float(conformal_rescue_min_agreement)
                and label_class_conformal_pvalue >= float(conformal_rescue_min_pvalue)
            )
        )
        risk_scale = 1.0
        if rescue_applied:
            risk_scale = min(risk_scale, max(0.0, min(1.0, float(seen_new_rescue_risk_scale))))
        if conformal_rescue_applied:
            risk_scale = min(risk_scale, max(0.0, min(1.0, float(conformal_rescue_risk_scale))))
        decision_unknown_risk = label_unknown_risk if policy == "cp_set_cvs" and label else unknown_risk
        decision_risk_component_agreement = (
            label_risk_component_agreement if policy == "cp_set_cvs" and label else risk_component_agreement
        )
        effective_unknown_risk = decision_unknown_risk * risk_scale
        high_risk = effective_unknown_risk >= float(unknown_risk_threshold)
        multi_channel_risk = decision_risk_component_agreement >= float(scorer_component_vote_threshold)
        gate_passed, gate_reason = _class_set_gate_decision()
        if high_risk and multi_channel_risk and (policy == "cp_set_cvs" or not strong_known):
            decision = "unknown_reject"
            output_label = UNKNOWN_LABEL
        elif high_risk:
            decision = "defer"
            output_label = ""
        elif strong_known and not cp_set_gate_passed and within_request_budget:
            decision = "request_more"
            output_label = ""
        elif strong_known and not cp_set_gate_passed:
            decision = "defer"
            output_label = ""
        elif strong_known and not gate_passed and within_request_budget:
            decision = "request_more"
            output_label = ""
        elif strong_known and not gate_passed:
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
        raise ValueError("fusion_policy must be risk_margin, consensus_veto, scorer_cvs, or cp_set_cvs")
    else:
        raise ValueError("fusion_policy must be risk_margin, consensus_veto, scorer_cvs, or cp_set_cvs")

    resource_budget_reason = _resource_budget_reason(
        selected,
        max_event_bytes=max_event_bytes,
        max_event_latency_ms=max_event_latency_ms,
    )
    resource_budget_passed = resource_budget_reason == ""
    if not resource_budget_passed:
        decision = "defer"
        output_label = ""

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
        "virtual_unknown_risk": float(virtual_unknown_risk),
        "effective_unknown_risk": float(locals().get("effective_unknown_risk", unknown_risk)),
        "decision_unknown_risk": float(locals().get("decision_unknown_risk", unknown_risk)),
        "label_unknown_risk": float(label_unknown_risk),
        "seen_new_rescue_applied": bool(locals().get("rescue_applied", False)),
        "seen_new_rescue_label_match": bool(rescue_label_match if policy == "scorer_cvs" else False),
        "conformal_rescue_applied": bool(locals().get("conformal_rescue_applied", False)),
        "label_class_conformal_pvalue": float(label_class_conformal_pvalue),
        "label_class_conformal_support_count": float(label_class_conformal_support_count),
        "label_candidate_receiver_count": int(selected_label_candidate_receivers),
        "label_top1_receiver_count": int(selected_label_top1_receivers),
        "label_min_evidence_rank": int(selected_label_min_evidence_rank),
        "label_risk_component_agreement": float(label_risk_component_agreement),
        "filtered_candidate_count": int(filtered_candidate_count),
        "cp_set_gate_passed": bool(locals().get("cp_set_gate_passed", True)),
        "class_set_gate_applied": bool(class_set_gate_enabled and output_label_set in {"old", "seen_new"}),
        "class_set_gate_passed": bool(locals().get("gate_passed", True)),
        "class_set_gate_reason": str(locals().get("gate_reason", "")),
        "output_label_set": output_label_set,
        "label_fusion_policy": label_fusion_policy,
        "label_support_density": float(label_support_density),
        "label_radius_z": float(label_radius_z),
        "risk_component_agreement": float(risk_component_agreement),
        "known_margin": float(mean_margin),
        "mean_known_score": float(mean_score),
        "known_score": float(score),
        "agreement": float(agreement),
        "vote_gap": float(vote_gap),
        "score_gap_ratio": float(score_gap_ratio),
        "bytes": float(sum(_float(row, "bytes", 0.0) for row in selected)),
        "latency_ms": latency_ms,
        "resource_budget_passed": bool(resource_budget_passed),
        "resource_budget_reason": resource_budget_reason,
        "max_event_bytes": float(max_event_bytes),
        "max_event_latency_ms": float(max_event_latency_ms),
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
    label_fusion_policy: str = "score_sum",
    latency_budget_ms: float = 0.0,
    max_event_bytes: float = 0.0,
    max_event_latency_ms: float = 0.0,
    old_labels: set[str] | None = None,
    seen_new_rescue_labels: set[str] | None = None,
    seen_new_rescue_enabled: bool = False,
    seen_new_rescue_risk_scale: float = 1.0,
    seen_new_rescue_min_score: float = 0.0,
    seen_new_rescue_min_margin: float = 0.0,
    seen_new_rescue_min_agreement: float = 0.5,
    conformal_rescue_enabled: bool = False,
    conformal_rescue_min_pvalue: float = 0.05,
    conformal_rescue_risk_scale: float = 0.5,
    conformal_rescue_min_agreement: float = 0.5,
    class_set_gate_enabled: bool = False,
    old_gate_min_receivers: int = 1,
    old_gate_max_effective_unknown_risk: float = 1.0,
    old_gate_max_component_agreement: float = 1.0,
    old_gate_min_support_density: float = 0.0,
    old_gate_max_radius_z: float = 1.0e12,
    seen_new_gate_min_receivers: int = 1,
    seen_new_gate_max_effective_unknown_risk: float = 1.0,
    seen_new_gate_max_component_agreement: float = 1.0,
    seen_new_gate_min_support_density: float = 0.0,
    seen_new_gate_max_radius_z: float = 1.0e12,
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
            label_fusion_policy=label_fusion_policy,
            can_request_more=used < min(max_receivers, len(ordered)),
            latency_budget_ms=latency_budget_ms,
            max_event_bytes=max_event_bytes,
            max_event_latency_ms=max_event_latency_ms,
            old_labels=old_labels,
            seen_new_rescue_labels=seen_new_rescue_labels,
            seen_new_rescue_enabled=seen_new_rescue_enabled,
            seen_new_rescue_risk_scale=seen_new_rescue_risk_scale,
            seen_new_rescue_min_score=seen_new_rescue_min_score,
            seen_new_rescue_min_margin=seen_new_rescue_min_margin,
            seen_new_rescue_min_agreement=seen_new_rescue_min_agreement,
            conformal_rescue_enabled=conformal_rescue_enabled,
            conformal_rescue_min_pvalue=conformal_rescue_min_pvalue,
            conformal_rescue_risk_scale=conformal_rescue_risk_scale,
            conformal_rescue_min_agreement=conformal_rescue_min_agreement,
            class_set_gate_enabled=class_set_gate_enabled,
            old_gate_min_receivers=old_gate_min_receivers,
            old_gate_max_effective_unknown_risk=old_gate_max_effective_unknown_risk,
            old_gate_max_component_agreement=old_gate_max_component_agreement,
            old_gate_min_support_density=old_gate_min_support_density,
            old_gate_max_radius_z=old_gate_max_radius_z,
            seen_new_gate_min_receivers=seen_new_gate_min_receivers,
            seen_new_gate_max_effective_unknown_risk=seen_new_gate_max_effective_unknown_risk,
            seen_new_gate_max_component_agreement=seen_new_gate_max_component_agreement,
            seen_new_gate_min_support_density=seen_new_gate_min_support_density,
            seen_new_gate_max_radius_z=seen_new_gate_max_radius_z,
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
        float(fused.get("virtual_unknown_risk", 0.0)),
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
    label_fusion_policy: str = "score_sum",
    latency_budget_ms: float = 0.0,
    max_event_bytes: float = 0.0,
    max_event_latency_ms: float = 0.0,
    adaptive_gain_min_risk: float = 0.80,
    adaptive_gain_latency_weight: float = 0.0,
    adaptive_gain_bytes_weight: float = 0.0,
    adaptive_gain_disagreement_weight: float = 0.5,
    old_labels: set[str] | None = None,
    seen_new_rescue_labels: set[str] | None = None,
    seen_new_rescue_enabled: bool = False,
    seen_new_rescue_risk_scale: float = 1.0,
    seen_new_rescue_min_score: float = 0.0,
    seen_new_rescue_min_margin: float = 0.0,
    seen_new_rescue_min_agreement: float = 0.5,
    conformal_rescue_enabled: bool = False,
    conformal_rescue_min_pvalue: float = 0.05,
    conformal_rescue_risk_scale: float = 0.5,
    conformal_rescue_min_agreement: float = 0.5,
    class_set_gate_enabled: bool = False,
    old_gate_min_receivers: int = 1,
    old_gate_max_effective_unknown_risk: float = 1.0,
    old_gate_max_component_agreement: float = 1.0,
    old_gate_min_support_density: float = 0.0,
    old_gate_max_radius_z: float = 1.0e12,
    seen_new_gate_min_receivers: int = 1,
    seen_new_gate_max_effective_unknown_risk: float = 1.0,
    seen_new_gate_max_component_agreement: float = 1.0,
    seen_new_gate_min_support_density: float = 0.0,
    seen_new_gate_max_radius_z: float = 1.0e12,
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
            label_fusion_policy=label_fusion_policy,
            can_request_more=len(selected) < budget,
            latency_budget_ms=latency_budget_ms,
            max_event_bytes=max_event_bytes,
            max_event_latency_ms=max_event_latency_ms,
            old_labels=old_labels,
            seen_new_rescue_labels=seen_new_rescue_labels,
            seen_new_rescue_enabled=seen_new_rescue_enabled,
            seen_new_rescue_risk_scale=seen_new_rescue_risk_scale,
            seen_new_rescue_min_score=seen_new_rescue_min_score,
            seen_new_rescue_min_margin=seen_new_rescue_min_margin,
            seen_new_rescue_min_agreement=seen_new_rescue_min_agreement,
            conformal_rescue_enabled=conformal_rescue_enabled,
            conformal_rescue_min_pvalue=conformal_rescue_min_pvalue,
            conformal_rescue_risk_scale=conformal_rescue_risk_scale,
            conformal_rescue_min_agreement=conformal_rescue_min_agreement,
            class_set_gate_enabled=class_set_gate_enabled,
            old_gate_min_receivers=old_gate_min_receivers,
            old_gate_max_effective_unknown_risk=old_gate_max_effective_unknown_risk,
            old_gate_max_component_agreement=old_gate_max_component_agreement,
            old_gate_min_support_density=old_gate_min_support_density,
            old_gate_max_radius_z=old_gate_max_radius_z,
            seen_new_gate_min_receivers=seen_new_gate_min_receivers,
            seen_new_gate_max_effective_unknown_risk=seen_new_gate_max_effective_unknown_risk,
            seen_new_gate_max_component_agreement=seen_new_gate_max_component_agreement,
            seen_new_gate_min_support_density=seen_new_gate_min_support_density,
            seen_new_gate_max_radius_z=seen_new_gate_max_radius_z,
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
        feasible = []
        for gain, row, parts in scored:
            if not _resource_budget_reason(
                [*selected, row],
                max_event_bytes=max_event_bytes,
                max_event_latency_ms=max_event_latency_ms,
            ):
                feasible.append((gain, row, parts))
        if not feasible:
            fused["adaptive_stop_reason"] = "resource_budget_exhausted"
            return fused
        gain, row, parts = max(
            feasible,
            key=lambda item: (item[0], _float(item[1], "reliability", 1.0), _str(item[1], "receiver_id")),
        )
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
    resource_budget_violations = 0
    adaptive_stop_reasons: defaultdict[str, int] = defaultdict(int)
    seen_new_rescue_total = 0

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
        resource_budget_violations += int(not bool(item.get("resource_budget_passed", True)))
        stop_reason = str(item.get("adaptive_stop_reason") or item.get("progressive_stop_reason") or "")
        if stop_reason:
            adaptive_stop_reasons[stop_reason] += 1
        seen_new_rescue_total += int(bool(item.get("seen_new_rescue_applied", False)))
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
        "resource_budget_violation_count": int(resource_budget_violations),
        "resource_budget_violation_rate": _safe_rate(resource_budget_violations, total_events),
        "open_set_confusion": dict(sorted(confusion.items())),
        "bytes_per_event": sum(bytes_values) / max(len(bytes_values), 1),
        "total_bytes": float(sum(bytes_values)),
        "latency_ms_p50": _percentile(latency_values, 0.50),
        "latency_ms_p95": _percentile(latency_values, 0.95),
        "collaboration_stop_reasons": dict(sorted(adaptive_stop_reasons.items())),
        "seen_new_rescue_count": int(seen_new_rescue_total),
        "seen_new_rescue_rate": _safe_rate(seen_new_rescue_total, total_events),
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
    label_fusion_policy: str = "score_sum",
    collaboration_policy: str = "fixed_k",
    latency_budget_ms: float = 0.0,
    max_event_bytes: float = 0.0,
    max_event_latency_ms: float = 0.0,
    adaptive_gain_min_risk: float = 0.80,
    adaptive_gain_latency_weight: float = 0.0,
    adaptive_gain_bytes_weight: float = 0.0,
    adaptive_gain_disagreement_weight: float = 0.5,
    seen_new_rescue_enabled: bool = False,
    seen_new_rescue_risk_scale: float = 1.0,
    seen_new_rescue_min_score: float = 0.0,
    seen_new_rescue_min_margin: float = 0.0,
    seen_new_rescue_min_agreement: float = 0.5,
    conformal_rescue_enabled: bool = False,
    conformal_rescue_min_pvalue: float = 0.05,
    conformal_rescue_risk_scale: float = 0.5,
    conformal_rescue_min_agreement: float = 0.5,
    class_set_gate_enabled: bool = False,
    old_gate_min_receivers: int = 1,
    old_gate_max_effective_unknown_risk: float = 1.0,
    old_gate_max_component_agreement: float = 1.0,
    old_gate_min_support_density: float = 0.0,
    old_gate_max_radius_z: float = 1.0e12,
    seen_new_gate_min_receivers: int = 1,
    seen_new_gate_max_effective_unknown_risk: float = 1.0,
    seen_new_gate_max_component_agreement: float = 1.0,
    seen_new_gate_min_support_density: float = 0.0,
    seen_new_gate_max_radius_z: float = 1.0e12,
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
                    label_fusion_policy=label_fusion_policy,
                    latency_budget_ms=latency_budget_ms,
                    max_event_bytes=max_event_bytes,
                    max_event_latency_ms=max_event_latency_ms,
                    old_labels=expected_old_labels,
                    seen_new_rescue_labels=expected_seen_new_labels,
                    seen_new_rescue_enabled=seen_new_rescue_enabled,
                    seen_new_rescue_risk_scale=seen_new_rescue_risk_scale,
                    seen_new_rescue_min_score=seen_new_rescue_min_score,
                    seen_new_rescue_min_margin=seen_new_rescue_min_margin,
                    seen_new_rescue_min_agreement=seen_new_rescue_min_agreement,
                    conformal_rescue_enabled=conformal_rescue_enabled,
                    conformal_rescue_min_pvalue=conformal_rescue_min_pvalue,
                    conformal_rescue_risk_scale=conformal_rescue_risk_scale,
                    conformal_rescue_min_agreement=conformal_rescue_min_agreement,
                    class_set_gate_enabled=class_set_gate_enabled,
                    old_gate_min_receivers=old_gate_min_receivers,
                    old_gate_max_effective_unknown_risk=old_gate_max_effective_unknown_risk,
                    old_gate_max_component_agreement=old_gate_max_component_agreement,
                    old_gate_min_support_density=old_gate_min_support_density,
                    old_gate_max_radius_z=old_gate_max_radius_z,
                    seen_new_gate_min_receivers=seen_new_gate_min_receivers,
                    seen_new_gate_max_effective_unknown_risk=seen_new_gate_max_effective_unknown_risk,
                    seen_new_gate_max_component_agreement=seen_new_gate_max_component_agreement,
                    seen_new_gate_min_support_density=seen_new_gate_min_support_density,
                    seen_new_gate_max_radius_z=seen_new_gate_max_radius_z,
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
                    label_fusion_policy=label_fusion_policy,
                    latency_budget_ms=latency_budget_ms,
                    max_event_bytes=max_event_bytes,
                    max_event_latency_ms=max_event_latency_ms,
                    adaptive_gain_min_risk=adaptive_gain_min_risk,
                    adaptive_gain_latency_weight=adaptive_gain_latency_weight,
                    adaptive_gain_bytes_weight=adaptive_gain_bytes_weight,
                    adaptive_gain_disagreement_weight=adaptive_gain_disagreement_weight,
                    old_labels=expected_old_labels,
                    seen_new_rescue_labels=expected_seen_new_labels,
                    seen_new_rescue_enabled=seen_new_rescue_enabled,
                    seen_new_rescue_risk_scale=seen_new_rescue_risk_scale,
                    seen_new_rescue_min_score=seen_new_rescue_min_score,
                    seen_new_rescue_min_margin=seen_new_rescue_min_margin,
                    seen_new_rescue_min_agreement=seen_new_rescue_min_agreement,
                    conformal_rescue_enabled=conformal_rescue_enabled,
                    conformal_rescue_min_pvalue=conformal_rescue_min_pvalue,
                    conformal_rescue_risk_scale=conformal_rescue_risk_scale,
                    conformal_rescue_min_agreement=conformal_rescue_min_agreement,
                    class_set_gate_enabled=class_set_gate_enabled,
                    old_gate_min_receivers=old_gate_min_receivers,
                    old_gate_max_effective_unknown_risk=old_gate_max_effective_unknown_risk,
                    old_gate_max_component_agreement=old_gate_max_component_agreement,
                    old_gate_min_support_density=old_gate_min_support_density,
                    old_gate_max_radius_z=old_gate_max_radius_z,
                    seen_new_gate_min_receivers=seen_new_gate_min_receivers,
                    seen_new_gate_max_effective_unknown_risk=seen_new_gate_max_effective_unknown_risk,
                    seen_new_gate_max_component_agreement=seen_new_gate_max_component_agreement,
                    seen_new_gate_min_support_density=seen_new_gate_min_support_density,
                    seen_new_gate_max_radius_z=seen_new_gate_max_radius_z,
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
                    label_fusion_policy=label_fusion_policy,
                    can_request_more=len({_str(row, "receiver_id") for row in group}) > int(k),
                    latency_budget_ms=latency_budget_ms,
                    max_event_bytes=max_event_bytes,
                    max_event_latency_ms=max_event_latency_ms,
                    old_labels=expected_old_labels,
                    seen_new_rescue_labels=expected_seen_new_labels,
                    seen_new_rescue_enabled=seen_new_rescue_enabled,
                    seen_new_rescue_risk_scale=seen_new_rescue_risk_scale,
                    seen_new_rescue_min_score=seen_new_rescue_min_score,
                    seen_new_rescue_min_margin=seen_new_rescue_min_margin,
                    seen_new_rescue_min_agreement=seen_new_rescue_min_agreement,
                    conformal_rescue_enabled=conformal_rescue_enabled,
                    conformal_rescue_min_pvalue=conformal_rescue_min_pvalue,
                    conformal_rescue_risk_scale=conformal_rescue_risk_scale,
                    conformal_rescue_min_agreement=conformal_rescue_min_agreement,
                    class_set_gate_enabled=class_set_gate_enabled,
                    old_gate_min_receivers=old_gate_min_receivers,
                    old_gate_max_effective_unknown_risk=old_gate_max_effective_unknown_risk,
                    old_gate_max_component_agreement=old_gate_max_component_agreement,
                    old_gate_min_support_density=old_gate_min_support_density,
                    old_gate_max_radius_z=old_gate_max_radius_z,
                    seen_new_gate_min_receivers=seen_new_gate_min_receivers,
                    seen_new_gate_max_effective_unknown_risk=seen_new_gate_max_effective_unknown_risk,
                    seen_new_gate_max_component_agreement=seen_new_gate_max_component_agreement,
                    seen_new_gate_min_support_density=seen_new_gate_min_support_density,
                    seen_new_gate_max_radius_z=seen_new_gate_max_radius_z,
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
        "label_fusion_policy": _normalize_scope(label_fusion_policy),
        "latency_budget_ms": float(latency_budget_ms),
        "max_event_bytes": float(max_event_bytes),
        "max_event_latency_ms": float(max_event_latency_ms),
        "adaptive_gain_min_risk": float(adaptive_gain_min_risk),
        "adaptive_gain_latency_weight": float(adaptive_gain_latency_weight),
        "adaptive_gain_bytes_weight": float(adaptive_gain_bytes_weight),
        "adaptive_gain_disagreement_weight": float(adaptive_gain_disagreement_weight),
        "seen_new_rescue_enabled": bool(seen_new_rescue_enabled),
        "seen_new_rescue_risk_scale": float(seen_new_rescue_risk_scale),
        "seen_new_rescue_min_score": float(seen_new_rescue_min_score),
        "seen_new_rescue_min_margin": float(seen_new_rescue_min_margin),
        "seen_new_rescue_min_agreement": float(seen_new_rescue_min_agreement),
        "conformal_rescue_enabled": bool(conformal_rescue_enabled),
        "conformal_rescue_min_pvalue": float(conformal_rescue_min_pvalue),
        "conformal_rescue_risk_scale": float(conformal_rescue_risk_scale),
        "conformal_rescue_min_agreement": float(conformal_rescue_min_agreement),
        "class_set_gate_enabled": bool(class_set_gate_enabled),
        "old_gate_min_receivers": int(old_gate_min_receivers),
        "old_gate_max_effective_unknown_risk": float(old_gate_max_effective_unknown_risk),
        "old_gate_max_component_agreement": float(old_gate_max_component_agreement),
        "old_gate_min_support_density": float(old_gate_min_support_density),
        "old_gate_max_radius_z": float(old_gate_max_radius_z),
        "seen_new_gate_min_receivers": int(seen_new_gate_min_receivers),
        "seen_new_gate_max_effective_unknown_risk": float(seen_new_gate_max_effective_unknown_risk),
        "seen_new_gate_max_component_agreement": float(seen_new_gate_max_component_agreement),
        "seen_new_gate_min_support_density": float(seen_new_gate_min_support_density),
        "seen_new_gate_max_radius_z": float(seen_new_gate_max_radius_z),
        "counts": out_counts,
    }
