from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


DEFAULT_JOINT_SAFE_WEIGHTS = {
    "val_tx": 0.20,
    "overall_tx": 0.20,
    "strict_udu": 0.25,
    "receiver_floor": 0.15,
    "sat_mean_tx": 0.10,
    "sat_strict_mean": 0.10,
}

DEFAULT_DROP_KEYS = (
    "strict_udu",
    "receiver_floor",
    "sat_mean_tx",
    "sat_floor_tx",
    "sat_strict_mean",
    "sat_strict_floor",
)

JOINT_SAFE_REQUIRED_BASE_KEYS = (
    "val_tx",
    "strict_udu",
    "receiver_floor",
)

JOINT_SAFE_REQUIRED_SATELLITE_KEYS = (
    "sat_mean_tx",
    "sat_floor_tx",
    "sat_strict_mean",
    "sat_strict_floor",
)


@dataclass(frozen=True)
class GuardDecision:
    fired: bool
    reason: str
    details: Dict[str, float]


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _finite_values(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        fv = finite_float(value)
        if math.isfinite(fv):
            out.append(float(fv))
    return out


def _tx_acc(stats: Mapping[str, Any] | None) -> float:
    return finite_float((stats or {}).get("tx_acc"))


def _sat_aggregate_scores(sat_test_stats: Mapping[str, Mapping[str, Any]] | None) -> list[float]:
    scores: list[float] = []
    for stats in (sat_test_stats or {}).values():
        if not isinstance(stats, Mapping):
            continue
        agg = stats.get("aggregate", {})
        if isinstance(agg, Mapping):
            score = _tx_acc(agg)
            if math.isfinite(score):
                scores.append(score)
    return scores


def _sat_strict_scores(sat_test_stats: Mapping[str, Mapping[str, Any]] | None) -> list[float]:
    return _finite_values(
        stats.get("strict_udu")
        for stats in (sat_test_stats or {}).values()
        if isinstance(stats, Mapping)
    )


def protected_metric_snapshot(
    *,
    val_stats: Mapping[str, Any] | None,
    test_stats: Mapping[str, Any] | None,
    named_test_stats: Mapping[str, Mapping[str, Any]] | None,
    sat_test_stats: Mapping[str, Mapping[str, Any]] | None,
) -> Dict[str, float]:
    named = named_test_stats or {}
    named_scores = _finite_values(_tx_acc(stats) for stats in named.values() if isinstance(stats, Mapping))
    strict_udu = _tx_acc(named.get("test_unseen_day_unseen_rx"))
    if not math.isfinite(strict_udu):
        strict_udu = finite_float((test_stats or {}).get("strict_udu"))

    sat_agg = _sat_aggregate_scores(sat_test_stats)
    sat_strict = _sat_strict_scores(sat_test_stats)
    return {
        "val_tx": _tx_acc(val_stats),
        "overall_tx": _tx_acc(test_stats),
        "strict_udu": strict_udu,
        "receiver_floor": min(named_scores) if named_scores else float("nan"),
        "sat_mean_tx": sum(sat_agg) / len(sat_agg) if sat_agg else float("nan"),
        "sat_floor_tx": min(sat_agg) if sat_agg else float("nan"),
        "sat_strict_mean": sum(sat_strict) / len(sat_strict) if sat_strict else float("nan"),
        "sat_strict_floor": min(sat_strict) if sat_strict else float("nan"),
    }


def missing_joint_safe_metrics(
    metrics: Mapping[str, Any],
    *,
    require_satellite: bool = True,
    extra_required: Iterable[str] | None = None,
) -> Tuple[str, ...]:
    required = list(JOINT_SAFE_REQUIRED_BASE_KEYS)
    if require_satellite:
        required.extend(JOINT_SAFE_REQUIRED_SATELLITE_KEYS)
    if extra_required:
        required.extend(str(key) for key in extra_required)

    missing: list[str] = []
    for key in required:
        if not math.isfinite(finite_float(metrics.get(key))):
            missing.append(key)
    return tuple(missing)


def joint_safe_score(
    metrics: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
    minimums: Mapping[str, float] | None = None,
    require_satellite: bool = False,
) -> float:
    weights = dict(weights or DEFAULT_JOINT_SAFE_WEIGHTS)
    minimums = dict(minimums or {})
    if missing_joint_safe_metrics(metrics, require_satellite=require_satellite):
        return float("-inf")
    for key, threshold in minimums.items():
        value = finite_float(metrics.get(key))
        if math.isfinite(float(threshold)) and (not math.isfinite(value) or value < float(threshold)):
            return float("-inf")

    total_weight = 0.0
    score = 0.0
    for key, weight in weights.items():
        value = finite_float(metrics.get(key))
        weight = float(weight)
        if not math.isfinite(value) or weight <= 0.0:
            continue
        total_weight += weight
        score += weight * value
    if total_weight <= 0.0:
        return float("-inf")
    return score / total_weight


def detect_one_epoch_drop(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    threshold_pp: float,
    keys: Iterable[str] = DEFAULT_DROP_KEYS,
) -> GuardDecision:
    if previous is None or float(threshold_pp) <= 0.0:
        return GuardDecision(False, "", {})
    details: Dict[str, float] = {}
    for key in keys:
        cur = finite_float(current.get(key))
        prev = finite_float(previous.get(key))
        if math.isfinite(cur) and math.isfinite(prev):
            drop = prev - cur
            if drop >= float(threshold_pp):
                details[f"{key}_drop_pp"] = float(drop)
    if not details:
        return GuardDecision(False, "", {})
    reason = "one_epoch_drop:" + ",".join(sorted(details))
    return GuardDecision(True, reason, details)


def detect_paic_variance_guard(
    current_train: Mapping[str, Any],
    previous_train: Mapping[str, Any] | None,
    *,
    sat_ce_delta: float,
    grad_delta: float,
    reliable_drop: float,
    domain_delta: float = 0.0,
    sat_cons_delta: float = 0.0,
) -> GuardDecision:
    if previous_train is None:
        return GuardDecision(False, "", {})

    cur_sat = finite_float(current_train.get("train/w_loss_sat_cls_labeled"))
    prev_sat = finite_float(previous_train.get("train/w_loss_sat_cls_labeled"))
    cur_grad = finite_float(current_train.get("train/grad_total"))
    prev_grad = finite_float(previous_train.get("train/grad_total"))
    cur_rel = finite_float(current_train.get("train/reliable_ratio"))
    prev_rel = finite_float(previous_train.get("train/reliable_ratio"))
    cur_dom = finite_float(current_train.get("train/loss_domain_labeled"))
    prev_dom = finite_float(previous_train.get("train/loss_domain_labeled"))
    cur_cons = finite_float(current_train.get("train/w_loss_sat_cons_labeled"))
    prev_cons = finite_float(previous_train.get("train/w_loss_sat_cons_labeled"))

    details = {
        "sat_ce_delta": cur_sat - prev_sat,
        "grad_total_delta": cur_grad - prev_grad,
        "pseudo_reliable_drop": prev_rel - cur_rel,
    }
    if math.isfinite(cur_dom) and math.isfinite(prev_dom):
        details["domain_loss_delta"] = cur_dom - prev_dom
    if math.isfinite(cur_cons) and math.isfinite(prev_cons):
        details["sat_cons_delta"] = cur_cons - prev_cons

    required = (
        details["sat_ce_delta"] >= float(sat_ce_delta),
        details["grad_total_delta"] >= float(grad_delta),
        details["pseudo_reliable_drop"] >= float(reliable_drop),
    )
    optional_domain_ok = float(domain_delta) <= 0.0 or details.get("domain_loss_delta", float("-inf")) >= float(domain_delta)
    optional_cons_ok = float(sat_cons_delta) <= 0.0 or details.get("sat_cons_delta", float("-inf")) >= float(sat_cons_delta)
    if all(required) and optional_domain_ok and optional_cons_ok:
        return GuardDecision(True, "paic_variance", {k: float(v) for k, v in details.items() if math.isfinite(v)})
    return GuardDecision(False, "", {k: float(v) for k, v in details.items() if math.isfinite(v)})


def guard_minimums_from_args(args: Any) -> Dict[str, float]:
    mapping = {
        "strict_udu": "joint_guard_min_strict_udu",
        "receiver_floor": "joint_guard_min_receiver_floor",
        "sat_mean_tx": "joint_guard_min_sat_mean",
        "sat_floor_tx": "joint_guard_min_sat_floor",
        "sat_strict_mean": "joint_guard_min_sat_strict_mean",
        "sat_strict_floor": "joint_guard_min_sat_strict_floor",
    }
    out: Dict[str, float] = {}
    for metric, attr in mapping.items():
        value = finite_float(getattr(args, attr, 0.0), 0.0)
        if value > 0.0:
            out[metric] = value
    return out
