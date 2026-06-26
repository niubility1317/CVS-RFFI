"""Safety and rollback gates for deployment-time CVS adaptation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SafetyRule:
    metric: str
    mode: str
    threshold: float
    baseline_metric: str | None = None
    description: str = ""


@dataclass
class RollbackDecision:
    accepted: bool
    triggered_rules: list[dict[str, Any]] = field(default_factory=list)
    skipped_rules: list[dict[str, Any]] = field(default_factory=list)
    rules: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rollback_triggered(self) -> bool:
        return not bool(self.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "rollback_triggered": self.rollback_triggered,
            "triggered_rules": self.triggered_rules,
            "skipped_rules": self.skipped_rules,
            "rules": self.rules,
        }


DEFAULT_SFE_ROLLBACK_RULES = [
    SafetyRule("old_class_accuracy", "max_drop", 0.05, description="old-class accuracy drop guard"),
    SafetyRule("known_accuracy", "max_drop", 0.05, description="known-class accuracy drop guard"),
    SafetyRule("unknown_false_accept_rate", "max_rise", 0.05, description="unknown false-accept guard"),
    SafetyRule("coverage", "min", 0.20, description="coverage collapse guard"),
]

DEFAULT_TARGET_ROLLBACK_RULES = [
    SafetyRule("test.tx_acc", "max_drop", 1.0, description="overall clean test accuracy drop guard"),
    SafetyRule("sat_score", "max_drop", 1.0, description="satellite stress mean drop guard"),
    SafetyRule("target.tx_acc", "min_gain", 0.0, description="target accuracy must not regress"),
]


def metric_value(metrics: Mapping[str, Any], path: str) -> float | None:
    cur: Any = metrics
    for part in str(path).split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _rule_from_mapping(item: Mapping[str, Any]) -> SafetyRule:
    return SafetyRule(
        metric=str(item["metric"]),
        mode=str(item["mode"]),
        threshold=float(item["threshold"]),
        baseline_metric=str(item["baseline_metric"]) if item.get("baseline_metric") is not None else None,
        description=str(item.get("description", "")),
    )


def rules_from_policy(policy: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None, *, default: Iterable[SafetyRule]) -> list[SafetyRule]:
    if policy is None:
        return list(default)
    if isinstance(policy, Mapping):
        items = policy.get("rules", [])
    else:
        items = policy
    rules = [_rule_from_mapping(item) for item in items]
    return rules if rules else list(default)


def evaluate_rollback_gate(
    *,
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
    rules: Iterable[SafetyRule],
) -> RollbackDecision:
    triggered: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    normalized_rules = list(rules)
    for rule in normalized_rules:
        after = metric_value(after_metrics, rule.metric)
        before = metric_value(before_metrics, rule.baseline_metric or rule.metric)
        mode = str(rule.mode).lower()
        if after is None or (mode in {"max_drop", "max_rise", "min_gain", "max_gain"} and before is None):
            skipped.append(
                {
                    "metric": rule.metric,
                    "mode": mode,
                    "threshold": float(rule.threshold),
                    "reason": "metric_missing_or_nonfinite",
                }
            )
            continue

        violation = False
        observed = after
        if mode == "min":
            violation = after < float(rule.threshold)
        elif mode == "max":
            violation = after > float(rule.threshold)
        elif mode == "max_drop":
            observed = before - after
            violation = observed > float(rule.threshold)
        elif mode == "max_rise":
            observed = after - before
            violation = observed > float(rule.threshold)
        elif mode == "min_gain":
            observed = after - before
            violation = observed < float(rule.threshold)
        elif mode == "max_gain":
            observed = after - before
            violation = observed > float(rule.threshold)
        else:
            raise ValueError(f"unknown safety rule mode: {rule.mode}")
        if violation:
            triggered.append(
                {
                    "metric": rule.metric,
                    "baseline_metric": rule.baseline_metric or rule.metric,
                    "mode": mode,
                    "threshold": float(rule.threshold),
                    "before": before,
                    "after": after,
                    "observed": observed,
                    "description": rule.description,
                }
            )
    return RollbackDecision(
        accepted=len(triggered) == 0,
        triggered_rules=triggered,
        skipped_rules=skipped,
        rules=[asdict(rule) for rule in normalized_rules],
    )

