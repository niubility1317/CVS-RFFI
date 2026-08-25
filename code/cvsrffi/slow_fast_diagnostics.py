"""Truth-last response-surface diagnostics for slow/fast shadow states."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    centered_left = [value - left_mean for value in left_ranks]
    centered_right = [value - right_mean for value in right_ranks]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0.0:
        return None
    return float(
        sum(left_value * right_value for left_value, right_value in zip(centered_left, centered_right))
        / denominator
    )


def _mean_old_accuracy(payload: Mapping[str, Any]) -> float:
    metrics = payload.get("old_class_metrics", payload)
    if not isinstance(metrics, Mapping):
        raise ValueError("state score must contain old-class metrics")
    return _finite(metrics.get("mean_old_acc"), "mean_old_acc")


def build_shadow_response_surface(
    row_scores: Sequence[Mapping[str, Any]],
    support_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join frozen support diagnostics to truth-last query gains without selecting a state."""

    if not row_scores or len(row_scores) != len(support_receipts):
        raise ValueError("row scores and support receipts must be nonempty and aligned")
    receipt_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in support_receipts:
        key = (str(receipt.get("candidate_id")), str(receipt.get("scenario")))
        if key in receipt_by_key:
            raise ValueError("support receipts contain a duplicate candidate/scenario row")
        receipt_by_key[key] = receipt

    rows: list[dict[str, Any]] = []
    state_names: set[str] = set()
    for score in row_scores:
        key = (str(score.get("candidate_id")), str(score.get("scenario")))
        if key not in receipt_by_key:
            raise ValueError("row score lacks its matching support receipt")
        states = score.get("states")
        support = receipt_by_key[key].get("shadow_support_diagnostics")
        if not isinstance(states, Mapping) or "DA0_REG0" not in states:
            raise ValueError("row score must expose DA0_REG0 and shadow states")
        if not isinstance(support, Mapping) or set(support) != set(states):
            raise ValueError("support diagnostics must exactly cover scored shadow states")
        baseline_accuracy = _mean_old_accuracy(states["DA0_REG0"])
        ordered_states = sorted(states, key=lambda state: (state != "DA0_REG0", state))
        for state in ordered_states:
            support_payload = support[state]
            if not isinstance(support_payload, Mapping):
                raise ValueError("support state diagnostics must be mappings")
            query_accuracy = _mean_old_accuracy(states[state])
            row = {
                "candidate_id": key[0],
                "scenario": key[1],
                "state": state,
                "support_risk_gain": _finite(support_payload.get("risk_gain"), "risk_gain"),
                "q90_feature_move": _finite(
                    support_payload.get("q90_feature_move"), "q90_feature_move"
                ),
                "query_mean_old_acc": query_accuracy,
                "query_gain_pp": float((query_accuracy - baseline_accuracy) * 100.0),
            }
            rows.append(row)
            state_names.add(state)

    nonbaseline = [row for row in rows if row["state"] != "DA0_REG0"]
    support_query = _spearman(
        [row["support_risk_gain"] for row in nonbaseline],
        [row["query_gain_pp"] for row in nonbaseline],
    )
    move_query = _spearman(
        [row["q90_feature_move"] for row in nonbaseline],
        [row["query_gain_pp"] for row in nonbaseline],
    )
    p0_stop = support_query is not None and support_query < 0.2
    return {
        "schema": "cvs.slow_fast.response_surface.v1",
        "state_count": len(state_names),
        "row_count": len(rows),
        "rows": rows,
        "spearman_support_query": support_query,
        "spearman_move_query": move_query,
        "p0_stop_signal": bool(p0_stop),
        "p0_stop_reason": "WEAK_SUPPORT_QUERY_RANK_ASSOCIATION_LT_0P2" if p0_stop else None,
        "truth_last_selection_reused_for_adaptation": False,
    }


__all__ = ["build_shadow_response_surface"]
