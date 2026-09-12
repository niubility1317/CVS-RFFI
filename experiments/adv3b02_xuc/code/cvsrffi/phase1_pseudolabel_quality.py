"""Truth-blind Phase1 pseudo-label artifacts and isolated source-side scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

import torch


def _as_list(values: Iterable[object]) -> List[object]:
    return list(values)


def build_truth_blind_records(
    *,
    physical_sample_ids: Sequence[object],
    receivers: Sequence[object],
    days: Sequence[object],
    routes: Sequence[str],
    pseudo_labels: Sequence[int],
    candidate_masks: torch.Tensor,
    fused_probabilities: torch.Tensor,
    p_correct: Sequence[float],
    p_set_safe: Sequence[float],
    sample_weights: Sequence[float],
) -> List[Dict[str, object]]:
    """Build the artifact without accepting a truth-label argument."""

    columns = [
        _as_list(physical_sample_ids),
        _as_list(receivers),
        _as_list(days),
        _as_list(routes),
        _as_list(pseudo_labels),
        _as_list(p_correct),
        _as_list(p_set_safe),
        _as_list(sample_weights),
    ]
    count = len(columns[0])
    if any(len(column) != count for column in columns):
        raise ValueError("truth-blind artifact columns must have identical lengths")
    if candidate_masks.ndim != 2 or fused_probabilities.ndim != 2:
        raise ValueError("candidate masks and fused probabilities must be rank-2")
    if tuple(candidate_masks.shape) != tuple(fused_probabilities.shape):
        raise ValueError("candidate masks and fused probabilities must align")
    if int(candidate_masks.shape[0]) != count:
        raise ValueError("artifact tensor rows must match physical sample ids")
    normalized_ids = [str(value) for value in columns[0]]
    if len(set(normalized_ids)) != count:
        raise ValueError("physical_sample_id must be unique")

    probabilities = fused_probabilities.detach().float().cpu()
    masks = candidate_masks.detach().bool().cpu()
    records: List[Dict[str, object]] = []
    for index in range(count):
        candidate_set = torch.where(masks[index])[0].tolist()
        conditional = torch.zeros_like(probabilities[index])
        if candidate_set:
            mass = probabilities[index, candidate_set].sum()
            if bool(torch.isfinite(mass)) and float(mass) > 0.0:
                conditional[candidate_set] = probabilities[index, candidate_set] / mass
        records.append(
            {
                "physical_sample_id": normalized_ids[index],
                "receiver": str(columns[1][index]),
                "day": str(columns[2][index]),
                "route": str(columns[3][index]),
                "top1_pseudo_label": int(columns[4][index]),
                "candidate_set": [int(value) for value in candidate_set],
                "p_correct": round(float(columns[5][index]), 6),
                "p_set_safe": round(float(columns[6][index]), 6),
                "partial_conditional_distribution": [
                    round(float(value), 6) for value in conditional.tolist()
                ],
                "sample_weight": round(float(columns[7][index]), 6),
            }
        )
    return records


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _nearest_rank_p95(values: Sequence[int]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(int(value) for value in values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _aurc(correct: Sequence[bool], confidence: Sequence[float]) -> float:
    if not correct:
        return float("nan")
    order = sorted(range(len(correct)), key=lambda index: float(confidence[index]), reverse=True)
    cumulative_errors = 0
    risks = []
    for rank, index in enumerate(order, start=1):
        cumulative_errors += 0 if bool(correct[index]) else 1
        risks.append(cumulative_errors / float(rank))
    return _safe_mean(risks)


def _score_group(rows: Sequence[Mapping[str, object]], truths: Mapping[str, int]) -> Dict[str, float]:
    hard = [row for row in rows if row["route"] == "H"]
    partial = [row for row in rows if row["route"] == "P"]
    hard_correct = [
        int(row["top1_pseudo_label"]) == int(truths[str(row["physical_sample_id"])])
        for row in hard
    ]
    set_safe = [
        int(truths[str(row["physical_sample_id"])]) in set(int(v) for v in row["candidate_set"])
        for row in partial
    ]
    rank_correct = []
    for row, safe in zip(partial, set_safe):
        if not safe:
            continue
        distribution = list(row["partial_conditional_distribution"])
        rank_correct.append(
            int(max(range(len(distribution)), key=lambda index: float(distribution[index])))
            == int(truths[str(row["physical_sample_id"])])
        )
    sizes = [len(row["candidate_set"]) for row in partial]
    return {
        "h_precision": _safe_mean([float(value) for value in hard_correct]),
        "h_coverage": float(len(hard)) / float(max(1, len(rows))),
        "h_aurc": _aurc(hard_correct, [float(row["p_correct"]) for row in hard]),
        "p_set_coverage": _safe_mean([float(value) for value in set_safe]),
        "p_mean_set_size": _safe_mean([float(value) for value in sizes]),
        "p_p95_set_size": _nearest_rank_p95(sizes),
        "p_rank_accuracy_when_set_safe": _safe_mean([float(value) for value in rank_correct]),
    }


def _grouped(
    rows: Sequence[Mapping[str, object]], truths: Mapping[str, int], key_fn
) -> Dict[str, Dict[str, float]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row, truths))].append(row)
    return {key: _score_group(value, truths) for key, value in sorted(groups.items())}


def _worst_group(groups: Mapping[str, Mapping[str, float]], metric: str, label: str) -> Dict[str, object]:
    eligible = [
        (key, float(values[metric]))
        for key, values in groups.items()
        if math.isfinite(float(values[metric]))
    ]
    if not eligible:
        return {label: None, metric: float("nan")}
    key, value = min(eligible, key=lambda item: (item[1], item[0]))
    return {label: key, metric: value}


def score_truth_blind_records(
    records: Sequence[Mapping[str, object]], truth_by_physical_id: Mapping[object, int]
) -> Dict[str, object]:
    """Join truth only after artifact closure and compute source-side quality."""

    ids = [str(row["physical_sample_id"]) for row in records]
    truths = {str(key): int(value) for key, value in truth_by_physical_id.items()}
    if len(set(ids)) != len(ids) or set(ids) != set(truths):
        raise ValueError("truth ids must exactly match artifact ids")
    rows = list(records)
    by_receiver = _grouped(rows, truths, lambda row, _truth: row["receiver"])
    by_receiver_day = _grouped(
        rows, truths, lambda row, _truth: f"{row['receiver']}|{row['day']}"
    )
    by_class = _grouped(
        rows, truths, lambda row, truth: truth[str(row["physical_sample_id"])]
    )
    by_class_receiver = _grouped(
        rows,
        truths,
        lambda row, truth: f"{truth[str(row['physical_sample_id'])]}|{row['receiver']}",
    )
    return {
        "counts": {
            "all": len(rows),
            "H": sum(row["route"] == "H" for row in rows),
            "P": sum(row["route"] == "P" for row in rows),
            "R": sum(row["route"] == "R" for row in rows),
        },
        "overall": _score_group(rows, truths),
        "by_class": by_class,
        "by_receiver": by_receiver,
        "by_receiver_day": by_receiver_day,
        "by_class_receiver": by_class_receiver,
        "worst_receiver": _worst_group(by_receiver, "p_set_coverage", "receiver"),
        "worst_class_receiver": _worst_group(
            by_class_receiver, "p_set_coverage", "class_receiver"
        ),
    }


__all__ = ["build_truth_blind_records", "score_truth_blind_records"]
