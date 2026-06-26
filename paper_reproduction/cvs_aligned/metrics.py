from __future__ import annotations

import math
from typing import Iterable

import torch


def _as_1d_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().reshape(-1).float()


def _as_1d_long(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().reshape(-1).long()


def _safe_rate(numer: torch.Tensor, denom: torch.Tensor) -> float:
    d = int(denom.sum().item())
    if d <= 0:
        return float("nan")
    return float((numer & denom).sum().item() / d)


def _auroc(binary_positive: torch.Tensor, scores: torch.Tensor) -> float:
    y = binary_positive.bool()
    s = scores.float()
    n_pos = int(y.sum().item())
    n_neg = int((~y).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[y].sum().item()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fpr_at_tpr(binary_positive: torch.Tensor, scores: torch.Tensor, *, target_tpr: float = 0.95) -> float:
    y = binary_positive.bool()
    s = scores.float()
    n_pos = int(y.sum().item())
    n_neg = int((~y).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    thresholds = torch.unique(s, sorted=True, return_inverse=False)
    best = 1.0
    for threshold in thresholds:
        pred_pos = s >= threshold
        tpr = float((pred_pos & y).sum().item() / n_pos)
        fpr = float((pred_pos & (~y)).sum().item() / n_neg)
        if tpr >= target_tpr:
            best = min(best, fpr)
    return float(best)


def compute_cvs_stage2_metrics(
    *,
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    unknown_scores: torch.Tensor,
    old_labels: Iterable[int],
    new_labels: Iterable[int] | None = None,
) -> dict[str, float | bool | str]:
    true = _as_1d_long(true_labels)
    pred = _as_1d_long(predicted_labels)
    scores = _as_1d_float(unknown_scores)
    if true.numel() != pred.numel() or true.numel() != scores.numel():
        raise ValueError("true_labels, predicted_labels, and unknown_scores must have the same length")

    old = {int(v) for v in old_labels}
    new = {int(v) for v in (new_labels or set())}
    accepted = pred >= 0
    unknown_mask = true == -1
    old_mask = torch.tensor([int(v) in old for v in true.tolist()], dtype=torch.bool)
    new_mask = torch.tensor([int(v) in new for v in true.tolist()], dtype=torch.bool)
    pred_old = torch.tensor([int(v) in old for v in pred.tolist()], dtype=torch.bool)
    pred_new = torch.tensor([int(v) in new for v in pred.tolist()], dtype=torch.bool)
    correct = pred == true

    metrics: dict[str, float | bool | str] = {
        "target_old_full_acc": _safe_rate(correct, old_mask),
        "old_acc": _safe_rate(correct, old_mask),
        "target_old_coverage": _safe_rate(accepted, old_mask),
        "unknown_false_accept_rate": _safe_rate(accepted, unknown_mask),
        "unknown_FAR": _safe_rate(accepted, unknown_mask),
        "unknown_to_seen_new_rate": _safe_rate(pred_new, unknown_mask),
        "unknown_to_old_rate": _safe_rate(pred_old, unknown_mask),
        "old_to_seen_new_rate": _safe_rate(pred_new, old_mask),
        "old_reject_rate": _safe_rate(~accepted, old_mask),
        "AUROC": _auroc(unknown_mask, scores),
        "FPR95": _fpr_at_tpr(unknown_mask, scores, target_tpr=0.95),
        "stage2_seen_new_identity_evaluated": bool(len(new) > 0 and int(new_mask.sum().item()) > 0),
        "unknown_score_kind": "negative_max_similarity",
    }
    old_accepted = old_mask & accepted
    metrics["target_old_accepted_acc"] = _safe_rate(correct, old_accepted)

    if bool(metrics["stage2_seen_new_identity_evaluated"]):
        new_acc = _safe_rate(correct, new_mask)
        old_acc = float(metrics["old_acc"])
        metrics["seen_new_acc"] = new_acc
        metrics["seen_new_to_old_rate"] = _safe_rate(pred_old, new_mask)
        metrics["seen_new_reject_rate"] = _safe_rate(~accepted, new_mask)
        if math.isfinite(old_acc) and math.isfinite(new_acc) and (old_acc + new_acc) > 0.0:
            metrics["H_old_new"] = float(2.0 * old_acc * new_acc / (old_acc + new_acc))
    return metrics
