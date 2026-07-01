from collections import Counter
from typing import Any, Iterable, Mapping

import torch


def summarize_gate_decisions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = max(1, len(rows))
    decisions = [str(r.get("decision", "")) for r in rows]
    counts = Counter(decisions)
    reject_counts = {k: v for k, v in counts.items() if k.startswith("REJECT")}
    return {
        "count": len(rows),
        "known_core_accept": counts.get("ACCEPT_KNOWN_CORE", 0) / n,
        "known_tail_review": counts.get("REVIEW_KNOWN_TAIL", 0) / n,
        "known_auto_accept": (counts.get("ACCEPT_KNOWN_CORE", 0) + counts.get("ACCEPT_KNOWN_TAIL_STRICT", 0)) / n,
        "reject_rate": sum(reject_counts.values()) / n,
        "reject_reason_counts": dict(reject_counts),
        "decision_counts": dict(counts),
    }


def _auc_binary(y_true: torch.Tensor, scores: torch.Tensor) -> float | None:
    y = y_true.bool().view(-1)
    s = scores.float().view(-1)
    pos = s[y]
    neg = s[~y]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    comp = (pos.view(-1, 1) > neg.view(1, -1)).float()
    ties = (pos.view(-1, 1) == neg.view(1, -1)).float() * 0.5
    return float((comp + ties).mean().item())


def _fpr95(y_unknown: torch.Tensor, scores: torch.Tensor) -> float | None:
    y = y_unknown.bool().view(-1)
    s = scores.float().view(-1)
    if int(y.sum().item()) == 0 or int((~y).sum().item()) == 0:
        return None
    thresholds = torch.unique(s).sort(descending=True).values
    best = None
    for th in thresholds:
        pred_unknown = s >= th
        tpr = (pred_unknown & y).sum().float() / y.sum().float().clamp_min(1)
        if float(tpr.item()) >= 0.95:
            fpr = (pred_unknown & ~y).sum().float() / (~y).sum().float().clamp_min(1)
            best = float(fpr.item()) if best is None else min(best, float(fpr.item()))
    return best


def binary_reject_metrics(y_unknown: torch.Tensor, reject_scores: torch.Tensor, accepted: torch.Tensor) -> dict[str, Any]:
    y = y_unknown.bool().view(-1)
    scores = reject_scores.float().view(-1)
    acc = accepted.bool().view(-1)
    if y.numel() == 0 or int(y.sum().item()) == 0:
        return {"unknown_FAR": None, "unknown_reject_rate": None, "FPR95": None, "AUROC_unknown": None}
    far = float((acc & y).sum().float().div(y.sum().float().clamp_min(1)).item())
    return {
        "unknown_FAR": far,
        "unknown_reject_rate": 1.0 - far,
        "FPR95": _fpr95(y, scores),
        "AUROC_unknown": _auc_binary(y, scores),
    }

