from __future__ import annotations

from typing import Mapping

import torch


TensorDict = dict[str, torch.Tensor]


def _scalar(value: float, ref: torch.Tensor | None = None) -> torch.Tensor:
    if ref is None:
        return torch.tensor(float(value))
    return ref.new_tensor(float(value))


def compute_flip_metrics(base_logits: torch.Tensor, final_logits: torch.Tensor, labels: torch.Tensor) -> TensorDict:
    y = labels.long().view(-1).to(device=final_logits.device)
    base_correct = base_logits.argmax(dim=-1).eq(y)
    final_correct = final_logits.argmax(dim=-1).eq(y)
    wrong_to_right = (~base_correct) & final_correct
    right_to_wrong = base_correct & (~final_correct)
    rank_change = base_logits.argmax(dim=-1).ne(final_logits.argmax(dim=-1))
    ref = final_logits
    return {
        "wrong_to_right_count": _scalar(float(wrong_to_right.sum()), ref),
        "right_to_wrong_count": _scalar(float(right_to_wrong.sum()), ref),
        "wrong_to_right_rate": wrong_to_right.float().mean(),
        "right_to_wrong_rate": right_to_wrong.float().mean(),
        "net_gain": wrong_to_right.float().mean() - right_to_wrong.float().mean(),
        "top1_flip_rate": rank_change.float().mean(),
    }


def compute_pseudo_label_metrics(prob: torch.Tensor, labels: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> TensorDict:
    conf, pseudo = prob.max(dim=-1)
    if mask is None:
        mask = torch.ones_like(conf, dtype=torch.bool)
    out: TensorDict = {
        "pseudo_label_coverage": mask.float().mean(),
        "pseudo_confidence_mean": conf[mask].mean() if mask.any() else conf.new_tensor(0.0),
        "accepted_samples": mask.float().sum(),
    }
    if labels is not None:
        y = labels.to(device=prob.device).long().view(-1)
        out["pseudo_label_precision"] = pseudo[mask].eq(y[mask]).float().mean() if mask.any() else conf.new_tensor(0.0)
        out["pseudo_wrong_high_conf_rate"] = ((~pseudo.eq(y)) & mask).float().mean()
    hist = torch.bincount(pseudo[mask], minlength=prob.size(-1)).float() if mask.any() else prob.new_zeros(prob.size(-1))
    p = hist / hist.sum().clamp_min(1.0)
    out["class_balance_entropy"] = -(p * (p + 1e-8).log()).sum()
    return out


def check_constrained_improvement(
    current: Mapping[str, float],
    baseline: Mapping[str, float],
    safety: Mapping[str, float],
    *,
    metric: str = "clear_leo_tx",
    min_gain: float = 0.3,
    clean_drop_max: float = 0.5,
    normal_drop_max: float = 0.5,
    gate_clean_max: float = 0.05,
) -> bool:
    gain = float(current.get(metric, float("-inf"))) - float(baseline.get(metric, float("-inf")))
    return (
        gain >= float(min_gain)
        and float(safety.get("clean_drop", 999.0)) <= float(clean_drop_max)
        and float(safety.get("normal_drop", 999.0)) <= float(normal_drop_max)
        and float(safety.get("gate_clean_mean", 999.0)) <= float(gate_clean_max)
        and float(safety.get("net_gain_sat", -999.0)) > 0.0
    )
