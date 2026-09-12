from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PseudoLabelGateConfig:
    min_confidence: float = 0.85
    min_margin: float = 0.05
    max_uncertainty: float = 0.08
    require_prototype_agreement: bool = True
    class_quota: int = 0
    receiver_quota: int = 0


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    p = prob.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=1)


def select_pseudo_labels(
    teacher_logits: torch.Tensor,
    *,
    features: Optional[torch.Tensor] = None,
    class_prototypes: Optional[torch.Tensor] = None,
    uncertainty: Optional[torch.Tensor] = None,
    receiver_ids: Optional[torch.Tensor] = None,
    config: PseudoLabelGateConfig = PseudoLabelGateConfig(),
) -> Dict[str, torch.Tensor]:
    """Select high-quality source-unlabeled TX pseudo labels.

    This is intentionally framework-neutral: it consumes already-computed
    teacher logits/features and emits a mask plus audit metrics. The caller
    decides whether accepted samples enter TX CE, prototype pull, or satellite
    strong-view consistency.
    """
    if teacher_logits.ndim != 2:
        raise ValueError("teacher_logits must have shape [N, C]")
    logits = torch.nan_to_num(teacher_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    prob = F.softmax(logits, dim=1)
    top2 = torch.topk(prob, k=min(2, prob.size(1)), dim=1)
    pseudo_y = top2.indices[:, 0].long()
    confidence = top2.values[:, 0]
    if prob.size(1) > 1:
        margin = top2.values[:, 0] - top2.values[:, 1]
    else:
        margin = top2.values[:, 0]
    if uncertainty is None:
        uncertainty_t = _entropy(prob) / max(1.0, float(prob.size(1)))
    else:
        uncertainty_t = torch.nan_to_num(uncertainty.float().view(-1), nan=1.0, posinf=1.0, neginf=1.0)
    if uncertainty_t.numel() != logits.size(0):
        raise ValueError("uncertainty must have one value per sample")

    mask = (
        (confidence >= float(config.min_confidence))
        & (margin >= float(config.min_margin))
        & (uncertainty_t <= float(config.max_uncertainty))
    )

    proto_agree = torch.ones_like(mask, dtype=torch.bool)
    if features is not None and class_prototypes is not None:
        feat = F.normalize(torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
        proto = F.normalize(torch.nan_to_num(class_prototypes.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
        if feat.size(1) != proto.size(1):
            raise ValueError("features and class_prototypes must have the same feature dimension")
        proto_y = (feat @ proto.t()).argmax(dim=1).long()
        proto_agree = proto_y.eq(pseudo_y)
        if bool(config.require_prototype_agreement):
            mask = mask & proto_agree

    if int(config.class_quota) > 0:
        quota_mask = torch.zeros_like(mask)
        for cls in torch.unique(pseudo_y[mask]):
            idx = torch.nonzero(mask & pseudo_y.eq(cls), as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            score = confidence[idx] - uncertainty_t[idx]
            keep = idx[torch.argsort(score, descending=True)[: int(config.class_quota)]]
            quota_mask[keep] = True
        mask = mask & quota_mask

    if int(config.receiver_quota) > 0 and receiver_ids is not None:
        rx = receiver_ids.view(-1).to(pseudo_y.device)
        if rx.numel() != logits.size(0):
            raise ValueError("receiver_ids must have one value per sample")
        quota_mask = torch.zeros_like(mask)
        for cls in torch.unique(pseudo_y[mask]):
            for rid in torch.unique(rx[mask & pseudo_y.eq(cls)]):
                idx = torch.nonzero(mask & pseudo_y.eq(cls) & rx.eq(rid), as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                score = confidence[idx] - uncertainty_t[idx]
                keep = idx[torch.argsort(score, descending=True)[: int(config.receiver_quota)]]
                quota_mask[keep] = True
        mask = mask & quota_mask

    return {
        "mask": mask,
        "pseudo_y": pseudo_y,
        "confidence": confidence,
        "margin": margin,
        "uncertainty": uncertainty_t,
        "prototype_agreement": proto_agree,
        "accepted_count": mask.sum(),
        "coverage": mask.float().mean(),
        "proto_agreement_rate": proto_agree.float().mean(),
    }
