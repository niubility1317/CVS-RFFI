from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class SourcePrototypeBank:
    """EMA source-domain class prototypes for lightweight adaptation losses."""

    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.9, device=None):
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.prototypes = torch.zeros(self.num_classes, self.feat_dim, device=device)
        self.initialized = torch.zeros(self.num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        z = z.detach()
        y = y.detach().long()
        for cls_idx in range(self.num_classes):
            mask = y == cls_idx
            if not mask.any():
                continue
            mean_k = z[mask].mean(dim=0)
            if not self.initialized[cls_idx]:
                self.prototypes[cls_idx] = mean_k
                self.initialized[cls_idx] = True
            else:
                self.prototypes[cls_idx] = (
                    self.momentum * self.prototypes[cls_idx]
                    + (1.0 - self.momentum) * mean_k
                )

    def get(self, y: torch.Tensor) -> torch.Tensor:
        return self.prototypes[y.long()]

    def alignment_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if z.numel() == 0:
            return self.prototypes.sum() * 0.0
        proto = self.get(labels).to(device=z.device, dtype=z.dtype)
        z_norm = F.normalize(z, dim=1)
        p_norm = F.normalize(proto, dim=1)
        return (1.0 - (z_norm * p_norm).sum(dim=1)).mean()


def feature_consistency_loss(
    feat_clean: torch.Tensor,
    feat_sgc: torch.Tensor,
    mode: str = "mse",
) -> torch.Tensor:
    feat_clean = feat_clean.detach()
    if str(mode).lower() == "cosine":
        return 1.0 - F.cosine_similarity(feat_clean, feat_sgc, dim=1).mean()
    return F.mse_loss(feat_sgc, feat_clean)


def pseudo_label_loss(
    logits: torch.Tensor,
    threshold: float = 0.85,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    prob = torch.softmax(logits, dim=1)
    conf, pseudo = prob.max(dim=1)
    mask = conf > float(threshold)
    conf_mean = float(conf.mean().detach().item())
    high_conf_ratio = float(mask.float().mean().detach().item())

    if not mask.any():
        return logits.sum() * 0.0, mask, conf_mean, high_conf_ratio

    return F.cross_entropy(logits[mask], pseudo[mask]), mask, conf_mean, high_conf_ratio


def consistency_regularization(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    prob1 = torch.softmax(logits1, dim=1)
    prob2 = torch.softmax(logits2, dim=1)
    return F.mse_loss(prob1, prob2)


def entropy_minimization(logits: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    log_prob = torch.log_softmax(logits, dim=1)
    ent = -(prob * log_prob).sum(dim=1)
    if mask is not None and mask.any():
        return ent[mask].mean()
    return ent.mean()


def residual_regularization(adapter_aux: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not adapter_aux:
        return torch.tensor(0.0, requires_grad=True)
    gamma = adapter_aux.get("residual_gamma")
    if torch.is_tensor(gamma):
        return gamma.mean()
    delta_rms = adapter_aux.get("adapter_delta_rms")
    if torch.is_tensor(delta_rms):
        return delta_rms.mean()
    ref = next((v for v in adapter_aux.values() if torch.is_tensor(v)), None)
    if ref is not None:
        return ref.sum() * 0.0
    return torch.tensor(0.0, requires_grad=True)
