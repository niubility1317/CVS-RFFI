from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PseudoLabelFilterConfig:
    conf_threshold: float = 0.95
    margin_threshold: float = 0.5
    view_agreement_threshold: float = 0.8


def probability_margin(prob: torch.Tensor) -> torch.Tensor:
    top2 = prob.topk(min(2, prob.size(-1)), dim=-1).values
    if top2.size(-1) == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def build_pseudo_label_mask(
    prob: torch.Tensor,
    *,
    view_agreement: torch.Tensor | None = None,
    prototype_distance: torch.Tensor | None = None,
    prototype_threshold: float | None = None,
    cfg: PseudoLabelFilterConfig | None = None,
) -> torch.Tensor:
    cfg = cfg or PseudoLabelFilterConfig()
    conf = prob.max(dim=-1).values
    margin = probability_margin(prob)
    mask = (conf >= float(cfg.conf_threshold)) & (margin >= float(cfg.margin_threshold))
    if view_agreement is not None:
        mask = mask & (view_agreement.to(device=prob.device).view(-1) >= float(cfg.view_agreement_threshold))
    if prototype_distance is not None and prototype_threshold is not None:
        mask = mask & (prototype_distance.to(device=prob.device).view(-1) <= float(prototype_threshold))
    return mask


class TargetMemoryBank:
    def __init__(self, size: int, feature_dim: int, num_classes: int, device: torch.device | str = "cpu") -> None:
        self.size = int(size)
        self.z_ema = torch.zeros(self.size, int(feature_dim), device=device)
        self.p_ema = torch.zeros(self.size, int(num_classes), device=device)
        self.pseudo_y = torch.full((self.size,), -1, dtype=torch.long, device=device)
        self.confidence = torch.zeros(self.size, device=device)
        self.class_count = torch.zeros(int(num_classes), device=device)
        self.last_update_step = torch.full((self.size,), -1, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, index: torch.Tensor, z: torch.Tensor, prob: torch.Tensor, step: int, ema_decay: float = 0.99) -> None:
        idx = index.long().to(device=self.z_ema.device).view(-1)
        z = z.detach().to(device=self.z_ema.device)
        prob = prob.detach().to(device=self.p_ema.device)
        decay = float(ema_decay)
        self.z_ema[idx] = decay * self.z_ema[idx] + (1.0 - decay) * z
        self.p_ema[idx] = decay * self.p_ema[idx] + (1.0 - decay) * prob
        conf, pseudo = self.p_ema[idx].max(dim=-1)
        self.pseudo_y[idx] = pseudo
        self.confidence[idx] = conf
        self.last_update_step[idx] = int(step)
        self.class_count.zero_()
        valid = self.pseudo_y >= 0
        if valid.any():
            self.class_count += torch.bincount(self.pseudo_y[valid], minlength=self.p_ema.size(-1)).to(self.class_count)
