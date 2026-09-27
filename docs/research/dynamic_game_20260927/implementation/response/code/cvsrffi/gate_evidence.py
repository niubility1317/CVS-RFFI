from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateEvidenceState(nn.Module):
    """Checkpointable support-training evidence for the five physical branches."""

    def __init__(self, branch_names: Sequence[str], momentum: float = 0.95) -> None:
        super().__init__()
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be in [0,1)")
        self.branch_names = tuple(branch_names)
        if not self.branch_names:
            raise ValueError("branch_names cannot be empty")
        self.momentum = float(momentum)
        self.register_buffer(
            "discriminability_ema", torch.zeros(len(self.branch_names), dtype=torch.float32)
        )
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("discriminability_frozen", torch.zeros((), dtype=torch.bool))

    def freeze_discriminability(self, frozen: bool = True) -> None:
        self.discriminability_frozen.fill_(bool(frozen))

    @staticmethod
    def _fisher_ratio(embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        value = torch.nan_to_num(embedding.detach().float())
        labels = labels.detach().reshape(-1).to(device=value.device)
        if value.size(0) != labels.numel():
            raise ValueError("embedding batch and labels must match")
        global_mean = value.mean(dim=0)
        between = value.new_zeros(())
        within = value.new_zeros(())
        for class_label in torch.unique(labels, sorted=True):
            class_value = value[labels == class_label]
            class_mean = class_value.mean(dim=0)
            between = between + class_value.size(0) * (class_mean - global_mean).square().sum()
            within = within + (class_value - class_mean).square().sum()
        return (between / (between + within + 1e-6)).clamp(0.0, 1.0)

    @torch.no_grad()
    def update_discriminability(
        self, embeddings: Mapping[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        if bool(self.discriminability_frozen.item()):
            return self.discriminability_ema
        missing = [name for name in self.branch_names if name not in embeddings]
        if missing:
            raise KeyError(f"missing branch embeddings: {missing}")
        current = torch.stack(
            [self._fisher_ratio(embeddings[name], labels) for name in self.branch_names]
        ).to(self.discriminability_ema)
        if int(self.update_count.item()) == 0 or self.momentum == 0.0:
            self.discriminability_ema.copy_(current)
        else:
            self.discriminability_ema.mul_(self.momentum).add_(
                current, alpha=1.0 - self.momentum
            )
        self.update_count.add_(1)
        return self.discriminability_ema

    def paired_stability(
        self,
        clean_embeddings: Mapping[str, torch.Tensor],
        leo_embeddings: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        scores = []
        for name in self.branch_names:
            clean = torch.nan_to_num(clean_embeddings[name].float())
            leo = torch.nan_to_num(leo_embeddings[name].float())
            if clean.shape != leo.shape:
                raise ValueError(f"paired embeddings differ for branch {name}")
            similarity = F.cosine_similarity(clean, leo, dim=-1, eps=1e-6)
            scores.append(((similarity + 1.0) * 0.5).clamp(0.0, 1.0))
        return torch.stack(scores, dim=-1)

    @staticmethod
    def _sanitize(value: torch.Tensor, nonfinite: float) -> torch.Tensor:
        return torch.nan_to_num(
            value.float(), nan=nonfinite, posinf=nonfinite, neginf=nonfinite
        ).clamp(0.0, 1.0)

    def compose(
        self,
        *,
        identifiability: torch.Tensor,
        stability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if identifiability.shape != stability.shape or stability.shape != uncertainty.shape:
            raise ValueError("I, S and U must share shape [B,branch_count]")
        if identifiability.size(-1) != len(self.branch_names):
            raise ValueError("evidence branch count does not match state")
        d = self.discriminability_ema.to(identifiability).unsqueeze(0)
        d = d.expand(identifiability.size(0), -1)
        return {
            "I": self._sanitize(identifiability, 0.0),
            "D": self._sanitize(d, 0.0),
            "S": self._sanitize(stability, 0.0),
            "U": self._sanitize(uncertainty, 1.0),
        }
