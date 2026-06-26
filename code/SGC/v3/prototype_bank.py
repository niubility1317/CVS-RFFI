from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrototypeBuildConfig:
    num_classes: int
    prototypes_per_class: int = 2
    eps: float = 1e-6


class PrototypeBank(nn.Module):
    def __init__(self, prototypes: torch.Tensor, radius: torch.Tensor | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        if prototypes.dim() != 3:
            raise ValueError("prototypes must be shaped [C, K, D].")
        self.eps = float(eps)
        self.register_buffer("prototypes", F.normalize(prototypes.float(), dim=-1, eps=self.eps))
        if radius is None:
            radius = torch.ones(prototypes.size(0), prototypes.size(1), dtype=torch.float32)
        self.register_buffer("radius", radius.float())

    @classmethod
    def from_features(
        cls,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        num_classes: int,
        prototypes_per_class: int = 2,
        eps: float = 1e-6,
    ) -> "PrototypeBank":
        z = F.normalize(features.float(), dim=-1, eps=eps)
        y = labels.long().view(-1).to(device=z.device)
        C, K, D = int(num_classes), int(prototypes_per_class), z.size(-1)
        protos = z.new_zeros(C, K, D)
        radius = z.new_ones(C, K)
        for cls_idx in range(C):
            members = z[y == cls_idx]
            if members.numel() == 0:
                protos[cls_idx] = F.one_hot(torch.tensor(cls_idx % D, device=z.device), D).float().view(1, D).repeat(K, 1)
                continue
            chosen = [members[0]]
            while len(chosen) < K:
                current = torch.stack(chosen, dim=0)
                dist = 1.0 - torch.matmul(members, current.t()).max(dim=-1).values
                chosen.append(members[dist.argmax()])
            class_protos = F.normalize(torch.stack(chosen[:K], dim=0), dim=-1, eps=eps)
            protos[cls_idx] = class_protos
            sim = torch.matmul(members, class_protos.t()).max(dim=-1).values
            radius[cls_idx] = (1.0 - sim).mean().clamp_min(eps)
        return cls(protos.detach().cpu(), radius.detach().cpu(), eps=eps).to(device=features.device)

    def class_scores(self, features: torch.Tensor) -> torch.Tensor:
        z = F.normalize(features.float(), dim=-1, eps=self.eps)
        p = F.normalize(self.prototypes.to(device=z.device, dtype=z.dtype), dim=-1, eps=self.eps)
        return torch.einsum("bd,ckd->bck", z, p).max(dim=-1).values

    def prototype_distance(self, features: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.class_scores(features).max(dim=-1).values

    def pull_push_loss(self, features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1, margin: float = 0.1) -> torch.Tensor:
        scores = self.class_scores(features)
        y = labels.long().view(-1).to(device=scores.device)
        ce = F.cross_entropy(scores / float(temperature), y)
        pos = scores.gather(1, y.view(-1, 1)).squeeze(1)
        neg = scores.masked_fill(F.one_hot(y, scores.size(1)).bool(), -1e4).max(dim=-1).values
        return ce + F.relu(float(margin) + neg - pos).mean()
