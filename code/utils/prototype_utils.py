from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _as_class_proto(prototypes: torch.Tensor) -> torch.Tensor:
    if prototypes.ndim == 2:
        return prototypes.unsqueeze(1)
    if prototypes.ndim == 3:
        return prototypes
    raise ValueError(f"prototypes must be [C, D] or [C, K, D], got {tuple(prototypes.shape)}.")


def class_proto_distances(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    proto = _as_class_proto(prototypes).to(device=features.device, dtype=features.dtype)
    feat = features.unsqueeze(1).unsqueeze(2)
    distances = (feat - proto.unsqueeze(0)).pow(2).sum(dim=-1)
    return distances.min(dim=2).values


def compute_proto_margin(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    distances = class_proto_distances(features, prototypes)
    if distances.size(1) <= 1:
        return torch.full((features.size(0),), float("inf"), device=features.device, dtype=features.dtype)
    top2 = distances.topk(k=2, dim=1, largest=False).values
    return top2[:, 1] - top2[:, 0]


def nearest_proto_class(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return class_proto_distances(features, prototypes).argmin(dim=1)


def prototype_pull_loss(features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    if features.numel() == 0:
        return features.sum() * 0.0
    proto = _as_class_proto(prototypes).to(device=features.device, dtype=features.dtype)
    selected = proto[labels]
    distances = (features.unsqueeze(1) - selected).pow(2).sum(dim=-1)
    return distances.min(dim=1).values.mean()


def classifier_weight_centers(classifier: torch.nn.Module) -> torch.Tensor:
    weight = getattr(classifier, "weight", None)
    if weight is None:
        raise ValueError("classifier does not expose a weight tensor.")
    return F.normalize(weight.detach(), dim=-1)
