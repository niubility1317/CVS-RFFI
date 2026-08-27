from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def shared_triplet_loss(
    encoder: nn.Module,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Run all three triplet inputs through one shared encoder instance."""
    anchor_embedding = encoder(anchor)
    positive_embedding = encoder(positive)
    negative_embedding = encoder(negative)
    positive_distance = torch.linalg.vector_norm(anchor_embedding - positive_embedding, dim=1)
    negative_distance = torch.linalg.vector_norm(anchor_embedding - negative_embedding, dim=1)
    return F.relu(positive_distance - negative_distance + margin).mean()


def build_sgd_optimizer(encoder: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)
