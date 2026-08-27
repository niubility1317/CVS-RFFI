from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_positive_negative_indices(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mine one hardest positive and nearest negative for every anchor."""
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings must be [batch, dim] and labels must be [batch]")
    distances = torch.cdist(embeddings, embeddings, p=2)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    if not bool(same.any(dim=1).all()) or not bool(different.any(dim=1).all()):
        raise ValueError("each anchor requires a positive and a negative example")
    positive = distances.masked_fill(~same, float("-inf")).argmax(dim=1)
    negative = distances.masked_fill(~different, float("inf")).argmin(dim=1)
    return positive, negative


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Mean max(||A-P||-||A-N||+margin,0) after batch-hard mining."""
    positive, negative = hard_positive_negative_indices(embeddings, labels)
    anchors = torch.arange(embeddings.shape[0], device=embeddings.device)
    positive_distance = F.pairwise_distance(embeddings[anchors], embeddings[positive], p=2)
    negative_distance = F.pairwise_distance(embeddings[anchors], embeddings[negative], p=2)
    return F.relu(positive_distance - negative_distance + margin).mean()
