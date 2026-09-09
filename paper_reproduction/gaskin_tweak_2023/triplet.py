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
    positive_distance = torch.linalg.vector_norm(embeddings[anchors] - embeddings[positive], dim=1)
    negative_distance = torch.linalg.vector_norm(embeddings[anchors] - embeddings[negative], dim=1)
    return F.relu(positive_distance - negative_distance + margin).mean()


def random_triplet_indices(
    labels: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one positive and one negative candidate per anchor for online mining."""
    if labels.ndim != 1 or labels.numel() < 3:
        raise ValueError("labels must describe at least three examples")
    anchors = torch.arange(labels.numel(), device=labels.device)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    same_counts, different_counts = same.sum(dim=1), different.sum(dim=1)
    if not bool(same_counts.all()) or not bool(different_counts.all()):
        raise ValueError("each anchor requires a positive and a negative example")

    def choose(mask: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        # Convert a per-row uniform rank into a column index without host synchronization.
        ranks = torch.floor(torch.rand(labels.numel(), generator=generator, device=labels.device) * counts).to(torch.long)
        return (mask.to(torch.long).cumsum(dim=1) > ranks[:, None]).to(torch.long).argmax(dim=1)

    return anchors, choose(same, same_counts), choose(different, different_counts)


def margin_violating_triplet_loss(
    embeddings: torch.Tensor,
    *,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Average only sampled triplets whose loss violates the specified margin."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [batch, dim]")
    if any(index.ndim != 1 for index in (anchors, positives, negatives)) or not (anchors.shape == positives.shape == negatives.shape):
        raise ValueError("anchor, positive, and negative indices must be equally shaped vectors")
    if not anchors.numel() or min(int(index.min()) for index in (anchors, positives, negatives)) < 0 or max(int(index.max()) for index in (anchors, positives, negatives)) >= embeddings.shape[0]:
        raise ValueError("triplet indices must be nonempty and within the embedding batch")
    positive_distance = torch.linalg.vector_norm(embeddings[anchors] - embeddings[positives], dim=1)
    negative_distance = torch.linalg.vector_norm(embeddings[anchors] - embeddings[negatives], dim=1)
    losses = F.relu(positive_distance - negative_distance + margin)
    violating = losses > torch.finfo(losses.dtype).eps
    return losses[violating].mean() if bool(violating.any()) else embeddings.sum() * 0


def strict_hard_triplet_loss(
    embeddings: torch.Tensor,
    *,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    margin: float = 0.1,
) -> tuple[torch.Tensor, bool]:
    """Use only Tweak IV-A hard triplets, where the negative is closer than the positive."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [batch, dim]")
    if any(index.ndim != 1 for index in (anchors, positives, negatives)) or not (anchors.shape == positives.shape == negatives.shape):
        raise ValueError("anchor, positive, and negative indices must be equally shaped vectors")
    if not anchors.numel() or min(int(index.min()) for index in (anchors, positives, negatives)) < 0 or max(int(index.max()) for index in (anchors, positives, negatives)) >= embeddings.shape[0]:
        raise ValueError("triplet indices must be nonempty and within the embedding batch")
    positive_distance = torch.linalg.vector_norm(embeddings[anchors] - embeddings[positives], dim=1)
    negative_distance = torch.linalg.vector_norm(embeddings[anchors] - embeddings[negatives], dim=1)
    hard = negative_distance < positive_distance
    if not bool(hard.any()):
        return embeddings.sum() * 0, False
    return F.relu(positive_distance[hard] - negative_distance[hard] + margin).mean(), True
