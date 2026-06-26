from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_prototypes(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted class ids and class-mean prototypes."""
    if features.ndim != 2:
        raise ValueError("features must have shape [num_samples, feature_dim]")
    if labels.ndim != 1 or labels.numel() != features.shape[0]:
        raise ValueError("labels must have shape [num_samples]")

    class_ids = torch.unique(labels, sorted=True)
    prototypes = []
    for class_id in class_ids:
        mask = labels == class_id
        if not bool(mask.any()):
            raise ValueError(f"class {int(class_id)} has no support samples")
        prototypes.append(features[mask].mean(dim=0))
    return class_ids, torch.stack(prototypes, dim=0)


def distance_logits(
    query_features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    metric: str = "euclidean",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert query-to-prototype distances to logits for softmax classification."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if query_features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("query_features and prototypes must be 2-D tensors")
    if query_features.shape[1] != prototypes.shape[1]:
        raise ValueError("feature dimensions must match")

    if metric == "euclidean":
        distances = torch.cdist(query_features, prototypes, p=2)
        return -distances / temperature
    if metric == "sqeuclidean":
        distances = torch.cdist(query_features, prototypes, p=2).pow(2)
        return -distances / temperature
    if metric == "cosine":
        query_norm = F.normalize(query_features, dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        return (query_norm @ proto_norm.t()) / temperature
    raise ValueError(f"unsupported metric: {metric}")


def prototypical_nll(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    metric: str = "euclidean",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the target-query NLL/CE objective for a prototypical episode."""
    class_ids, prototypes = compute_prototypes(support_features, support_labels)
    logits = distance_logits(query_features, prototypes, metric=metric, temperature=temperature)
    target = torch.empty_like(query_labels)
    for compact_index, class_id in enumerate(class_ids):
        target[query_labels == class_id] = compact_index
    if not torch.isin(query_labels, class_ids).all():
        raise ValueError("all query labels must appear in support labels")
    loss = F.cross_entropy(logits, target)
    pred = class_ids[logits.argmax(dim=1)]
    return loss, pred
