from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import torch


@dataclass(frozen=True)
class EpisodeBatch:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    support_ids: Sequence[Hashable] | None = None
    query_ids: Sequence[Hashable] | None = None
    support_receivers: Sequence[Hashable] | None = None
    query_receivers: Sequence[Hashable] | None = None
    source_receivers: Sequence[Hashable] | None = None
    target_receiver: Hashable | None = None


def _labels_and_counts(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D tensor")
    return torch.unique(labels.detach().cpu(), sorted=True, return_counts=True)


def validate_closed_set_episode(
    episode: EpisodeBatch,
    k_shot: int,
    *,
    n_way: int | None = None,
    require_receiver_metadata: bool = False,
) -> None:
    """Validate the closed-set target support/query episode used by both papers."""
    if k_shot <= 0:
        raise ValueError("K-shot must be a positive integer")
    if episode.support_x.shape[0] != episode.support_y.numel():
        raise ValueError("support_x and support_y length mismatch")
    if episode.query_x.shape[0] != episode.query_y.numel():
        raise ValueError("query_x and query_y length mismatch")

    class_ids, counts = _labels_and_counts(episode.support_y)
    if n_way is not None and int(class_ids.numel()) != n_way:
        raise ValueError(f"N-way mismatch: expected {n_way} classes")
    if not torch.all(counts == k_shot):
        raise ValueError(f"K-shot support mismatch: expected {k_shot} per class")

    query_labels = set(episode.query_y.detach().cpu().tolist())
    support_labels = set(class_ids.tolist())
    if not query_labels.issubset(support_labels):
        raise ValueError("query labels must be contained in support labels for closed-set evaluation")

    if episode.support_ids is not None or episode.query_ids is not None:
        if episode.support_ids is None or episode.query_ids is None:
            raise ValueError("support_ids and query_ids must be provided together")
        if len(episode.support_ids) != episode.support_y.numel():
            raise ValueError("support_ids length mismatch")
        if len(episode.query_ids) != episode.query_y.numel():
            raise ValueError("query_ids length mismatch")
        overlap = set(episode.support_ids).intersection(set(episode.query_ids))
        if overlap:
            raise ValueError(f"support/query leakage detected: {sorted(map(str, overlap))[:3]}")

    if require_receiver_metadata:
        if episode.target_receiver is None:
            raise ValueError("target receiver metadata is required")
        if episode.support_receivers is None or episode.query_receivers is None:
            raise ValueError("support/query receiver metadata is required")
        if len(episode.support_receivers) != episode.support_y.numel():
            raise ValueError("support_receivers length mismatch")
        if len(episode.query_receivers) != episode.query_y.numel():
            raise ValueError("query_receivers length mismatch")
        target = episode.target_receiver
        support_receivers = set(episode.support_receivers)
        query_receivers = set(episode.query_receivers)
        if support_receivers != {target} or query_receivers != {target}:
            raise ValueError("support/query samples must come from the target receiver")
        if episode.source_receivers is None:
            raise ValueError("source receiver metadata is required")
        if target in set(episode.source_receivers):
            raise ValueError("source/target receiver sets must be disjoint")
