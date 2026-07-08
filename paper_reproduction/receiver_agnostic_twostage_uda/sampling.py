from __future__ import annotations

import torch


def rank_uncertain_samples(
    logits: torch.Tensor,
    *,
    strategy: str,
    k: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Rank target samples for the paper's uncertainty-sampling fine-tuning step."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, num_classes]")
    if k is not None and k < 0:
        raise ValueError("k must be non-negative")
    probs = torch.softmax(logits, dim=1)
    strategy = strategy.lower().strip()
    if strategy == "entropy":
        score = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        order = torch.argsort(score, descending=True, stable=True)
    elif strategy == "margin":
        top2 = torch.topk(probs, k=2, dim=1).values
        score = top2[:, 0] - top2[:, 1]
        order = torch.argsort(score, descending=False, stable=True)
    elif strategy == "least_confidence":
        score = probs.max(dim=1).values
        order = torch.argsort(score, descending=False, stable=True)
    elif strategy == "random":
        generator = None
        if seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(int(seed))
        order = torch.randperm(logits.shape[0], generator=generator, device=logits.device)
    else:
        raise ValueError(f"unknown uncertainty sampling strategy: {strategy}")
    if k is None:
        return order
    return order[: int(k)]


def fine_tune_budget_from_unlabeled(unlabeled_count: int, *, denominator: int = 50) -> int:
    """Paper Fig.8 uses a labeled fine-tuning set equal to 1/50 of the unlabeled pool."""
    if unlabeled_count <= 0:
        raise ValueError("unlabeled_count must be positive")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return max(1, int(unlabeled_count) // int(denominator))


def balanced_source_replay_indices(labels: torch.Tensor, *, per_class: int, seed: int = 0) -> torch.Tensor:
    """Select a small balanced source replay set for preserving source-domain performance."""
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D tensor")
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    generator = torch.Generator(device=labels.device)
    generator.manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for class_id in torch.unique(labels, sorted=True):
        idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=generator, device=labels.device)]
        selected.append(perm[: min(int(per_class), idx.numel())])
    if not selected:
        return torch.empty(0, dtype=torch.long, device=labels.device)
    return torch.cat(selected, dim=0)
