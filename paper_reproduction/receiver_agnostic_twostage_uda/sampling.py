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


def balanced_target_selection(
    ranked_indices: torch.Tensor,
    *,
    k: int,
    labels: torch.Tensor | None = None,
    receivers: list[str] | None = None,
    balance_mode: str = "none",
) -> torch.Tensor:
    """Select target labels from a ranked list with optional class/receiver balance."""
    if ranked_indices.ndim != 1:
        raise ValueError("ranked_indices must be 1-D")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return ranked_indices[:0]
    mode = str(balance_mode).lower().strip()
    if mode in {"", "none"}:
        return ranked_indices[: int(k)]
    if mode == "class":
        if labels is None:
            raise ValueError("class-balanced target selection requires labels")
        groups = [str(int(labels[int(i)].item())) for i in ranked_indices.detach().cpu()]
    elif mode == "receiver":
        if receivers is None:
            raise ValueError("receiver-balanced target selection requires receivers")
        groups = [str(receivers[int(i)]) for i in ranked_indices.detach().cpu()]
    elif mode in {"class_receiver", "receiver_class"}:
        if labels is None or receivers is None:
            raise ValueError("class_receiver-balanced target selection requires labels and receivers")
        groups = [f"{int(labels[int(i)].item())}|{receivers[int(i)]}" for i in ranked_indices.detach().cpu()]
    else:
        raise ValueError(f"unknown target balance mode: {balance_mode}")
    unique_groups = sorted(set(groups))
    if not unique_groups:
        return ranked_indices[: int(k)]
    quota = max(1, (int(k) + len(unique_groups) - 1) // len(unique_groups))
    counts = {group: 0 for group in unique_groups}
    selected_positions: list[int] = []
    for pos, group in enumerate(groups):
        if counts[group] >= quota:
            continue
        selected_positions.append(pos)
        counts[group] += 1
        if len(selected_positions) >= int(k):
            break
    if len(selected_positions) < int(k):
        used = set(selected_positions)
        for pos in range(len(groups)):
            if pos in used:
                continue
            selected_positions.append(pos)
            if len(selected_positions) >= int(k):
                break
    take = torch.as_tensor(selected_positions[: int(k)], dtype=torch.long, device=ranked_indices.device)
    return ranked_indices[take]


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
