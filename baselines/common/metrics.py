from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Sequence

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def grouped_accuracy(preds: Sequence[int], labels: Sequence[int], groups: Sequence[int]) -> Dict[int, float]:
    buckets = defaultdict(lambda: [0, 0])
    for p, y, g in zip(preds, labels, groups):
        buckets[int(g)][1] += 1
        buckets[int(g)][0] += int(int(p) == int(y))
    return {g: correct / max(1, total) for g, (correct, total) in buckets.items()}


def confusion_matrix(preds: Iterable[int], labels: Iterable[int], num_classes: int) -> torch.Tensor:
    mat = torch.zeros(int(num_classes), int(num_classes), dtype=torch.long)
    for p, y in zip(preds, labels):
        mat[int(y), int(p)] += 1
    return mat
