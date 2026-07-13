from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def _as_label_set(labels: Iterable[int]) -> set[int]:
    return {int(label) for label in labels}


def pairwise_fingerprint_cosines(fingerprints: torch.Tensor) -> torch.Tensor:
    if fingerprints.ndim != 2:
        raise ValueError("fingerprints must be a 2-D tensor")
    count = int(fingerprints.shape[0])
    if count < 2:
        return fingerprints.new_empty(0)
    normalized = F.normalize(fingerprints.float(), dim=1)
    similarity = normalized @ normalized.T
    row, col = torch.triu_indices(count, count, offset=1, device=fingerprints.device)
    return similarity[row, col]


def topological_degree(fingerprints: torch.Tensor) -> float:
    """Return the paper-style sum of pairwise fingerprint cosine values."""
    cosines = pairwise_fingerprint_cosines(fingerprints)
    if cosines.numel() == 0:
        return 0.0
    value = float(cosines.sum().item())
    return 0.0 if abs(value) < 1e-7 else value


def degree_of_conflict(fingerprints: torch.Tensor) -> float:
    """Deviation from the ideal equiangular fingerprint layout.

    The paper's derivation gives an optimal pairwise cosine of
    `-1 / (C - 1)` when C fingerprints are unit-normalized. This helper
    reports the mean absolute deviation from that ideal, so an ideal
    simplex has conflict 0 and colliding fingerprints have larger values.
    """
    count = int(fingerprints.shape[0])
    if count < 2:
        return 0.0
    cosines = pairwise_fingerprint_cosines(fingerprints)
    ideal = -1.0 / float(count - 1)
    value = float(torch.mean(torch.abs(cosines - ideal)).item())
    return 0.0 if value < 1e-6 else value


def stage_accuracy_breakdown(
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    old_class_ids: Iterable[int],
    new_class_ids: Iterable[int],
) -> dict[str, float]:
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("true_labels and predicted_labels must have the same shape")
    old_ids = _as_label_set(old_class_ids)
    new_ids = _as_label_set(new_class_ids)
    true_cpu = true_labels.detach().cpu().long()
    pred_cpu = predicted_labels.detach().cpu().long()

    def _accuracy(mask: torch.Tensor) -> float:
        total = int(mask.sum().item())
        if total == 0:
            return float("nan")
        correct = int((pred_cpu[mask] == true_cpu[mask]).sum().item())
        return correct / float(total)

    old_mask = torch.tensor([int(v) in old_ids for v in true_cpu.tolist()], dtype=torch.bool)
    new_mask = torch.tensor([int(v) in new_ids for v in true_cpu.tolist()], dtype=torch.bool)
    return {
        "old_device_accuracy": _accuracy(old_mask),
        "new_device_accuracy": _accuracy(new_mask),
        "overall_accuracy": _accuracy(torch.ones_like(true_cpu, dtype=torch.bool)),
    }
