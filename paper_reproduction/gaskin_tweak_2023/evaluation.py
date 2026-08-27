from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .calibration import CalibrationState, closed_set_predict, open_set_admit


def _binary_auroc(known_scores: torch.Tensor, unknown_scores: torch.Tensor) -> float:
    """Mann-Whitney AUROC with higher score meaning a known device."""
    comparisons = known_scores[:, None] - unknown_scores[None, :]
    return float((comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean())


@dataclass(frozen=True)
class OpenSetTrial:
    known_points: torch.Tensor
    known_labels: torch.Tensor
    unknown_points: torch.Tensor
    unknown_labels: torch.Tensor


@torch.no_grad()
def closed_set_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    state: CalibrationState,
    *,
    group_size: int = 10,
) -> float:
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0] * group_size:
        raise ValueError("embeddings must contain group_size embeddings for every label")
    grouped = embeddings.reshape(labels.shape[0], group_size, embeddings.shape[1]).mean(dim=1)
    return float(closed_set_predict(grouped, state).eq(labels).float().mean())


def balanced_open_set_trials(
    examples_by_label: Mapping[int, torch.Tensor],
    *,
    known_device_count: int = 5,
    unknown_device_count: int = 5,
    trials: int = 5,
    seed: int,
) -> list[OpenSetTrial]:
    """Draw the paper's five balanced known/unknown device trials."""
    labels = sorted(examples_by_label)
    if trials <= 0 or known_device_count <= 0 or unknown_device_count <= 0:
        raise ValueError("trial and device counts must be positive")
    if len(labels) < known_device_count + unknown_device_count:
        raise ValueError("not enough distinct devices for a balanced open-set trial")
    generator = torch.Generator().manual_seed(seed)
    result: list[OpenSetTrial] = []
    for _ in range(trials):
        order = torch.randperm(len(labels), generator=generator).tolist()
        known = [labels[index] for index in order[:known_device_count]]
        unknown = [labels[index] for index in order[known_device_count : known_device_count + unknown_device_count]]
        known_points = torch.cat([examples_by_label[label] for label in known])
        unknown_points = torch.cat([examples_by_label[label] for label in unknown])
        known_labels = torch.cat([torch.full((examples_by_label[label].shape[0],), label, dtype=torch.long) for label in known])
        unknown_labels = torch.cat([torch.full((examples_by_label[label].shape[0],), label, dtype=torch.long) for label in unknown])
        if known_points.shape[0] != unknown_points.shape[0]:
            raise ValueError("known and unknown trial examples must be balanced")
        result.append(OpenSetTrial(known_points, known_labels, unknown_points, unknown_labels))
    return result


@torch.no_grad()
def open_set_trial_metrics(
    *,
    known_points: torch.Tensor,
    known_labels: torch.Tensor,
    unknown_points: torch.Tensor,
    state: CalibrationState,
) -> dict[str, float]:
    """Compute the paper's open-set score and Algorithm-3 operating metrics."""
    if known_points.ndim != 2 or unknown_points.ndim != 2 or known_labels.shape != (known_points.shape[0],):
        raise ValueError("known points, unknown points and known labels must align")
    known_distances = torch.cdist(known_points, state.centroids, p=2).amin(dim=1)
    unknown_distances = torch.cdist(unknown_points, state.centroids, p=2).amin(dim=1)
    known_admit = open_set_admit(known_points, state)
    unknown_admit = open_set_admit(unknown_points, state)
    known_prediction = closed_set_predict(known_points, state)
    accepted = known_admit
    accepted_accuracy = (
        float(known_prediction[accepted].eq(known_labels[accepted]).float().mean()) if bool(accepted.any()) else 0.0
    )
    return {
        "auroc": _binary_auroc(-known_distances, -unknown_distances),
        "tpr": float(known_admit.float().mean()),
        "fpr": float(unknown_admit.float().mean()),
        "accepted_known_accuracy": accepted_accuracy,
    }
