from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CalibrationState:
    labels: torch.Tensor
    centroids: torch.Tensor
    radii: torch.Tensor


@torch.no_grad()
def calibrate(features: torch.Tensor, labels: torch.Tensor) -> CalibrationState:
    """Store the paper's per-class mean embedding and mean L2 radius."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features must be [samples, dim] and labels must be [samples]")
    classes = torch.unique(labels, sorted=True)
    centroids, radii = [], []
    for label in classes:
        members = features[labels.eq(label)]
        centroid = members.mean(dim=0)
        centroids.append(centroid)
        radii.append(torch.linalg.vector_norm(members - centroid, dim=1).mean())
    return CalibrationState(
        labels=classes.detach().clone(),
        centroids=torch.stack(centroids).detach(),
        radii=torch.stack(radii).detach(),
    )


def _distances(points: torch.Tensor, state: CalibrationState) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != state.centroids.shape[1]:
        raise ValueError("points must be [samples, embedding_dim]")
    return torch.cdist(points, state.centroids, p=2)


@torch.no_grad()
def closed_set_predict(points: torch.Tensor, state: CalibrationState) -> torch.Tensor:
    """Apply Algorithm 2: nearest contained class, otherwise minimum excess."""
    distances = _distances(points, state)
    inside = distances < state.radii.unsqueeze(0)
    nearest_inside = distances.masked_fill(~inside, float("inf")).argmin(dim=1)
    excess = distances - state.radii.unsqueeze(0)
    nearest_excess = excess.argmin(dim=1)
    positions = torch.where(inside.any(dim=1), nearest_inside, nearest_excess)
    return state.labels[positions]


@torch.no_grad()
def open_set_admit(points: torch.Tensor, state: CalibrationState) -> torch.Tensor:
    """Apply Algorithm 3: admit when at least one class radius contains a point."""
    return (_distances(points, state) <= state.radii.unsqueeze(0)).any(dim=1)
