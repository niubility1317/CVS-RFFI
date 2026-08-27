from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CalibrationState:
    labels: torch.Tensor
    centroids: torch.Tensor
    radii: torch.Tensor


@dataclass(frozen=True)
class DomainCalibrationBank:
    by_domain: dict[Hashable, CalibrationState]


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


@torch.no_grad()
def aggregate_embeddings(embeddings: torch.Tensor, group_size: int = 10) -> torch.Tensor:
    """Average the paper's M input embeddings before a single decision."""
    if embeddings.ndim != 2 or group_size <= 0 or embeddings.shape[0] % group_size:
        raise ValueError("embeddings must contain a positive whole number of groups")
    return embeddings.reshape(-1, group_size, embeddings.shape[1]).mean(dim=1)


@torch.no_grad()
def calibrate_domains(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: Sequence[Hashable],
    *,
    samples_per_class: int,
) -> DomainCalibrationBank:
    """Calibrate one independent centroid/radius state for every target domain."""
    if samples_per_class <= 0 or len(domains) != features.shape[0]:
        raise ValueError("domains must align with features and samples_per_class must be positive")
    ordered_domains = list(dict.fromkeys(domains))
    by_domain: dict[Hashable, CalibrationState] = {}
    for domain in ordered_domains:
        indices = torch.tensor([index for index, value in enumerate(domains) if value == domain], device=features.device)
        domain_features, domain_labels = features[indices], labels[indices]
        for label in torch.unique(domain_labels, sorted=True):
            count = int(domain_labels.eq(label).sum())
            if count != samples_per_class:
                raise ValueError(f"domain {domain!r}, class {int(label)} must contain exactly {samples_per_class} samples")
        by_domain[domain] = calibrate(domain_features, domain_labels)
    return DomainCalibrationBank(by_domain=by_domain)


@torch.no_grad()
def closed_set_predict_grouped(
    embeddings: torch.Tensor,
    state: CalibrationState,
    *,
    group_size: int = 10,
) -> torch.Tensor:
    return closed_set_predict(aggregate_embeddings(embeddings, group_size=group_size), state)


@torch.no_grad()
def open_set_admit_grouped(
    embeddings: torch.Tensor,
    state: CalibrationState,
    *,
    group_size: int = 10,
) -> torch.Tensor:
    return open_set_admit(aggregate_embeddings(embeddings, group_size=group_size), state)
