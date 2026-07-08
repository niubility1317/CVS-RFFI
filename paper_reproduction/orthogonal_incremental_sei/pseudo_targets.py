from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def _as_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)


def make_simplex_pseudo_targets(
    *,
    num_targets: int,
    feature_dim: int,
    total_classes: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct the paper's uniformly separated pseudo-target directions.

    The paper states the practical bound ``|C| <= N <= d + 1`` and derives the
    optimal pairwise inner product ``-1 / (N - 1)``. A regular simplex realizes
    that geometry exactly when ``N <= d + 1``.
    """

    if num_targets <= 0:
        raise ValueError("num_targets must be positive")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    if total_classes is not None and num_targets < int(total_classes):
        raise ValueError("num_targets must be >= total_classes")
    if num_targets > feature_dim + 1:
        raise ValueError("num_targets must be <= feature_dim + 1")

    target_device = _as_device(device)
    if num_targets == 1:
        out = torch.zeros(1, feature_dim, dtype=dtype, device=target_device)
        out[0, 0] = 1.0
        return out

    eye = torch.eye(num_targets, dtype=torch.float64, device=target_device)
    centered = eye - torch.full_like(eye, 1.0 / float(num_targets))
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    coords = u[:, : num_targets - 1] * s[: num_targets - 1]
    coords = F.normalize(coords, dim=1)
    if feature_dim > coords.size(1):
        pad = torch.zeros(num_targets, feature_dim - coords.size(1), dtype=coords.dtype, device=coords.device)
        coords = torch.cat([coords, pad], dim=1)
    return coords.to(dtype=dtype)


def optimize_pseudo_targets(
    *,
    num_targets: int,
    feature_dim: int,
    total_classes: int | None = None,
    temperature: float = 0.01,
    steps: int = 0,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return pseudo-targets for formula (4).

    ``steps`` is accepted for configuration parity with the paper. The simplex
    solution is already the closed-form optimum for the stated bound, so no
    iterative optimizer is needed for the reproduction default.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if seed is not None:
        torch.manual_seed(int(seed))
    if steps == 0:
        return make_simplex_pseudo_targets(
            num_targets=num_targets,
            feature_dim=feature_dim,
            total_classes=total_classes,
            device=device,
        )

    target_device = _as_device(device)
    if total_classes is not None and num_targets < int(total_classes):
        raise ValueError("num_targets must be >= total_classes")
    if num_targets > feature_dim + 1:
        raise ValueError("num_targets must be <= feature_dim + 1")
    raw = torch.randn(num_targets, feature_dim, dtype=torch.float32, device=target_device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = pseudo_target_orthogonal_loss(raw, temperature=temperature)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw.copy_(F.normalize(raw, dim=1))
    return F.normalize(raw.detach(), dim=1)


def pseudo_target_orthogonal_loss(targets: torch.Tensor, *, temperature: float = 0.01) -> torch.Tensor:
    if targets.ndim != 2:
        raise ValueError("targets must have shape [num_targets, feature_dim]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normed = F.normalize(targets.float(), dim=1)
    sim = normed @ normed.t()
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    if sim.size(0) == 1:
        return sim.new_tensor(0.0)
    logits = sim[mask].view(sim.size(0), sim.size(0) - 1) / temperature
    return torch.logsumexp(logits, dim=1).mean()


def perturb_pseudo_targets(
    targets: torch.Tensor,
    *,
    noise_range: float,
    seed: int | None = None,
    renormalize: bool = True,
) -> torch.Tensor:
    if noise_range < 0:
        raise ValueError("noise_range must be non-negative")
    if seed is not None:
        generator = torch.Generator(device=targets.device)
        generator.manual_seed(int(seed))
    else:
        generator = None
    noise = torch.empty_like(targets).uniform_(-noise_range, noise_range, generator=generator)
    out = targets + noise
    return F.normalize(out, dim=1) if renormalize else out


def assign_base_targets(base_labels: Iterable[int], pseudo_targets: torch.Tensor) -> dict[int, torch.Tensor]:
    labels = [int(label) for label in base_labels]
    if len(set(labels)) != len(labels):
        raise ValueError("base_labels must be unique")
    if len(labels) > pseudo_targets.size(0):
        raise ValueError("not enough pseudo targets for base labels")
    return {label: pseudo_targets[index].detach().clone() for index, label in enumerate(labels)}


def target_matrix_for_labels(labels: torch.Tensor, assigned_targets: dict[int, torch.Tensor]) -> torch.Tensor:
    rows = []
    for label in labels.detach().cpu().tolist():
        key = int(label)
        if key not in assigned_targets:
            raise ValueError(f"label {key} has no assigned pseudo target")
        rows.append(assigned_targets[key].to(device=labels.device))
    return torch.stack(rows, dim=0)
