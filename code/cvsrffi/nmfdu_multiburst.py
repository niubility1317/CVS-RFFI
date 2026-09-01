from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch


def _matrix_rank(matrix: torch.Tensor, relative_tolerance: float) -> int:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.mT)).abs()
    threshold = eigenvalues.max().clamp_min(1.0) * float(relative_tolerance)
    return int((eigenvalues > threshold).sum().item())


def aggregate_independent_bursts(
    fingerprints: torch.Tensor,
    q_sample: torch.Tensor,
    *,
    physical_sample_ids: Sequence[object],
    branch_fisher: Mapping[str, torch.Tensor] | None = None,
    eps: float = 1e-6,
    rank_tolerance: float = 1e-6,
) -> Dict[str, object]:
    """Aggregate independent physical bursts according to the NMFDU report.

    Fingerprints use ``sum(Q_m z_m) / (sum(Q_m) + eps)``. Fisher matrices
    remain additive observations and are therefore summed without Q weighting.
    Multiple mathematical views of one received IQ are rejected by physical ID.
    """

    fingerprints = torch.as_tensor(fingerprints)
    if fingerprints.dim() != 2 or fingerprints.size(0) < 1:
        raise ValueError("fingerprints must have shape [M,D] with M>=1")
    if not fingerprints.is_floating_point():
        fingerprints = fingerprints.float()
    q_sample = torch.as_tensor(
        q_sample, device=fingerprints.device, dtype=fingerprints.dtype
    ).reshape(-1)
    burst_count = int(fingerprints.size(0))
    if q_sample.shape != (burst_count,):
        raise ValueError("q_sample must have shape [M]")
    if not torch.isfinite(fingerprints).all() or not torch.isfinite(q_sample).all():
        raise ValueError("fingerprints and q_sample must be finite")
    if bool(((q_sample < 0.0) | (q_sample > 1.0)).any()):
        raise ValueError("q_sample must remain in [0,1]")

    physical_ids = tuple(str(value) for value in physical_sample_ids)
    if len(physical_ids) != burst_count or any(not value for value in physical_ids):
        raise ValueError("physical_sample_ids must provide one non-empty ID per burst")
    if len(set(physical_ids)) != burst_count:
        raise ValueError(
            "multi-burst aggregation requires independent physical samples; "
            "duplicate views of one IQ do not add a burst"
        )

    quality_sum = q_sample.sum()
    fingerprint = (
        q_sample.unsqueeze(1) * fingerprints
    ).sum(dim=0) / quality_sum.add(float(eps))

    fisher_total: Dict[str, torch.Tensor] = {}
    branch_rank: Dict[str, int] = {}
    for name, value in (branch_fisher or {}).items():
        matrices = torch.as_tensor(
            value, device=fingerprints.device, dtype=fingerprints.dtype
        )
        if (
            matrices.dim() != 3
            or matrices.size(0) != burst_count
            or matrices.size(1) != matrices.size(2)
        ):
            raise ValueError(f"branch_fisher[{name}] must have shape [M,P,P]")
        if not torch.isfinite(matrices).all():
            raise ValueError(f"branch_fisher[{name}] contains non-finite values")
        total = matrices.sum(dim=0)
        fisher_total[str(name)] = total
        branch_rank[str(name)] = _matrix_rank(total, rank_tolerance)

    return {
        "fingerprint": fingerprint,
        "quality_sum": quality_sum,
        "effective_burst_count": float(quality_sum.detach().cpu().item()),
        "physical_burst_count": burst_count,
        "physical_sample_ids": physical_ids,
        "branch_fisher": fisher_total,
        "branch_rank": branch_rank,
    }
