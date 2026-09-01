from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import torch


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.is_complex():
        z = x
    elif x.dim() >= 2 and int(x.size(-2)) == 2:
        z = torch.complex(x[..., 0, :].float(), x[..., 1, :].float())
    else:
        z = torch.complex(x.float(), torch.zeros_like(x.float()))
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.dim() != 2:
        raise ValueError("IQ input must be complex [B,T] or real [B,2,T]")
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _matrix_summary(matrix: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2).conj())
    eigenvalues = torch.linalg.eigvalsh(matrix).real.clamp_min(0.0)
    trace = eigenvalues.sum(dim=-1)
    probabilities = eigenvalues / trace.unsqueeze(-1).clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    effective_rank = torch.exp(entropy)
    lambda_min = eigenvalues[..., 0]
    lambda_max = eigenvalues[..., -1]
    return {
        "matrix": matrix,
        "eigenvalues": eigenvalues,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition": lambda_max / lambda_min.clamp_min(eps),
        "effective_rank": effective_rank,
        "log_volume": torch.log(eigenvalues + eps).sum(dim=-1),
        "trace": trace,
    }


def _weighted_jacobian(
    jacobian: torch.Tensor, weight: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if jacobian.dim() == 2:
        jacobian = jacobian.unsqueeze(0)
    if jacobian.dim() != 3:
        raise ValueError("jacobian must have shape [B,N,P] or [N,P]")
    if jacobian.is_complex():
        jacobian = torch.cat([jacobian.real, jacobian.imag], dim=1)
        if weight is not None:
            weight = torch.cat([weight, weight], dim=-1)
    jacobian = torch.nan_to_num(jacobian.float(), nan=0.0, posinf=0.0, neginf=0.0)
    batch, observations, _ = jacobian.shape
    if weight is None:
        weight = jacobian.new_ones((batch, observations))
    elif weight.dim() == 1:
        weight = weight.unsqueeze(0)
    weight = torch.nan_to_num(
        weight.to(device=jacobian.device, dtype=jacobian.dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    if tuple(weight.shape) != (batch, observations):
        raise ValueError("weight must match jacobian batch and observation dimensions")
    return jacobian * weight.sqrt().unsqueeze(-1), weight


def effective_fisher_summary(
    j_target: torch.Tensor,
    j_nuisance: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Return nuisance-marginalized Fisher evidence via a Schur complement."""

    target, normalized_weight = _weighted_jacobian(j_target, weight)
    j_tt = target.transpose(-1, -2) @ target
    if j_nuisance is None or int(j_nuisance.shape[-1]) == 0:
        summary = _matrix_summary(j_tt, eps)
        summary["raw_matrix"] = j_tt
        return summary
    nuisance, _ = _weighted_jacobian(j_nuisance, normalized_weight)
    if nuisance.shape[:2] != target.shape[:2]:
        raise ValueError("target and nuisance jacobians must share [B,N]")
    j_tn = target.transpose(-1, -2) @ nuisance
    j_nn = nuisance.transpose(-1, -2) @ nuisance
    identity = torch.eye(j_nn.size(-1), device=j_nn.device, dtype=j_nn.dtype).expand_as(j_nn)
    projected = torch.linalg.solve(
        j_nn + float(eps) * identity, j_tn.transpose(-1, -2)
    )
    j_eff = j_tt - j_tn @ projected
    summary = _matrix_summary(j_eff, eps)
    summary["raw_matrix"] = j_tt
    summary["nuisance_matrix"] = j_nn
    return summary


def complex_excitation_stats(
    s_hat: torch.Tensor,
    segment_ids: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute non-circular excitation and real-IQ directional evidence."""

    z = _as_complex(s_hat)
    power = z.abs().square().mean(dim=-1).clamp_min(eps)
    beta = z.square().mean(dim=-1) / power
    centered = torch.stack(
        [
            z.real - z.real.mean(dim=-1, keepdim=True),
            z.imag - z.imag.mean(dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    covariance = centered.transpose(-1, -2) @ centered
    covariance = covariance / max(1, int(z.size(-1)))
    covariance = covariance / covariance.diagonal(
        dim1=-2, dim2=-1
    ).sum(dim=-1).view(-1, 1, 1).clamp_min(eps)
    summary = _matrix_summary(covariance, eps)
    out = {
        "beta_real": beta.real,
        "beta_imag": beta.imag,
        "rho": beta.abs().clamp(0.0, 1.0),
        "iq_matrix": summary["matrix"],
        "iq_eigenvalues": summary["eigenvalues"],
        "iq_lambda_min": summary["lambda_min"],
        "iq_condition": summary["condition"],
        "power": power,
    }
    if segment_ids is not None:
        segment_ids = segment_ids.to(device=z.device)
        if segment_ids.dim() == 1:
            segment_ids = segment_ids.unsqueeze(0).expand(z.size(0), -1)
        variances = []
        for batch_index in range(z.size(0)):
            values = []
            for segment in torch.unique(segment_ids[batch_index], sorted=True):
                mask = segment_ids[batch_index] == segment
                z_segment = z[batch_index, mask]
                p_segment = z_segment.abs().square().mean().clamp_min(eps)
                values.append((z_segment.square().mean() / p_segment).abs())
            variances.append(torch.stack(values).var(unbiased=False))
        out["segment_rho_variance"] = torch.stack(variances)
    return out


def _amplitude_entropy(
    amplitude: torch.Tensor, bins: int = 16, eps: float = 1e-6
) -> torch.Tensor:
    normalized = amplitude / amplitude.amax(dim=-1, keepdim=True).clamp_min(eps)
    indices = torch.clamp((normalized * bins).long(), min=0, max=bins - 1)
    counts = amplitude.new_zeros((amplitude.size(0), bins))
    counts.scatter_add_(1, indices, torch.ones_like(amplitude))
    probabilities = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1) / math.log(bins)


def memory_polynomial_gram_stats(
    s_hat: torch.Tensor,
    order: Sequence[int] | int = (1, 3, 5),
    memory_depth: int = 1,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute column-normalized memory-polynomial Gram identifiability."""

    z = _as_complex(s_hat)
    orders: Iterable[int] = (order,) if isinstance(order, int) else tuple(order)
    orders = tuple(int(value) for value in orders)
    memory_depth = int(memory_depth)
    if memory_depth < 1 or not orders or any(value < 1 or value % 2 == 0 for value in orders):
        raise ValueError("order must contain positive odd integers and memory_depth must be positive")
    if z.size(-1) < memory_depth:
        raise ValueError("excitation is shorter than memory_depth")
    aligned_length = int(z.size(-1)) - memory_depth + 1
    columns = []
    for delay in range(memory_depth):
        delayed = z[
            :, memory_depth - 1 - delay : memory_depth - 1 - delay + aligned_length
        ]
        for polynomial_order in orders:
            columns.append(delayed * delayed.abs().pow(polynomial_order - 1))
    design = torch.stack(columns, dim=-1)
    column_rms = design.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    design = design / column_rms
    if weight is None:
        sample_weight = design.real.new_ones((z.size(0), aligned_length))
    else:
        sample_weight = weight[..., -aligned_length:].to(
            device=z.device, dtype=design.real.dtype
        )
        if sample_weight.dim() == 1:
            sample_weight = sample_weight.unsqueeze(0)
        sample_weight = torch.nan_to_num(sample_weight, nan=0.0).clamp_min(0.0)
    weighted_design = design * sample_weight.sqrt().unsqueeze(-1)
    gram = weighted_design.transpose(-1, -2).conj() @ weighted_design
    gram = gram / sample_weight.sum(dim=-1).view(-1, 1, 1).clamp_min(eps)
    gram = gram / gram.diagonal(dim1=-2, dim2=-1).real.sum(
        dim=-1
    ).view(-1, 1, 1).clamp_min(eps)
    summary = _matrix_summary(gram, eps)
    amplitude = z.abs()
    mean_power = amplitude.square().mean(dim=-1).clamp_min(eps)
    peak_power = amplitude.square().amax(dim=-1)
    mean_amplitude = amplitude.mean(dim=-1).clamp_min(eps)
    q05 = torch.quantile(amplitude, 0.05, dim=-1)
    q95 = torch.quantile(amplitude, 0.95, dim=-1)
    return {
        "gram": summary["matrix"],
        "gram_eigenvalues": summary["eigenvalues"],
        "lambda_min": summary["lambda_min"],
        "condition": summary["condition"],
        "effective_rank": summary["effective_rank"],
        "log_volume": summary["log_volume"],
        "papr": peak_power / mean_power,
        "mu4": amplitude.pow(4).mean(dim=-1) / mean_power.square(),
        "mu6": amplitude.pow(6).mean(dim=-1) / mean_power.pow(3),
        "amplitude_entropy": _amplitude_entropy(amplitude, eps=eps),
        "amplitude_dynamic_range": (q95 - q05) / mean_amplitude,
        "clipping_rate": (
            amplitude >= 0.99 * amplitude.amax(dim=-1, keepdim=True)
        ).float().mean(dim=-1),
        "power_coverage": q95.square() / mean_power,
    }


def spectral_occupancy_stats(
    s_hat: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    z = _as_complex(s_hat)
    if weight is not None:
        window = weight.to(device=z.device, dtype=z.real.dtype)
        if window.dim() == 1:
            window = window.unsqueeze(0)
        z = z * window
    spectrum = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
    power = spectrum.abs().square()
    probability = power / power.sum(dim=-1, keepdim=True).clamp_min(eps)
    length = int(z.size(-1))
    entropy = -(probability * probability.clamp_min(eps).log()).sum(
        dim=-1
    ) / math.log(max(2, length))
    sorted_power = torch.sort(probability, dim=-1, descending=True).values
    occupied_bins = (sorted_power.cumsum(dim=-1) < 0.90).sum(dim=-1) + 1
    edge = max(1, length // 8)
    edge_energy = probability[:, :edge].sum(dim=-1) + probability[:, -edge:].sum(dim=-1)
    occupied_fraction = (probability > (1.0 / length)).float().mean(dim=-1)
    strongest = sorted_power[:, : max(1, length // 10)].sum(dim=-1)
    residual = (1.0 - strongest).clamp_min(eps)
    return {
        "effective_bandwidth": occupied_bins.to(probability.dtype) / float(length),
        "occupied_fraction": occupied_fraction,
        "edge_energy": edge_energy,
        "spectral_entropy": entropy,
        "spectral_snr": 10.0 * torch.log10(strongest.clamp_min(eps) / residual),
    }


def _unwrap_phase(phase: torch.Tensor) -> torch.Tensor:
    if phase.size(-1) <= 1:
        return phase
    difference = phase[:, 1:] - phase[:, :-1]
    wrapped = torch.remainder(difference + math.pi, 2.0 * math.pi) - math.pi
    wrapped = torch.where(
        (wrapped == -math.pi) & (difference > 0),
        torch.full_like(wrapped, math.pi),
        wrapped,
    )
    return torch.cat(
        [phase[:, :1], phase[:, :1] + wrapped.cumsum(dim=-1)], dim=-1
    )


def phase_residual_stats(
    r_canonical: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    polynomial_order: int = 2,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    z = _as_complex(r_canonical)
    phase = _unwrap_phase(torch.angle(z))
    batch, length = phase.shape
    if valid_mask is None:
        valid_mask = phase.new_ones((batch, length))
    elif valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    valid_mask = valid_mask.to(device=phase.device, dtype=phase.dtype).clamp(0.0, 1.0)
    time = torch.linspace(-1.0, 1.0, length, device=phase.device, dtype=phase.dtype)
    design = torch.stack(
        [time.pow(power) for power in range(int(polynomial_order) + 1)], dim=-1
    ).unsqueeze(0).expand(batch, -1, -1)
    weighted_design = design * valid_mask.sqrt().unsqueeze(-1)
    weighted_phase = phase * valid_mask.sqrt()
    normal = weighted_design.transpose(-1, -2) @ weighted_design
    identity = torch.eye(
        normal.size(-1), device=normal.device, dtype=normal.dtype
    ).expand_as(normal)
    rhs = weighted_design.transpose(-1, -2) @ weighted_phase.unsqueeze(-1)
    coefficients = torch.linalg.solve(normal + eps * identity, rhs)
    fitted = (design @ coefficients).squeeze(-1)
    residual = (phase - fitted) * valid_mask
    denominator = valid_mask.sum(dim=-1).clamp_min(1.0)
    residual_rms = (residual.square().sum(dim=-1) / denominator).sqrt()
    if length > 1:
        residual_jump = (residual[:, 1:] - residual[:, :-1]).abs()
        cycle_slip_rate = (residual_jump > (math.pi / 2.0)).float().mean(dim=-1)
    else:
        cycle_slip_rate = residual_rms.new_zeros(residual_rms.shape)
    return {
        "residual_rms": residual_rms,
        "stability": torch.exp(-residual_rms),
        "cycle_slip_rate": cycle_slip_rate,
        "coefficients": coefficients.squeeze(-1),
        "phase_snr": 10.0
        * torch.log10(
            phase.var(dim=-1, unbiased=False).clamp_min(eps)
            / residual_rms.square().clamp_min(eps)
        ),
    }


def _complex_cumulants(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m20 = z.square().mean()
    m21 = z.abs().square().mean()
    c40 = z.pow(4).mean() - 3.0 * m20.square()
    c42 = z.abs().pow(4).mean() - m20.abs().square() - 2.0 * m21.square()
    return c40, c42


def hos_confidence_stats(
    r_canonical: torch.Tensor,
    segment_ids: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    z = _as_complex(r_canonical)
    batch, length = z.shape
    if segment_ids is None:
        segment_count = min(4, length)
        segment_ids = torch.div(
            torch.arange(length, device=z.device) * segment_count,
            length,
            rounding_mode="floor",
        ).unsqueeze(0).expand(batch, -1)
    elif segment_ids.dim() == 1:
        segment_ids = segment_ids.unsqueeze(0).expand(batch, -1)
    segment_ids = segment_ids.to(device=z.device)
    if tuple(segment_ids.shape) != (batch, length):
        raise ValueError("segment_ids must match the IQ batch and length")
    batch_c40 = []
    batch_c42 = []
    batch_variance = []
    batch_confidence = []
    for batch_index in range(batch):
        c40_values = []
        c42_values = []
        for segment in torch.unique(segment_ids[batch_index], sorted=True):
            segment_value = z[batch_index, segment_ids[batch_index] == segment]
            c40, c42 = _complex_cumulants(segment_value)
            c40_values.append(c40)
            c42_values.append(c42)
        c40_stack = torch.stack(c40_values)
        c42_stack = torch.stack(c42_values)
        evidence = torch.stack([c40_stack.abs(), c42_stack.abs()], dim=-1)
        variance = evidence.var(dim=0, unbiased=False).mean()
        mean_square = evidence.mean(dim=0).square().mean().clamp_min(eps)
        confidence = 1.0 / (1.0 + variance / mean_square)
        batch_c40.append(c40_stack.mean().abs())
        batch_c42.append(c42_stack.mean().abs())
        batch_variance.append(variance)
        batch_confidence.append(confidence)
    return {
        "c40_abs": torch.stack(batch_c40),
        "c42_abs": torch.stack(batch_c42),
        "segment_variance": torch.stack(batch_variance),
        "confidence": torch.stack(batch_confidence),
    }
