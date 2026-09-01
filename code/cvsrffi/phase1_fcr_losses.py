from __future__ import annotations

import math

import torch

from .phase1_fcr_physics import FisherGateOutput, FrozenFingerprintFeatureBank, _as_complex_iq
from .phase1_fcr_types import FCRConfig


def heteroscedastic_complex_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    config: FCRConfig,
) -> torch.Tensor:
    """Complex Gaussian NLL with variance interpreted only in config bounds."""

    target_complex = _as_complex_iq(target)
    mean_complex = _as_complex_iq(mean)
    if target_complex.shape != mean_complex.shape:
        raise ValueError("target and mean must have matching complex IQ shapes")
    variance = log_variance.to(target_complex.real.dtype).exp()
    variance = variance.clamp(config.variance_floor, config.variance_ceiling)
    if variance.ndim == 1:
        variance = variance[:, None]
    if variance.shape != target_complex.shape:
        variance = variance.expand_as(target_complex)
    error_power = (target_complex - mean_complex).abs().square()
    return (error_power / variance + variance.log() + math.log(math.pi)).mean()


def mrstft_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    windows: tuple[int, int, int] = (32, 64, 128),
    noise_floor: float = 1e-6,
) -> torch.Tensor:
    """Log-magnitude multi-resolution STFT loss with explicit low-energy floor."""

    target_complex = _as_complex_iq(target)
    prediction_complex = _as_complex_iq(prediction)
    if target_complex.shape != prediction_complex.shape:
        raise ValueError("target and prediction must have matching complex IQ shapes")
    length = target_complex.size(-1)
    losses = []
    for requested in windows:
        n_fft = min(int(requested), length)
        if n_fft < 2:
            continue
        hop = max(1, n_fft // 4)
        window = torch.hann_window(n_fft, device=target_complex.device, dtype=target_complex.real.dtype)
        target_spec = torch.stft(target_complex, n_fft=n_fft, hop_length=hop, window=window, return_complex=True, onesided=False)
        prediction_spec = torch.stft(prediction_complex, n_fft=n_fft, hop_length=hop, window=window, return_complex=True, onesided=False)
        target_log = (target_spec.abs() + float(noise_floor)).log()
        prediction_log = (prediction_spec.abs() + float(noise_floor)).log()
        losses.append((target_log - prediction_log).square().mean())
    if not losses:
        return target_complex.real.new_zeros(())
    return torch.stack(losses).mean()


def phase_increment_loss(target: torch.Tensor, prediction: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Amplitude-weighted wrapped phase-increment loss using unit conjugate products."""

    target_complex = _as_complex_iq(target)
    prediction_complex = _as_complex_iq(prediction)
    if target_complex.shape != prediction_complex.shape:
        raise ValueError("target and prediction must have matching complex IQ shapes")
    if target_complex.size(-1) < 2:
        return target_complex.real.new_zeros(())
    target_increment = target_complex[:, 1:] * target_complex[:, :-1].conj()
    prediction_increment = prediction_complex[:, 1:] * prediction_complex[:, :-1].conj()
    target_unit = target_increment / target_increment.abs().clamp_min(eps)
    prediction_unit = prediction_increment / prediction_increment.abs().clamp_min(eps)
    circular_error = 1.0 - (target_unit * prediction_unit.conj()).real.clamp(-1.0, 1.0)
    weight = (target_complex[:, 1:].abs() * target_complex[:, :-1].abs()).detach()
    return (circular_error * weight).sum() / weight.sum().clamp_min(eps)


def physical_feature_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    gate: FisherGateOutput,
    *,
    bank: FrozenFingerprintFeatureBank | None = None,
) -> torch.Tensor:
    """Compare only fixed physical feature blocks under their matching Fisher weights."""

    feature_bank = bank if bank is not None else FrozenFingerprintFeatureBank()
    target_features = feature_bank(target)
    prediction_features = feature_bank(prediction)
    losses = []
    for name, target_block in target_features.blocks.items():
        prediction_block = prediction_features.blocks[name]
        per_example = (target_block - prediction_block).square().reshape(target_block.size(0), -1).mean(dim=1)
        weight = gate.block_weights.get(name, gate.block_weights.get("pa", torch.ones_like(per_example)))
        weight = torch.as_tensor(weight, device=per_example.device, dtype=per_example.dtype).detach()
        weight = weight.expand_as(per_example) if weight.numel() == 1 else weight.reshape(-1)[: per_example.numel()]
        losses.append((per_example * weight.clamp(0.0, 1.0)).mean())
    return torch.stack(losses).mean() if losses else _as_complex_iq(target).real.new_zeros(())
