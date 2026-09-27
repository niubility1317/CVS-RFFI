from __future__ import annotations

import math
from collections.abc import Callable
from typing import AbstractSet

import torch

from .phase1_fcr_physics import FisherGateOutput, FrozenFingerprintFeatureBank, _as_complex_iq
from .phase1_fcr_transplant import TransplantLossOutput, compute_directed_transplant_losses
from .phase1_fcr_types import FCRConfig, FCRDecodeOutput, FCRFactorOutput, FCRLossOutput, FCRPairBatch


def compute_transplant_losses(**kwargs) -> TransplantLossOutput:
    """Expose Task8 directed-transplant loss through the shared FCR loss module."""

    return compute_directed_transplant_losses(**kwargs)


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


def _connected_zero(reference: torch.Tensor) -> torch.Tensor:
    """Produce an exact scalar zero without fabricating a replacement pair."""

    return reference.reshape(-1).sum() * 0.0


def _pair_mask(pair: FCRPairBatch, direction: str) -> torch.Tensor:
    """Return a directional mask, falling back only to the synchronized nuisance pair."""

    mask = pair.pair_valid_mask.get(direction, pair.pair_valid_mask.get("nuisance"))
    if mask is None:
        return torch.zeros(pair.clean_iq.size(0), device=pair.clean_iq.device, dtype=torch.bool)
    return torch.as_tensor(mask, device=pair.clean_iq.device, dtype=torch.bool).reshape(-1)


def _masked_nll(
    target: torch.Tensor,
    decoded: FCRDecodeOutput,
    mask: torch.Tensor,
    config: FCRConfig,
) -> torch.Tensor:
    """Apply the existing bounded NLL only to genuine valid source/destination pairs."""

    if not bool(mask.any()):
        return _connected_zero(decoded.mu_iq)
    return heteroscedastic_complex_nll(
        target[mask], decoded.mu_iq[mask], decoded.log_variance[mask], config
    )


def _nuisance_vector(factors: FCRFactorOutput) -> torch.Tensor:
    """Flatten structured nuisance parts, excluding eta supervision predictions."""

    parts = [value.reshape(value.size(0), -1) for name, value in factors.z_n_parts.items() if name != "eta_pred"]
    if not parts:
        return _connected_zero(factors.z_s).expand(factors.z_s.size(0), 1)
    return torch.cat(parts, dim=1)


def _unit_sphere(z_f_id: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return z_f_id / z_f_id.norm(dim=-1, keepdim=True).clamp_min(eps)


def _strict_axis_rows(pair: FCRPairBatch, axis: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve only explicitly valid, in-bounds intervention indices."""

    batch_size = int(pair.clean_iq.size(0))
    device = pair.clean_iq.device
    index = torch.as_tensor(
        getattr(pair, f"{axis}_pair_index"),
        device=device,
        dtype=torch.long,
    ).reshape(-1)
    mask = torch.as_tensor(
        pair.pair_valid_mask.get(axis, torch.zeros(batch_size, device=device)),
        device=device,
        dtype=torch.bool,
    ).reshape(-1)
    if index.numel() != batch_size or mask.numel() != batch_size:
        raise ValueError(f"{axis} intervention fields must have one entry per batch row")
    valid = mask & index.ge(0) & index.lt(batch_size)
    source = torch.arange(batch_size, device=device)[valid]
    return source, index[valid]


def compute_three_axis_intervention_loss(
    *,
    pair: FCRPairBatch,
    clean_factors: FCRFactorOutput,
    leo_factors: FCRFactorOutput,
    allow_fingerprint: bool,
) -> FCRLossOutput:
    """Consume strict nuisance/content/fingerprint axes without fallback pairs."""

    zero = _connected_zero(clean_factors.z_s) + _connected_zero(leo_factors.z_s)
    components: dict[str, torch.Tensor] = {}
    metrics: dict[str, float | str] = {}

    nuisance_source, nuisance_target = _strict_axis_rows(pair, "nuisance")
    if nuisance_source.numel() == 0:
        components["nuisance_axis"] = zero
        metrics["nuisance_status"] = "N/A"
    else:
        components["nuisance_axis"] = (
            symmetric_stopgrad_distance(
                clean_factors.z_s[nuisance_source],
                leo_factors.z_s[nuisance_target],
            )
            + symmetric_stopgrad_distance(
                clean_factors.z_f_id[nuisance_source],
                leo_factors.z_f_id[nuisance_target],
            )
            + symmetric_stopgrad_distance(
                clean_factors.z_tx_state[nuisance_source],
                leo_factors.z_tx_state[nuisance_target],
            )
        )
        metrics["nuisance_status"] = "available"
    metrics["nuisance_pairs"] = float(nuisance_source.numel())

    content_source, content_target = _strict_axis_rows(pair, "content")
    if content_source.numel() == 0:
        components["content_axis"] = zero
        metrics["content_status"] = "N/A"
    else:
        components["content_axis"] = (
            symmetric_stopgrad_distance(
                clean_factors.z_f_id[content_source],
                clean_factors.z_f_id[content_target],
            )
            + symmetric_stopgrad_distance(
                clean_factors.z_tx_state[content_source],
                clean_factors.z_tx_state[content_target],
            )
            + symmetric_stopgrad_distance(
                _nuisance_vector(clean_factors)[content_source],
                _nuisance_vector(clean_factors)[content_target],
            )
        )
        metrics["content_status"] = "available"
    metrics["content_pairs"] = float(content_source.numel())

    fingerprint_source, fingerprint_target = _strict_axis_rows(pair, "fingerprint")
    if not bool(allow_fingerprint) or fingerprint_source.numel() == 0:
        components["fingerprint_axis"] = zero
        metrics["fingerprint_status"] = "N/A"
        fingerprint_count = 0
    else:
        components["fingerprint_axis"] = (
            symmetric_stopgrad_distance(
                clean_factors.z_s[fingerprint_source],
                clean_factors.z_s[fingerprint_target],
            )
            + symmetric_stopgrad_distance(
                _nuisance_vector(clean_factors)[fingerprint_source],
                _nuisance_vector(clean_factors)[fingerprint_target],
            )
        )
        metrics["fingerprint_status"] = "available"
        fingerprint_count = int(fingerprint_source.numel())
    metrics["fingerprint_pairs"] = float(fingerprint_count)

    total = sum(components.values(), zero)
    return FCRLossOutput(total=total, components=components, metrics=metrics)


def symmetric_stopgrad_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Symmetric two-way stop-gradient distance, retaining gradients for both branches."""

    if left.shape != right.shape:
        raise ValueError("shared factor tensors must have matching shapes")
    return 0.5 * ((left - right.detach()).square().mean() + (left.detach() - right).square().mean())


def _anti_collapse_loss(z: torch.Tensor, *, minimum_std: float = 0.1) -> torch.Tensor:
    flat = z.reshape(z.size(0), -1)
    if flat.size(0) == 0:
        return _connected_zero(z)
    centered = flat - flat.mean(dim=0, keepdim=True)
    # The derivative of sqrt(x) is singular at x=0. Constant latent
    # dimensions are expected early in training, so keep their backward pass
    # finite instead of relying on the later non-finite batch skip.
    std = (centered.square().mean(dim=0) + 1e-6).sqrt()
    variance_floor = (float(minimum_std) - std).clamp_min(0.0).square().mean()
    if flat.size(1) < 2:
        covariance = _connected_zero(flat)
    else:
        covariance_matrix = centered.transpose(0, 1).matmul(centered) / max(int(flat.size(0)), 1)
        off_diagonal = covariance_matrix - torch.diag(torch.diagonal(covariance_matrix))
        covariance = off_diagonal.square().mean()
    return variance_floor + covariance


def _cross_covariance_loss(*factors: torch.Tensor) -> torch.Tensor:
    flattened = [factor.reshape(factor.size(0), -1) for factor in factors]
    terms: list[torch.Tensor] = []
    for left_index, left in enumerate(flattened):
        left_centered = left - left.mean(dim=0, keepdim=True)
        for right in flattened[left_index + 1 :]:
            right_centered = right - right.mean(dim=0, keepdim=True)
            covariance = left_centered.transpose(0, 1).matmul(right_centered) / max(int(left.size(0)), 1)
            terms.append(covariance.square().mean())
    return torch.stack(terms).mean() if terms else _connected_zero(flattened[0])


def _cycle_distance(reencoded: FCRFactorOutput, source: FCRFactorOutput, destination: FCRFactorOutput) -> torch.Tensor:
    """Recover detached source content/fingerprint and detached destination nuisance."""

    recovered_nuisance = _nuisance_vector(reencoded)
    destination_nuisance = _nuisance_vector(destination)
    if recovered_nuisance.shape != destination_nuisance.shape:
        raise ValueError("re-encoded nuisance must match destination nuisance shape")
    return (
        (reencoded.z_s - source.z_s.detach()).square().mean()
        + (_unit_sphere(reencoded.z_f_id) - _unit_sphere(source.z_f_id.detach())).square().mean()
        + (recovered_nuisance - destination_nuisance.detach()).square().mean()
    )


def compute_cross_losses(
    *,
    clean_factors: FCRFactorOutput,
    leo_factors: FCRFactorOutput,
    clean_self: FCRDecodeOutput,
    leo_self: FCRDecodeOutput,
    clean_to_leo: FCRDecodeOutput,
    leo_to_clean: FCRDecodeOutput,
    pair: FCRPairBatch,
    reencode_clean_to_leo: Callable[[torch.Tensor], FCRFactorOutput],
    reencode_leo_to_clean: Callable[[torch.Tensor], FCRFactorOutput],
    config: FCRConfig,
    active: AbstractSet[str] | None = None,
    domain_confusion_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    probe_metric: Callable[[tuple[FCRFactorOutput, FCRFactorOutput], torch.Tensor], float] | None = None,
) -> FCRLossOutput:
    """Compute Phase1-only FCR reconstruction, cycle, supervision, and decorrelation losses.

    Pair-sensitive terms use only Task2's synchronized valid mask; invalid rows
    remain exact finite zeroes.  The function intentionally never consumes
    ``pair.labels`` or any hard pseudo-label, so it is usable for legal ``U_s``
    reconstruction terms without TX-label leakage.  ``domain_confusion_loss``
    is an optional caller-owned conditional/GRL-compatible loss, while
    ``probe_metric`` is a detached training-external diagnostic hook.
    """

    enabled = None if active is None else frozenset(str(name) for name in active)

    def is_enabled(name: str) -> bool:
        return enabled is None or name in enabled

    zero = _connected_zero(clean_factors.z_s)
    forward_mask = _pair_mask(pair, "clean_to_leo")
    reverse_mask = _pair_mask(pair, "leo_to_clean")
    self_loss = (
        0.5
        * (
            heteroscedastic_complex_nll(pair.clean_iq, clean_self.mu_iq, clean_self.log_variance, config)
            + heteroscedastic_complex_nll(pair.leo_iq, leo_self.mu_iq, leo_self.log_variance, config)
        )
        if is_enabled("self")
        else zero
    )
    swap_clean_to_leo = (
        _masked_nll(pair.leo_iq, clean_to_leo, forward_mask, config) if is_enabled("swap") else zero
    )
    swap_leo_to_clean = (
        _masked_nll(pair.clean_iq, leo_to_clean, reverse_mask, config) if is_enabled("swap") else zero
    )
    swap = 0.5 * (swap_clean_to_leo + swap_leo_to_clean)

    clean_zf = _unit_sphere(clean_factors.z_f_id)
    leo_zf = _unit_sphere(leo_factors.z_f_id)
    shared = (
        symmetric_stopgrad_distance(clean_factors.z_s, leo_factors.z_s)
        + symmetric_stopgrad_distance(clean_zf, leo_zf)
        if is_enabled("shared")
        else zero
    )
    anti_collapse = (
        0.5
        * (
            _anti_collapse_loss(torch.cat((clean_factors.z_s, leo_factors.z_s), dim=0))
            + _anti_collapse_loss(torch.cat((clean_zf, leo_zf), dim=0))
        )
        if is_enabled("factor")
        else zero
    )

    if is_enabled("latent_cycle") and bool(forward_mask.any()):
        cycle_clean_to_leo = _cycle_distance(
            reencode_clean_to_leo(clean_to_leo.mu_iq), clean_factors, leo_factors
        )
    else:
        cycle_clean_to_leo = _connected_zero(clean_to_leo.mu_iq)
    if is_enabled("latent_cycle") and bool(reverse_mask.any()):
        cycle_leo_to_clean = _cycle_distance(
            reencode_leo_to_clean(leo_to_clean.mu_iq), leo_factors, clean_factors
        )
    else:
        cycle_leo_to_clean = _connected_zero(leo_to_clean.mu_iq)
    latent_cycle = 0.5 * (cycle_clean_to_leo + cycle_leo_to_clean)

    eta_pred = leo_factors.z_n_parts.get("eta_pred")
    eta_target = pair.nuisance.to(device=pair.leo_iq.device)
    eta_valid = torch.as_tensor(pair.nuisance_valid, device=pair.leo_iq.device, dtype=torch.bool)
    if not is_enabled("eta") or eta_pred is None:
        eta = _connected_zero(leo_factors.z_s)
    else:
        if eta_pred.shape != eta_target.shape:
            raise ValueError("eta_pred and pair nuisance must have matching shapes")
        if eta_valid.shape != eta_target.shape:
            eta_valid = eta_valid.reshape(-1, 1).expand_as(eta_target)
        if bool(eta_valid.any()):
            eta = (eta_pred[eta_valid] - eta_target[eta_valid]).square().mean()
        else:
            eta = _connected_zero(eta_pred)

    if is_enabled("factor"):
        clean_nuisance = _nuisance_vector(clean_factors)
        leo_nuisance = _nuisance_vector(leo_factors)
        factor = 0.5 * (
            _cross_covariance_loss(clean_factors.z_s, clean_zf, clean_nuisance)
            + _cross_covariance_loss(leo_factors.z_s, leo_zf, leo_nuisance)
        )
        if domain_confusion_loss is not None:
            domain_terms = (
                domain_confusion_loss(clean_zf, pair.receiver_id.detach())
                + domain_confusion_loss(leo_zf, pair.receiver_id.detach())
            )
            factor = factor + 0.5 * domain_terms
    else:
        factor = zero

    components = {
        "self": self_loss,
        "swap": swap,
        "swap_clean_to_leo": swap_clean_to_leo,
        "swap_leo_to_clean": swap_leo_to_clean,
        "shared": shared,
        "latent_cycle": latent_cycle,
        "eta": eta,
        "factor": factor,
        "anti_collapse": anti_collapse,
    }
    total = torch.stack(tuple(components.values())).sum()
    metrics = {name: float(value.detach().cpu()) for name, value in components.items()}
    if probe_metric is not None:
        metrics["factor_probe"] = float(probe_metric((clean_factors, leo_factors), pair.receiver_id.detach()))
    return FCRLossOutput(total=total, components=components, metrics=metrics)
