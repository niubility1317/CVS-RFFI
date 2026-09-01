from __future__ import annotations

import math

import torch

from cvsrffi.identifiability_stats import (
    complex_excitation_stats,
    effective_fisher_summary,
    hos_confidence_stats,
    memory_polynomial_gram_stats,
    phase_residual_stats,
    spectral_occupancy_stats,
)


def _complex(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.complex64).unsqueeze(0)


def test_effective_fisher_removes_a_collinear_nuisance_direction() -> None:
    nuisance = torch.tensor([[[1.0], [0.0], [0.0]]])
    collinear_target = torch.tensor([[[1.0], [0.0], [0.0]]])
    orthogonal_target = torch.tensor([[[0.0], [1.0], [0.0]]])
    weight = torch.ones(1, 3)

    collinear = effective_fisher_summary(
        collinear_target, nuisance, weight=weight, eps=1e-6
    )
    orthogonal = effective_fisher_summary(
        orthogonal_target, nuisance, weight=weight, eps=1e-6
    )

    assert collinear["lambda_min"].item() < 1e-4
    assert orthogonal["lambda_min"].item() > 0.99
    assert orthogonal["effective_rank"].item() > 0.99


def test_complex_excitation_separates_real_and_circular_symbol_geometry() -> None:
    bpsk = _complex([1, -1, 1, -1, 1, -1, 1, -1])
    qpsk = _complex([1, 1j, -1, -1j, 1, 1j, -1, -1j])

    bpsk_stats = complex_excitation_stats(bpsk)
    qpsk_stats = complex_excitation_stats(qpsk)

    torch.testing.assert_close(bpsk_stats["rho"], torch.ones(1), atol=1e-6, rtol=0.0)
    assert qpsk_stats["rho"].item() < 1e-6
    assert bpsk_stats["iq_lambda_min"].item() < qpsk_stats["iq_lambda_min"].item()


def test_memory_polynomial_gram_exposes_rank_and_is_scale_normalized() -> None:
    phase = torch.linspace(0.0, 2.0 * math.pi, 96)
    constant = torch.exp(1j * phase).unsqueeze(0)
    amplitude = torch.linspace(0.2, 1.8, 96)
    rich = (amplitude * torch.exp(1j * phase)).unsqueeze(0)

    constant_stats = memory_polynomial_gram_stats(
        constant, order=(1, 3, 5), memory_depth=1
    )
    rich_stats = memory_polynomial_gram_stats(
        rich, order=(1, 3, 5), memory_depth=1
    )
    scaled_stats = memory_polynomial_gram_stats(
        rich * 7.0, order=(1, 3, 5), memory_depth=1
    )

    assert constant_stats["effective_rank"].item() < 1.2
    assert rich_stats["effective_rank"].item() > constant_stats["effective_rank"].item() + 0.2
    torch.testing.assert_close(
        rich_stats["gram_eigenvalues"],
        scaled_stats["gram_eigenvalues"],
        atol=2e-5,
        rtol=2e-5,
    )
    assert rich_stats["papr"].item() > constant_stats["papr"].item()
    assert rich_stats["amplitude_entropy"].item() > constant_stats["amplitude_entropy"].item()


def test_spectral_occupancy_distinguishes_tone_from_broadband_content() -> None:
    length = 128
    t = torch.arange(length, dtype=torch.float32)
    tone = torch.exp(1j * 2.0 * math.pi * 7.0 * t / length).unsqueeze(0)
    generator = torch.Generator().manual_seed(17)
    broadband = torch.complex(
        torch.randn(1, length, generator=generator),
        torch.randn(1, length, generator=generator),
    )

    tone_stats = spectral_occupancy_stats(tone)
    broadband_stats = spectral_occupancy_stats(broadband)

    assert tone_stats["effective_bandwidth"].item() < broadband_stats["effective_bandwidth"].item()
    assert tone_stats["spectral_entropy"].item() < broadband_stats["spectral_entropy"].item()


def test_phase_projection_removes_low_order_nuisance_but_not_cycle_slip() -> None:
    t = torch.linspace(-1.0, 1.0, 128)
    phase = 0.4 + 0.7 * t + 0.3 * t.square()
    clean = torch.exp(1j * phase).unsqueeze(0)
    slipped = clean.clone()
    slipped[:, 64:] *= -1.0

    clean_stats = phase_residual_stats(clean, polynomial_order=2)
    slipped_stats = phase_residual_stats(slipped, polynomial_order=2)

    assert clean_stats["residual_rms"].item() < 1e-3
    assert slipped_stats["residual_rms"].item() > 0.2
    assert slipped_stats["stability"].item() < clean_stats["stability"].item()


def test_hos_confidence_penalizes_segment_to_segment_instability() -> None:
    base = _complex([1, -1, 1j, -1j] * 8)
    stable = base.repeat(1, 4)
    unstable = torch.cat(
        [base * scale for scale in (0.5, 1.0, 2.0, 3.0)], dim=1
    )
    segment_ids = torch.arange(4).repeat_interleave(base.size(1)).unsqueeze(0)

    stable_stats = hos_confidence_stats(stable, segment_ids=segment_ids)
    unstable_stats = hos_confidence_stats(unstable, segment_ids=segment_ids)

    assert stable_stats["segment_variance"].item() < 1e-6
    assert unstable_stats["segment_variance"].item() > stable_stats["segment_variance"].item()
    assert unstable_stats["confidence"].item() < stable_stats["confidence"].item()
