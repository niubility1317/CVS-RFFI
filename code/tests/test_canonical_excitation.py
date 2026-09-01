from __future__ import annotations

import inspect
import math

import pytest
import torch

from cvsrffi.canonical_excitation import (
    AnalyticCanonicalizer,
    CanonicalExcitationEstimator,
    ContentExcitationEstimator,
    NuisanceEstimator,
    unique_physical_sample_mask,
)


def _qpsk(length: int) -> torch.Tensor:
    values = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j])
    generator = torch.Generator().manual_seed(23)
    indices = torch.randint(0, 4, (length,), generator=generator)
    return values[indices].unsqueeze(0).to(torch.complex64)


def _delay_with_zeros(z: torch.Tensor, shift: int) -> torch.Tensor:
    out = torch.zeros_like(z)
    out[:, shift:] = z[:, :-shift]
    return out


def test_reference_aided_canonicalization_recovers_gain_phase_cfo_and_shift() -> None:
    length = 128
    reference = _qpsk(length)
    shift = 3
    gain = 2.5
    phase0 = 0.7
    omega = 0.025
    time = torch.arange(length, dtype=torch.float32)
    received = _delay_with_zeros(reference, shift)
    received = gain * received * torch.exp(1j * (phase0 + omega * time)).unsqueeze(0)

    estimator = NuisanceEstimator(max_time_shift=6)
    estimate = estimator(received, reference_iq=reference)
    canonical, valid = AnalyticCanonicalizer()(received, estimate)
    canonical_complex = torch.complex(canonical[:, 0], canonical[:, 1])

    assert estimate.time_shift.item() == shift
    torch.testing.assert_close(estimate.log_gain.exp(), torch.tensor([gain]), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(estimate.normalized_cfo, torch.tensor([omega]), atol=2e-4, rtol=0.0)
    phase_error = torch.atan2(
        torch.sin(estimate.phase0 - phase0), torch.cos(estimate.phase0 - phase0)
    ).abs()
    assert phase_error.item() < 2e-3
    nmse = ((canonical_complex - reference).abs().square() * valid).sum() / (
        reference.abs().square() * valid
    ).sum()
    assert nmse.item() < 1e-5


def test_content_estimator_returns_detached_excitation_and_bounded_confidence() -> None:
    canonical = torch.stack([_qpsk(64).real, _qpsk(64).imag], dim=1).requires_grad_()
    estimator = ContentExcitationEstimator(detach_gate_input=True)
    output = estimator(canonical)

    assert output.s_hat.shape == (1, 64)
    assert output.content_confidence.shape == (1, 64)
    assert output.s_hat.requires_grad is False
    assert torch.isfinite(output.s_hat).all()
    assert torch.all((output.content_confidence >= 0.0) & (output.content_confidence <= 1.0))
    assert output.uncertainty.item() < 0.1


def test_zero_information_input_falls_back_to_high_uncertainty_without_nan() -> None:
    model = CanonicalExcitationEstimator(max_time_shift=4, detach_gate_input=True)
    output = model(torch.zeros(2, 2, 48))

    assert torch.isfinite(output.s_hat).all()
    assert torch.isfinite(output.reconstruction_nmse).all()
    assert torch.all(output.content_confidence <= 0.05)
    assert torch.all(output.uncertainty >= 0.95)
    assert torch.equal(output.valid_mask, torch.ones_like(output.valid_mask))


def test_canonical_excitation_api_cannot_receive_tx_or_query_labels() -> None:
    signature = inspect.signature(CanonicalExcitationEstimator.forward)
    assert "tx_labels" not in signature.parameters
    assert "query_labels" not in signature.parameters
    with pytest.raises(TypeError):
        CanonicalExcitationEstimator()(torch.zeros(1, 2, 32), tx_labels=torch.tensor([1]))


def test_physical_sample_deduplication_does_not_count_views_as_new_bursts() -> None:
    mask = unique_physical_sample_mask(["sample-a", "sample-a", "sample-b", "sample-a"])
    assert mask.tolist() == [True, False, True, False]
    assert int(mask.sum()) == 2
