from __future__ import annotations

import math

import torch

from cvsrffi.phase1_fcr_types import FCRConfig


def _as_iq(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((value.real, value.imag), dim=1).to(torch.float32)


def _as_complex(value: torch.Tensor) -> torch.Tensor:
    return torch.complex(value[:, 0], value[:, 1])


def _common_effect(
    clean: torch.Tensor, *, gain: float, phase0: float, omega: float
) -> torch.Tensor:
    sample_index = torch.arange(clean.size(-1), dtype=clean.real.dtype)
    phase = phase0 + omega * sample_index
    return gain * clean * torch.exp(1j * phase)


def _complex_nmse(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (value - reference).abs().square().mean() / reference.abs().square().mean()


def test_canonicalizer_reduces_common_gain_phase_and_cfo_error() -> None:
    """Removing the three common terms must improve a unit-amplitude carrier."""

    from cvsrffi.phase1_fcr_canonicalizer import ConservativeCanonicalizer

    config = FCRConfig()
    clean = torch.ones(3, config.input_len, dtype=torch.complex64)
    perturbed = _common_effect(clean, gain=1.4, phase0=0.5, omega=0.03)

    output = ConservativeCanonicalizer(config)(_as_iq(perturbed))

    assert output.canonical_iq.shape == (3, 2, config.input_len)
    assert output.eta_hat.shape == (3, 3)
    assert _complex_nmse(_as_complex(output.canonical_iq), clean) < _complex_nmse(
        perturbed, clean
    )
    assert all(torch.isfinite(value).all() for value in output.quality.values())


def test_canonicalizer_keeps_noncommon_iq_imbalance_in_residual() -> None:
    """A conjugate fine residual must not be erased by common normalization."""

    from cvsrffi.phase1_fcr_canonicalizer import ConservativeCanonicalizer

    config = FCRConfig()
    sample_index = torch.arange(config.input_len, dtype=torch.float32)
    clean = torch.exp(1j * (0.14 * sample_index + 0.003 * sample_index.square()))
    fine_imbalance = clean + 0.18 * clean.conj()
    output = ConservativeCanonicalizer(config)(_as_iq(fine_imbalance.unsqueeze(0)))

    residual_energy = _as_complex(output.residual_iq).abs().square().mean()
    assert residual_energy > 1.0e-4
    assert math.isfinite(float(residual_energy))
