import pytest
import torch

from experiments.pamf_phase1_flow.source_fir_oracle import fit_paired_fir


def test_recovers_synthetic_widely_linear_iq_distortion():
    torch.manual_seed(6)
    received = torch.randn(3, 2, 256)
    x = torch.complex(received[:, 0], received[:, 1])
    desired = x + .3j * x.conj()
    clean = torch.stack((desired.real, desired.imag), dim=1)
    recovered = fit_paired_fir(received, clean, taps=1, widely_linear=True, ridge=1e-6)
    assert (recovered-clean).square().mean() < 1e-9
    restricted = fit_paired_fir(received, clean, taps=1, widely_linear=False, ridge=1e-6)
    assert (restricted-clean).square().mean() > .01


def test_finite_zero_and_input_validation():
    x = torch.zeros(2, 2, 256)
    y = fit_paired_fir(x, x, taps=3, widely_linear=True, ridge=1e-4)
    assert torch.equal(y, x) and torch.isfinite(y).all()
    with pytest.raises(ValueError):
        fit_paired_fir(x, x, taps=2, widely_linear=True, ridge=1e-4)
