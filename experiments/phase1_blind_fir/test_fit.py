import pytest
import torch
from experiments.phase1_blind_fir.fit import fir_design, fit_fir_delta, reflected_taps


def _complex(shape, generator):
    return torch.complex(torch.randn(shape, generator=generator), torch.randn(shape, generator=generator))


def test_reflection_matches_reference_fir_edges():
    signal = torch.tensor([[1+0j, 2+0j, 3+0j, 4+0j]])
    taps = reflected_taps(signal, 3)
    assert taps[0, :, 2].tolist() == [3+0j, 2+0j, 1+0j, 2+0j]


def test_real_linear_fir_fit_reduces_residual_and_recovers_taps():
    g = torch.Generator().manual_seed(12)
    x, main, image = (_complex((3, 128), g), _complex((3, 128), g), .15*_complex((3, 128), g))
    kernel = _complex((3, 16), g)
    truth = .01*_complex((3, 8), g)
    basis = fir_design(x, main, image, 8)
    coeff = torch.cat((truth.real, truth.imag), dim=-1).to(basis.dtype)
    residual = torch.einsum('bnk,bk->bn', basis, coeff)
    fitted = fit_fir_delta(x, residual, main, image, kernel, count=8, ridge=1e-6, max_relative=100)
    torch.testing.assert_close(fitted, truth, atol=3e-7, rtol=1e-4)
    recon = torch.einsum('bnk,bk->bn', basis, torch.cat((fitted.real, fitted.imag), -1).to(basis.dtype))
    assert ((residual-recon).abs().square().sum(-1) < residual.abs().square().sum(-1)).all()


def test_record_independence_cap_and_finite_signal_gradient():
    g = torch.Generator().manual_seed(24)
    x = _complex((2, 64), g).requires_grad_()
    residual, main, image, kernel = (_complex((2, 64), g), _complex((2, 64), g),
                                      .1*_complex((2, 64), g), _complex((2, 8), g))
    all_delta = fit_fir_delta(x, residual, main, image, kernel, count=8, max_relative=.01)
    first = fit_fir_delta(x[:1], residual[:1], main[:1], image[:1], kernel[:1], count=8, max_relative=.01)
    torch.testing.assert_close(all_delta[:1], first, rtol=1e-5, atol=1e-7)
    assert (all_delta.abs().square().sum(-1).sqrt() <= .010001*kernel.abs().square().sum(-1).sqrt()).all()
    all_delta.abs().square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_rejects_nonfinite_and_out_of_range_taps():
    x = torch.ones((1, 8), dtype=torch.complex64)
    with pytest.raises(ValueError):
        fit_fir_delta(x, x, x, x, x, count=9)
    x[0, 0] = complex(float('nan'), 0)
    with pytest.raises(ValueError):
        fit_fir_delta(x, x, x, x, x, count=2)
