import torch

from experiments.pamf_phase1_flow.source_fir_crossfit import crossfit_paired_fir
from experiments.pamf_phase1_flow.filter_student import project_delta


def test_recovers_stationary_complex_and_conjugate_distortion_out_of_sample():
    torch.manual_seed(9)
    anchor = torch.randn(3, 2, 256)
    x = torch.complex(anchor[:, 0], anchor[:, 1])
    clean_complex = (1+.1j)*x + .2*x.conj()
    clean = torch.stack((clean_complex.real, clean_complex.imag), dim=1)
    fitted = crossfit_paired_fir(anchor, clean, taps=1, ridge=1e-6)
    assert (fitted-clean).square().mean() < 1e-9


def test_fit_on_one_half_never_uses_clean_values_from_the_half_it_predicts():
    torch.manual_seed(7)
    anchor = torch.randn(2, 2, 256)
    clean = anchor*.9
    result = crossfit_paired_fir(anchor, clean, taps=3, ridge=1e-4)
    changed = clean.clone()
    changed[:, :, 128:] += torch.randn_like(changed[:, :, 128:])*5
    rerun = crossfit_paired_fir(anchor, changed, taps=3, ridge=1e-4)
    torch.testing.assert_close(result[:, :, 128:], rerun[:, :, 128:], rtol=0, atol=0)
    right = project_delta(anchor[:, :, 128:], result[:, :, 128:], epsilon=.2)
    right_again = project_delta(anchor[:, :, 128:], rerun[:, :, 128:], epsilon=.2)
    torch.testing.assert_close(right, right_again, rtol=0, atol=0)
    assert not torch.allclose(result[:, :, :128], rerun[:, :, :128])
    changed = clean.clone()
    changed[:, :, :128] += torch.randn_like(changed[:, :, :128])*5
    rerun = crossfit_paired_fir(anchor, changed, taps=3, ridge=1e-4)
    torch.testing.assert_close(result[:, :, :128], rerun[:, :, :128], rtol=0, atol=0)
    assert not torch.allclose(result[:, :, 128:], rerun[:, :, 128:])
