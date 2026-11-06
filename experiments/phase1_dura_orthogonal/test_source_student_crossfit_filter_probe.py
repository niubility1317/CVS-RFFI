import torch

from experiments.phase1_dura_orthogonal.source_student_crossfit_filter_probe import windows, fit_filter, crossfit_metrics


def test_crossfit_learns_linear_distortion_and_has_no_heldout_fit_leakage():
    torch.manual_seed(41)
    x = torch.randn(3, 2, 256)
    matrix, _ = windows(x)
    weights = torch.zeros(3, 17, dtype=torch.complex128)
    weights[:, 8] = 0.7+0.2j
    weights[:, 10] = -0.35+0.1j
    y = (matrix @ weights.unsqueeze(-1)).squeeze(-1)
    clean = torch.stack((y.real, y.imag), 1)
    result = crossfit_metrics(x, clean)
    assert torch.all(result[:, 2] < result[:, 1]*.001)
    fitted = fit_filter(matrix, y, 8, 128)
    changed = y.clone()
    changed[:, 128:248] += 1000+1000j
    torch.testing.assert_close(fitted, fit_filter(matrix, changed, 8, 128), rtol=0, atol=0)
