import torch

from experiments.phase1_dura_orthogonal.source_frequency_regression import solve, spectra


def test_conjugate_frequency_features_match_time_domain_conjugation():
    torch.manual_seed(1)
    a = torch.randn(3, 2, 256)
    x = torch.randn_like(a)
    f, af, xf, _ = spectra(x, a, a)
    expected_a = torch.fft.fft(torch.complex(a[:, 0], -a[:, 1]), dim=-1, norm='ortho')
    expected_x = torch.fft.fft(torch.complex(x[:, 0], -x[:, 1]), dim=-1, norm='ortho')
    torch.testing.assert_close(f[..., 0], af)
    torch.testing.assert_close(f[..., 1], xf)
    torch.testing.assert_close(f[..., 2], expected_a)
    torch.testing.assert_close(f[..., 3], expected_x)


def test_ridge_solver_recovers_shared_complex_residual():
    torch.manual_seed(2)
    z = torch.complex(torch.randn(5, 96, 4), torch.randn(5, 96, 4))
    truth = torch.complex(torch.randn(5, 4, 1), torch.randn(5, 4, 1))
    target = z @ truth
    gram = z.conj().transpose(1, 2) @ z
    rhs = z.conj().transpose(1, 2) @ target
    estimate = solve(gram, rhs).unsqueeze(-1)
    assert (z @ estimate - target).abs().square().mean() < 1e-4
    assert torch.equal(solve(gram, torch.zeros_like(rhs)), torch.zeros_like(truth.squeeze(-1)))
    nested = solve(torch.stack((gram, gram)), torch.stack((rhs, rhs)))
    torch.testing.assert_close(nested[0], estimate.squeeze(-1))
    torch.testing.assert_close(nested[1], estimate.squeeze(-1))
