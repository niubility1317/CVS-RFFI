import torch

from experiments.pamf_unrolled_restore.wisig_rank4_observation import posterior_increment


def test_rank_four_increment_matches_dense_gaussian_conditioning():
    torch.manual_seed(14)
    residual = torch.randn(3, 8)
    diagonal = torch.rand(8) + .2
    factors = torch.randn(8, 4) * .2
    noise = torch.tensor([[.1], [.4], [2.]])
    actual = posterior_increment(residual, diagonal, factors, noise)
    covariance = torch.diag(diagonal) + factors @ factors.T
    expected = torch.stack([covariance @ torch.linalg.solve(covariance + value * torch.eye(8), row)
                            for row, value in zip(residual, noise[:, 0])])
    assert torch.allclose(actual, expected, atol=1e-5)


def test_increment_reduces_as_noise_increases():
    residual = torch.ones(2, 8)
    diagonal = torch.ones(8)
    factors = torch.zeros(8, 4)
    value = posterior_increment(residual, diagonal, factors, torch.tensor([[.01], [10.]]))
    assert value[0].abs().mean() > value[1].abs().mean()
