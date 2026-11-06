import torch
from experiments.phase1_dura_orthogonal.source_covariance_posterior import posterior_cg, shrunk_second_moment


class MatrixAction:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, value):
        real = torch.cat((value.real, value.imag), 1) @ self.matrix.T
        return torch.complex(real[:, :2], real[:, 2:])

    def adjoint(self, value):
        real = torch.cat((value.real, value.imag), 1) @ self.matrix
        return torch.complex(real[:, :2], real[:, 2:])


def test_posterior_matches_dense_solution():
    torch.manual_seed(9)
    raw = torch.randn(4, 4, dtype=torch.float64)
    covariance = raw @ raw.T + .4*torch.eye(4, dtype=torch.float64)
    matrix = torch.tensor([[1., .1, .2, 0.], [.2, .9, 0., .1], [0., .1, 1.1, .2],
                           [.1, 0., .2, .8]], dtype=torch.float64)
    observed = torch.randn(3, 2, 2, dtype=torch.float64)
    variance = torch.tensor([.2, .3, .4], dtype=torch.float64)
    output, residual = posterior_cg(MatrixAction(matrix), observed, covariance, variance, steps=64, rtol=1e-12)
    flat = observed.reshape(3, 4)
    expected = torch.stack([covariance @ matrix.T @ torch.linalg.solve(
        matrix @ covariance @ matrix.T + variance[i]*torch.eye(4, dtype=torch.float64), flat[i])
        for i in range(len(flat))])
    torch.testing.assert_close(output.reshape(3, 4), expected, atol=1e-10, rtol=1e-10)
    assert (residual < 1e-10).all()


def test_shrinkage_preserves_psd_and_scale():
    value = torch.tensor([[1., 0.], [0., 2.]], dtype=torch.float64)
    covariance = shrunk_second_moment(value, shrinkage=.1)
    torch.testing.assert_close(covariance, covariance.T)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    torch.testing.assert_close(covariance.trace(), torch.tensor(3., dtype=torch.float64))
