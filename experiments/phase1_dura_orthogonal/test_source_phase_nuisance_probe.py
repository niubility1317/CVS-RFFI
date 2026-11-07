import torch

from experiments.phase1_dura_orthogonal.source_phase_nuisance_probe import evaluate_batch


def test_identity_has_equivariance_but_not_invariance():
    x = torch.randn(3, 2, 256)
    values = evaluate_batch(torch.nn.Identity(), x, x)
    torch.testing.assert_close(values[:, :, 1], torch.zeros_like(values[:, :, 1]))
    torch.testing.assert_close(values[:, :, 3], torch.zeros_like(values[:, :, 3]))
    torch.testing.assert_close(values[:, 1, 0], torch.full((3,), 2.0))
    torch.testing.assert_close(values[:, 1, 2], torch.full((3,), 2.0))


def test_fixed_output_has_invariance_but_not_equivariance():
    x = torch.randn(3, 2, 256)

    class Fixed:
        def __call__(self, value):
            return x

    values = evaluate_batch(Fixed(), x, x)
    torch.testing.assert_close(values[:, :, 0], torch.zeros_like(values[:, :, 0]))
    torch.testing.assert_close(values[:, :, 2], torch.zeros_like(values[:, :, 2]))
    torch.testing.assert_close(values[:, 1, 1], torch.full((3,), 2.0))
    torch.testing.assert_close(values[:, 1, 3], torch.full((3,), 2.0))
