import unittest

import torch

from experiments.phase1_dura_orthogonal.train_source import grouped_view_indices, crossview_dispersion


class CrossViewTrainingTest(unittest.TestCase):
    def test_view_major_indices_preserve_physical_pairing(self):
        actual = grouped_view_indices(torch.tensor([1, 3]), fit_count=5)
        self.assertTrue(torch.equal(actual, torch.tensor([1, 3, 6, 8, 11, 13])))

    def test_dispersion_is_zero_for_agreement_and_has_finite_gradient(self):
        clean = torch.ones(6, 2, 256)
        torch.testing.assert_close(crossview_dispersion(clean, clean), torch.tensor(0.))
        offsets = torch.tensor([-1., 0., 1.]).repeat_interleave(2)[:, None, None]
        output = (clean+offsets).requires_grad_()
        loss = crossview_dispersion(output, clean)
        torch.testing.assert_close(loss, torch.tensor(2/3))
        loss.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(float(output.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
