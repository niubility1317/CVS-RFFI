import unittest
import torch
from experiments.phase1_dura_iterative_geometry.objective import relational_geometry_loss


class RelationalGeometryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.clean = torch.tensor([[1., 0., 0.], [.6, .8, 0.], [0., 0., 1.]], requires_grad=True)
        self.groups = torch.tensor([4, 4, 8])

    def test_matching_features_produce_zero_geometry_loss(self):
        output = self.clean.detach().clone().requires_grad_()
        loss = relational_geometry_loss(output, self.clean, self.groups)
        self.assertAlmostEqual(float(loss.detach()), 0., places=6)
        loss.backward()
        self.assertIsNone(self.clean.grad)
        self.assertTrue(torch.isfinite(output.grad).all())

    def test_misassigned_features_are_penalized_and_differentiable(self):
        output = self.clean.detach()[torch.tensor([2, 0, 1])].clone().requires_grad_()
        loss = relational_geometry_loss(output, self.clean, self.groups)
        self.assertGreater(float(loss.detach()), .01)
        loss.backward()
        self.assertGreater(float(output.grad.abs().sum()), 0.)
        self.assertIsNone(self.clean.grad)

    def test_permutation_of_batch_preserves_loss(self):
        output = torch.tensor([[.7, .3, 0.], [.8, .1, .1], [0., .1, .9]])
        order = torch.tensor([2, 0, 1])
        a = relational_geometry_loss(output, self.clean, self.groups)
        b = relational_geometry_loss(output[order], self.clean[order], self.groups[order])
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_single_example_has_no_contrast(self):
        out = torch.tensor([[0., 1., 0.]], requires_grad=True)
        loss = relational_geometry_loss(out, self.clean[:1], self.groups[:1])
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(out.grad, torch.zeros_like(out)))

    def test_invalid_and_nonfinite_features_fail(self):
        with self.assertRaises(ValueError):
            relational_geometry_loss(torch.zeros(3, 4), self.clean, self.groups)
        with self.assertRaises(FloatingPointError):
            relational_geometry_loss(torch.full((3, 3), float('nan')), self.clean, self.groups)


if __name__ == '__main__':
    unittest.main()
