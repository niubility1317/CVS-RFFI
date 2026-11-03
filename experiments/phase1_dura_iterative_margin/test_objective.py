import unittest
import torch
from experiments.phase1_dura_iterative_margin.objective import relative_margin_penalty


class RelativeMarginPenaltyTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.teacher = torch.tensor([[10., 0., 0., 0., 0., 0.]]).repeat(3, 1)
        self.anchor = torch.tensor([[3., 0., 0., 0., 0., 0.], [0., 2., 0., 0., 0., 0.],
                                    [3., 0., 0., 0., 0., 0.]], requires_grad=True)
        self.groups = torch.tensor([4, 4, 8])

    def test_correct_anchor_does_not_penalize_preserved_margin(self):
        teacher = self.teacher[:1]; anchor = self.anchor[:1]
        output = torch.tensor([[2.5, 0., 0., 0., 0., 0.]], requires_grad=True)
        loss = relative_margin_penalty(output, anchor, teacher, self.groups[:1])
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(output.grad, torch.zeros_like(output)))
        self.assertIsNone(self.anchor.grad)

    def test_reduced_margin_and_wrong_anchor_get_recovery_gradients(self):
        output = torch.tensor([[1., 0., 0., 0., 0., 0.], [0., 2., 0., 0., 0., 0.]], requires_grad=True)
        teacher = self.teacher[:2].clone().requires_grad_()
        loss = relative_margin_penalty(output, self.anchor[:2], teacher, self.groups[:2])
        self.assertGreater(float(loss.detach()), 0.)
        loss.backward()
        self.assertLess(float(output.grad[0, 0]), 0.)
        self.assertLess(float(output.grad[1, 0]), 0.)
        self.assertIsNone(self.anchor.grad)
        self.assertIsNone(teacher.grad)

    def test_uncertain_pseudo_target_has_no_effect(self):
        output = torch.tensor([[0., 3., 0., 0., 0., 0.]], requires_grad=True)
        loss = relative_margin_penalty(output, self.anchor[:1], torch.zeros(1, 6), self.groups[:1])
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(output.grad, torch.zeros_like(output)))

    def test_receiver_groups_are_equally_weighted(self):
        output = torch.tensor([[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0.]], requires_grad=True)
        anchor = torch.tensor([[3., 0., 0., 0., 0., 0.]]).repeat(3, 1)
        loss = relative_margin_penalty(output, anchor, self.teacher, self.groups)
        self.assertAlmostEqual(float(loss.detach()), 1.5, places=6)

    def test_invalid_or_nonfinite_logits_fail(self):
        output = torch.zeros(1, 5, requires_grad=True)
        with self.assertRaises(ValueError):
            relative_margin_penalty(output, self.anchor[:1], self.teacher[:1], self.groups[:1])
        output = torch.full((1, 6), float('nan'), requires_grad=True)
        with self.assertRaises(FloatingPointError):
            relative_margin_penalty(output, self.anchor[:1], self.teacher[:1], self.groups[:1])


if __name__ == '__main__':
    unittest.main()
