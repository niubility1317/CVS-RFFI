import unittest

import torch

from experiments.pamf_lora_main.shared_agreement_loss_run import agreement_anchor_kl


class AgreementAnchorLossTest(unittest.TestCase):
    def test_anchor_loss_only_backpropagates_for_teacher_agreement(self):
        clean = torch.tensor([[4., 0.], [4., 0.]], requires_grad=True)
        anchor = torch.tensor([[3., 0.], [0., 3.]], requires_grad=True)
        output = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
        loss, agreement = agreement_anchor_kl(clean, anchor, output)
        self.assertEqual(agreement.tolist(), [True, False])
        loss.backward()
        self.assertGreater(float(output.grad[0].abs().sum()), 0.)
        self.assertEqual(float(output.grad[1].abs().sum()), 0.)
        self.assertIsNone(clean.grad)
        self.assertIsNone(anchor.grad)

    def test_no_agreement_is_finite_zero(self):
        clean = torch.tensor([[4., 0.]])
        anchor = torch.tensor([[0., 4.]])
        output = torch.tensor([[1., 0.]], requires_grad=True)
        loss, agreement = agreement_anchor_kl(clean, anchor, output)
        self.assertFalse(bool(agreement.any()))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(output.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
