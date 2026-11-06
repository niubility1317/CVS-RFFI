import unittest

import torch

from experiments.phase1_dura_orthogonal.model import DualDomainRecovery
from experiments.phase1_dura_orthogonal.train_source import source_loss


class DualDomainTest(unittest.TestCase):
    def test_initial_identity_both_heads_receive_gradients(self):
        torch.manual_seed(31)
        model = DualDomainRecovery(width=8)
        received = torch.randn(3, 2, 256)
        clean = received + .05*torch.randn_like(received)
        output = model(received)
        self.assertTrue(torch.equal(output, received))
        source_loss(output, received, clean).backward()
        for head in (model.wave, model.spectral_head):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.)

    def test_dura_context_never_changes_direct_model_output(self):
        torch.manual_seed(32)
        model = DualDomainRecovery(width=8)
        with torch.no_grad():
            model.wave.weight.normal_(0, .01)
            model.spectral_head.weight.normal_(0, .01)
        received = torch.randn(2, 2, 256)
        first = model(received, torch.randn_like(received), torch.randn_like(received), torch.ones(2))
        second = model(received, torch.randn_like(received), torch.randn_like(received), torch.ones(2)*9)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())


if __name__ == '__main__':
    unittest.main()
