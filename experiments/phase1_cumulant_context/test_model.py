import unittest
import torch
from experiments.phase1_cumulant_context.model import cumulant_features, cumulant_loss, CumulantContextRecovery

class CumulantContextTest(unittest.TestCase):
    def test_invariance_to_offset_and_global_complex_gain(self):
        torch.manual_seed(1)
        x = torch.randn(3, 2, 256)
        z = torch.complex(x[:, 0], x[:, 1]) * complex(1.3, .7) + complex(.2, -.4)
        y = torch.stack((z.real, z.imag), 1)
        torch.testing.assert_close(cumulant_features(x), cumulant_features(y), rtol=1e-4, atol=1e-5)

    def test_loss_finite_nonzero_gradient(self):
        torch.manual_seed(2)
        clean = torch.randn(2, 2, 256)
        output = (clean + .1*torch.randn_like(clean)).requires_grad_()
        loss = cumulant_loss(output, clean)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(output.grad.abs().sum().item(), 0)
        self.assertEqual(cumulant_loss(clean, clean).item(), 0)

    def test_initial_output_is_exact_dura(self):
        torch.manual_seed(3)
        model = CumulantContextRecovery(width=16, layers=1)
        y, anchor, direction = [torch.randn(2, 2, 256) for _ in range(3)]
        result = model(y, anchor, direction, torch.ones(2))
        self.assertTrue(torch.equal(result, anchor))
        result.square().mean().backward()
        self.assertGreater(model.out.weight.grad.abs().sum().item(), 0)

    def test_zero_lag_pairings_coincide(self):
        torch.manual_seed(4)
        x=torch.randn(3,2,256)
        values=cumulant_features(x)
        torch.testing.assert_close(values[:,0],values[:,1],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(values[:,2],torch.zeros_like(values[:,2]),rtol=0,atol=1e-6)

    def test_constant_signal_is_finite(self):
        x = torch.ones(2, 2, 256)
        self.assertTrue(torch.isfinite(cumulant_features(x)).all())

if __name__ == '__main__': unittest.main()
