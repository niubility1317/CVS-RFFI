import unittest
import torch
from experiments.phase1_dynamic_volterra.model import DynamicVolterra


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), 1)


class DynamicVolterraTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(672); torch.set_num_threads(2)
        self.x = torch.randn(3, 2, 256); self.base = torch.randn_like(self.x); self.lv = torch.randn(3)

    def test_initial_model_returns_input_base(self):
        self.assertTrue(torch.equal(DynamicVolterra()(self.x, self.base, self.lv), self.base))

    def test_phase_equivariance_and_batch_independence(self):
        m = DynamicVolterra(); torch.nn.init.normal_(m.gate_output.weight, std=.1)
        torch.nn.init.normal_(m.gate_output.bias, std=.1); a = torch.tensor(.83)
        output = m(self.x, self.base, self.lv)
        torch.testing.assert_close(m(rotate(self.x,a),rotate(self.base,a),self.lv),rotate(output,a),rtol=4e-5,atol=4e-5)
        torch.testing.assert_close(m(self.x[1:2],self.base[1:2],self.lv[1:2]),output[1:2],rtol=3e-5,atol=3e-5)

    def test_two_training_steps_reach_feature_network(self):
        m = DynamicVolterra(); opt = torch.optim.AdamW(m.parameters(), lr=.003)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            (m(self.x,self.base,self.lv)-self.x).square().mean().backward()
            self.assertGreater(float(m.gate_output.weight.grad.abs().sum()), 0.)
            opt.step()
        self.assertTrue(torch.isfinite(m.entry.weight.grad).all())
        self.assertGreater(float(m.entry.weight.grad.abs().sum()), 0.)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError): DynamicVolterra()(self.x,self.base,torch.zeros(2))


if __name__ == '__main__': unittest.main()
