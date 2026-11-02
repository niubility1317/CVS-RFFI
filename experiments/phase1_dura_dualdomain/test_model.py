import math
import unittest
import torch
from experiments.phase1_dura_dualdomain.model import DualDomainRecovery

def rotate(x, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), dim=1)

class DualDomainTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(917)
        self.model = DualDomainRecovery().eval()
        self.x, self.anchor, self.lv = torch.randn(3,2,256), torch.randn(3,2,256), torch.randn(3)

    def test_zero_initialized_output_equals_dura(self):
        torch.testing.assert_close(self.model(self.x,self.anchor,self.lv),self.anchor,rtol=0,atol=0)

    def test_phase_equivariance_with_nonzero_residual(self):
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0,0.03)
            self.model.gate_output.bias.normal_(0,0.03)
            y=self.model(self.x,self.anchor,self.lv)
            rotated=self.model(rotate(self.x,0.73),rotate(self.anchor,0.73),self.lv)
        torch.testing.assert_close(rotated,rotate(y,0.73),rtol=2e-5,atol=2e-5)

    def test_batch_independence(self):
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0,0.03)
            a=self.model(self.x,self.anchor,self.lv)
            b=self.model(self.x[:1],self.anchor[:1],self.lv[:1])
        torch.testing.assert_close(a[:1],b,rtol=2e-5,atol=2e-5)

    def test_both_paths_learn_after_initial_step(self):
        opt=torch.optim.SGD(self.model.parameters(),lr=0.1)
        for _ in range(2):
            opt.zero_grad()
            loss=(self.model(self.x,self.anchor,self.lv)-self.x).square().mean()
            loss.backward()
            opt.step()
        for layer in (self.model.time_entry,self.model.freq_entry,self.model.gate_output):
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(layer.weight.grad.abs().sum().item(),0)

    def test_nonfinite_rejected(self):
        self.x[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):
            self.model(self.x,self.anchor,self.lv)

if __name__ == '__main__':
    unittest.main()
