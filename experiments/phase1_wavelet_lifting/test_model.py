import math
import unittest
import torch
from experiments.phase1_wavelet_lifting.model import WaveletLifting, split_haar, join_haar

def rotate(x,angle):
    c,s=math.cos(angle),math.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),dim=1)

class WaveletLiftingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(543)
        self.model=WaveletLifting().eval()
        self.x=torch.randn(3,2,256);self.base=torch.randn(3,2,256);self.lv=torch.randn(3)

    def test_haar_reconstructs_and_preserves_energy(self):
        z=torch.complex(self.x[:,0],self.x[:,1])
        low,high=split_haar(z)
        torch.testing.assert_close(join_haar(low,high),z,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(low.abs().square().sum()+high.abs().square().sum(),z.abs().square().sum(),rtol=1e-6,atol=1e-5)

    def test_zero_init_returns_input_anchor(self):
        torch.testing.assert_close(self.model(self.x,self.base,self.lv),self.base,rtol=0,atol=0)

    def test_global_phase_equivariance(self):
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0,.03);self.model.gate_output.bias.normal_(0,.03)
            y=self.model(self.x,self.base,self.lv)
            ry=self.model(rotate(self.x,.71),rotate(self.base,.71),self.lv)
        torch.testing.assert_close(ry,rotate(y,.71),rtol=2e-5,atol=2e-5)

    def test_batch_independent(self):
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0,.03)
            a=self.model(self.x,self.base,self.lv)
            b=self.model(self.x[:1],self.base[:1],self.lv[:1])
        torch.testing.assert_close(a[:1],b,rtol=2e-5,atol=2e-5)

    def test_learns_multiscale_entry(self):
        opt=torch.optim.AdamW(self.model.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad();loss=(self.model(self.x,self.base,self.lv)-self.x).square().mean();loss.backward();opt.step()
        for layer in (self.model.entry,self.model.gate_output):
            self.assertIsNotNone(layer.weight.grad);self.assertGreater(layer.weight.grad.abs().sum().item(),0)

    def test_rejects_nonfinite(self):
        self.x[0,0,0]=float('inf')
        with self.assertRaises(FloatingPointError):self.model(self.x,self.base,self.lv)

if __name__=='__main__':unittest.main()
