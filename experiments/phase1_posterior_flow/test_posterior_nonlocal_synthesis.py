import unittest
import torch
from posterior_nonlocal_synthesis import PosteriorNonlocalSynthesis

def rotate(x,angle):
 c=torch.cos(torch.tensor(angle));s=torch.sin(torch.tensor(angle));return torch.stack((c*x[...,0,:]-s*x[...,1,:],s*x[...,0,:]+c*x[...,1,:]),dim=-2)

class PosteriorNonlocalSynthesisTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7);torch.set_num_threads(1);self.base=torch.randn(2,2,256);self.c=torch.randn(2,7,2,256);self.lv=torch.tensor([-.5,.3])
 def test_initial_output_is_physical_baseline(self):
  net=PosteriorNonlocalSynthesis();self.assertTrue(torch.equal(net(self.base,self.c,self.lv),self.base))
 def test_global_phase_rotation_after_nonzero_correction(self):
  net=PosteriorNonlocalSynthesis();torch.nn.init.normal_(net.coefficients.weight,std=.03);torch.nn.init.normal_(net.coefficients.bias,std=.03);a=net(self.base,self.c,self.lv);b=net(rotate(self.base,.73),rotate(self.c,.73),self.lv);self.assertTrue(torch.allclose(b,rotate(a,.73),atol=2e-5,rtol=2e-5))
 def test_can_move_signal_to_a_new_time_sample(self):
  net=PosteriorNonlocalSynthesis();base=torch.zeros(1,2,256);c=torch.zeros(1,7,2,256);c[0,0,0,128]=1
  with torch.no_grad():net.conv_real.weight.zero_();net.conv_imag.weight.zero_();net.conv_real.weight[0,1,0]=1;net.coefficients.weight.zero_();net.coefficients.bias.zero_();net.coefficients.bias[0]=1
  out=net(base,c,torch.zeros(1));self.assertGreater(abs(out[0,0,132].item()),1e-5)
 def test_learnable_residual_has_gradients(self):
  net=PosteriorNonlocalSynthesis();loss=(net(self.base,self.c,self.lv)-self.c[:,0]).square().mean();loss.backward();self.assertGreater(net.coefficients.weight.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
