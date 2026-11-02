import math,unittest
import torch
from global_spectral_flow import GlobalSpectralFlow

def rotate(x,theta):
 c,s=torch.cos(theta),torch.sin(theta);return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),1)

class GlobalSpectralFlowTest(unittest.TestCase):
 def test_initial_flow_is_identity(self):
  net=GlobalSpectralFlow();x=torch.randn(3,2,256);self.assertTrue(torch.equal(net.restore(x,4),x))
 def test_global_phase_equivariance_with_nonzero_parameters(self):
  torch.manual_seed(23);net=GlobalSpectralFlow();torch.nn.init.normal_(net.output.weight,std=.03);torch.nn.init.normal_(net.spectral_output.weight,std=.03);torch.nn.init.normal_(net.frequency_bias,std=.03);x=torch.randn(3,2,256);theta=torch.tensor(.73)
  torch.testing.assert_close(net.restore(rotate(x,theta),4),rotate(net.restore(x,4),theta),atol=3e-5,rtol=3e-5)
 def test_frequency_path_can_move_signal_beyond_local_support(self):
  net=GlobalSpectralFlow();x=torch.zeros(1,2,256);x[0,0,128]=1.;k=torch.arange(256);net.frequency_bias.data[0,0]=.5*torch.cos(2*math.pi*k*24/256);v=net(x,x,torch.zeros(1));self.assertGreater(float(v[0,:,152].detach().abs().sum()),1e-3)
 def test_frequency_parameters_receive_gradient(self):
  torch.manual_seed(2);net=GlobalSpectralFlow();x=torch.randn(2,2,256);y=.7*x;t=torch.tensor([.3,.7]);loss=(net((1-t[:,None,None])*x+t[:,None,None]*y,x,t)-(y-x)).square().mean();loss.backward();self.assertGreater(float(net.frequency_bias.grad.abs().sum()),0.)
if __name__=='__main__':unittest.main()
