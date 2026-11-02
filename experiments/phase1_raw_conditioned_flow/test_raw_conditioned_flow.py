import unittest
import torch
from raw_conditioned_flow import RawConditionedFlow

def rotate(x,theta):
 c,s=torch.cos(theta),torch.sin(theta)
 return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),1)

class RawFlowTest(unittest.TestCase):
 def test_zero_initialized_integration_is_identity(self):
  net=RawConditionedFlow();x=torch.randn(4,2,256);self.assertTrue(torch.equal(net.restore(x,steps=4),x))
 def test_global_phase_equivariance_after_nonzero_weights(self):
  torch.manual_seed(8);net=RawConditionedFlow();torch.nn.init.normal_(net.output.weight,std=.03);torch.nn.init.normal_(net.output.bias,std=.03);x=torch.randn(3,2,256);theta=torch.tensor(.77)
  torch.testing.assert_close(net.restore(rotate(x,theta),steps=4),rotate(net.restore(x,steps=4),theta),atol=2e-5,rtol=2e-5)
 def test_flow_matching_has_gradient_and_uses_observation(self):
  torch.manual_seed(9);net=RawConditionedFlow();x=torch.randn(2,2,256);clean=torch.randn_like(x);t=torch.tensor([.2,.8]);state=(1-t[:,None,None])*x+t[:,None,None]*clean;loss=(net(state,x,t)-(clean-x)).square().mean();loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(float(net.output.weight.grad.abs().sum()),0.)
 def test_velocity_can_fill_zero_sample_from_temporal_context(self):
  torch.manual_seed(10);net=RawConditionedFlow();net.output.bias.data.fill_(.3);x=torch.zeros(1,2,256);x[0,0,128]=1.;v=net(x,x,torch.zeros(1));self.assertGreater(float(v[0,:,129].detach().abs().sum()),1e-7)
 def test_invalid_steps_rejected(self):
  with self.assertRaises(ValueError):RawConditionedFlow().restore(torch.zeros(1,2,256),steps=0)

if __name__=='__main__':unittest.main()
