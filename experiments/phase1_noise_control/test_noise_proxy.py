import unittest
import torch
from noise_proxy import relative_logvar

class NoiseProxyTest(unittest.TestCase):
 def test_power_normalization_and_scaling(self):
  x=torch.tensor([[[1.,2.],[2.,1.]],[[3.,4.],[0.,0.]]]);logvar=torch.tensor([-1.,2.]);p=x.square().sum(1).mean(1);self.assertTrue(torch.allclose(relative_logvar(x,logvar),logvar-p.log()));a=3.7;self.assertTrue(torch.allclose(relative_logvar(a*x,logvar+2*torch.log(torch.tensor(a))),relative_logvar(x,logvar),atol=1e-6))
 def test_rejects_incompatible_shapes(self):
  with self.assertRaises(ValueError):relative_logvar(torch.zeros(2,2,256),torch.zeros(3))
if __name__=='__main__':unittest.main()
