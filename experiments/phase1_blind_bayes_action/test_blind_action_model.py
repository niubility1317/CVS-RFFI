import unittest,torch
try:from blind_action_model import BlindActionNet,effective_kernel,loss
except ImportError:BlindActionNet=None
class BlindActionTests(unittest.TestCase):
 def test_effective_complex_kernel(self):
  self.assertIsNotNone(BlindActionNet);h=torch.zeros(1,2,64);h[0,0,24]=2;h[0,1,24]=3;c=torch.zeros(1,4,256);c[:,0,:]=4;c[:,1,:]=5;target=effective_kernel(h,c,torch.tensor([2.]));self.assertEqual(target[0,0,24],-3.5);self.assertEqual(target[0,1,24],11.)
 def test_output_gradient_and_scale(self):
  torch.manual_seed(31);model=BlindActionNet();x=torch.randn(4,2,256);p,v=model(x);p2,v2=model(x*2);self.assertEqual(p.shape,(4,2,64));self.assertEqual(v.shape,(4,));self.assertTrue(torch.allclose(p2,p*2,atol=1e-5,rtol=1e-5));self.assertTrue(torch.allclose(v2,v+torch.log(torch.tensor(4.)),atol=1e-5,rtol=1e-5));h=torch.randn(4,2,64);c=torch.randn(4,4,256);variance=torch.full((4,),.05);l=loss(model,x,h,c,variance);l.backward();self.assertTrue(torch.isfinite(l));self.assertGreater(model.head[-1].weight.grad.abs().sum().item(),0.)
if __name__=='__main__':unittest.main()
