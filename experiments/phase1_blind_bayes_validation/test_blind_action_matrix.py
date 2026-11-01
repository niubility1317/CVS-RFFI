import unittest,torch
try:from blind_action_matrix import action_matrix
except ImportError:action_matrix=None
class ActionTests(unittest.TestCase):
 def test_identity_and_reflected_delay(self):
  self.assertIsNotNone(action_matrix);h=torch.zeros(2,2,64);h[0,0,0]=1;h[1,0,2]=2;h[1,1,2]=3;a=action_matrix(h);self.assertEqual(a.shape,(2,512,512));self.assertTrue(torch.equal(a[0],torch.eye(512)));x=torch.arange(256,dtype=torch.float32);z=torch.cat((x,torch.zeros_like(x)));y=a[1]@z;expected=x[torch.abs(torch.arange(256)-2)];self.assertTrue(torch.equal(y[:256],2*expected));self.assertTrue(torch.equal(y[256:],3*expected))
 def test_batch_linearity(self):
  torch.manual_seed(7);h=torch.randn(3,2,64);x=torch.randn(3,512);mat=action_matrix(h);self.assertTrue(torch.allclose(torch.bmm(mat,(2*x).unsqueeze(-1)).squeeze(-1),2*torch.bmm(mat,x.unsqueeze(-1)).squeeze(-1),rtol=1e-6,atol=1e-6))
if __name__=='__main__':unittest.main()
