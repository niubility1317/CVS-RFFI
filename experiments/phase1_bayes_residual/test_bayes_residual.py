import unittest,torch
try:from bayes_residual_model import ResidualCorrector,backproject,hybrid_loss
except ImportError:ResidualCorrector=None
class ResidualTests(unittest.TestCase):
 def test_identity_backprojection_and_initial_output(self):
  self.assertIsNotNone(ResidualCorrector);torch.manual_seed(7);x=torch.randn(2,2,256);z=torch.randn(2,2,256);A=torch.eye(512).repeat(2,1,1);b=backproject(x,z,A);self.assertTrue(torch.allclose(b,x-z,atol=1e-6));net=ResidualCorrector();self.assertTrue(torch.equal(net(z,b),z))
 def test_gradient_scaling_and_hybrid_label_mask(self):
  torch.manual_seed(3);net=ResidualCorrector();z=torch.randn(4,2,256);b=torch.randn_like(z);y=torch.randn_like(z);loss=(net(z,b)-y).square().mean();loss.backward();self.assertTrue(torch.isfinite(net.output.weight.grad).all());self.assertGreater(net.output.weight.grad.abs().sum().item(),0.);a=net(z,b);c=net(2*z,2*b);self.assertTrue(torch.allclose(c,2*a,atol=1e-6,rtol=1e-6));student=torch.randn(4,6,requires_grad=True);teacher=torch.softmax(torch.randn(4,6)/2,1);labels=torch.tensor([1,-1,-1,4]);total=hybrid_loss(student,teacher,labels);total.backward();self.assertTrue(torch.isfinite(student.grad).all())
if __name__=='__main__':unittest.main()
