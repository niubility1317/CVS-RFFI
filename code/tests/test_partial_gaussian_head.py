import math
import unittest
import torch
from cvsrffi.partial_gaussian_head import gaussian_scores, lowrank_gaussian_scores


class GaussianTests(unittest.TestCase):
    def test_missing_is_not_zero_and_not_precision_slice(self):
        z = torch.tensor([[0., float('nan')]], dtype=torch.double)
        means = torch.tensor([[0.,0.],[3.,0.],[0.,3.]], dtype=torch.double)
        cov = torch.tensor([[[.25,.20],[.20,.25]]], dtype=torch.double)
        out = gaussian_scores(z, means, cov, torch.tensor([[True,False]]))
        self.assertTrue(torch.allclose(out['mahalanobis'], torch.tensor([[0.,36.,0.]],dtype=torch.double)))
        self.assertAlmostEqual(out['scores'].softmax(-1)[0,0].item(), .5, places=7)
        self.assertAlmostEqual(out['scores'][0,0].item(), -.5*math.log(2*math.pi*.25))

    def test_woodbury_dense_masks_and_gradients(self):
        torch.manual_seed(7)
        z = torch.randn(3,4,dtype=torch.double,requires_grad=True)
        means = torch.randn(3,5,4,dtype=torch.double,requires_grad=True)
        diag = (torch.rand(3,4,dtype=torch.double)+.1).requires_grad_()
        factor = torch.randn(3,4,2,dtype=torch.double,requires_grad=True)
        mask = torch.tensor([[True]*4,[False,True,False,True],[False]*4])
        prior = torch.randn(5,dtype=torch.double,requires_grad=True)
        low = lowrank_gaussian_scores(z,means,diag,factor,mask,prior)
        dense = gaussian_scores(z,means,torch.diag_embed(diag)+factor@factor.transpose(-1,-2),mask,prior)
        for key in ('scores','mahalanobis','logdet'):
            torch.testing.assert_close(low[key],dense[key])
        torch.testing.assert_close(low['scores'][2],prior)
        low['scores'].sum().backward()
        for tensor in (z,means,diag,factor,prior):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_class_specific_covariance_and_masked_nan(self):
        z = torch.tensor([[1.,float('nan')]],dtype=torch.double)
        means = torch.zeros(1,2,2,dtype=torch.double)
        cov = torch.tensor([[[[1.,float('nan')],[float('nan'),float('nan')]],[[4.,0.],[0.,1.]]]],dtype=torch.double)
        out = gaussian_scores(z,means,cov,torch.tensor([[True,False]]))
        torch.testing.assert_close(out['mahalanobis'],torch.tensor([[1.,.25]],dtype=torch.double))
        empty = lowrank_gaussian_scores(z,means,torch.full((1,2),float('nan'),dtype=torch.double),torch.full((1,2,1),float('nan'),dtype=torch.double),torch.zeros(1,2,dtype=torch.bool))
        self.assertTrue((empty['scores']==0).all())

    def test_dense_reference_distribution(self):
        z = torch.tensor([[1.,2.]])
        means = torch.tensor([[0.,0.],[2.,3.]])
        cov = torch.tensor([[[2.,.4],[.4,1.]]])
        scores = gaussian_scores(z,means,cov,torch.ones_like(z,dtype=torch.bool))['scores'][0]
        reference = torch.distributions.MultivariateNormal(means,covariance_matrix=cov[0]).log_prob(z)
        torch.testing.assert_close(scores,reference)


if __name__ == '__main__':
    unittest.main()
