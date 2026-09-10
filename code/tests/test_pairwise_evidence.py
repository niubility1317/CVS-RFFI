import unittest
import torch
from cvsrffi.pairwise_evidence import shared_pair_distance, schur_incremental_information, effective_fisher_information, SharedPairResidual, project_pairwise


class PairTests(unittest.TestCase):
    def test_correlation_and_schur(self):
        cov = torch.tensor([[1.,.9],[.9,1.]],dtype=torch.double)
        delta = torch.ones(2,dtype=torch.double)
        means = torch.stack((torch.zeros_like(delta),delta))[None]
        distance = shared_pair_distance(means,cov[None],torch.ones(1,2,dtype=torch.bool))
        out = schur_incremental_information(delta,cov,[0],[1])
        self.assertAlmostEqual(distance[0,0,1].item(),2/1.9)
        self.assertAlmostEqual(out['incremental'].item(),2/1.9-1)
        torch.testing.assert_close(out['total'],distance[0,0,1])
        with self.assertRaises(ValueError):
            shared_pair_distance(means,cov[None,None].expand(1,2,2,2),torch.ones(1,2,dtype=torch.bool))

    def test_fisher_identifiable_rank(self):
        identity = torch.diag(torch.tensor([2.,1.],dtype=torch.double))
        nuisance = torch.diag(torch.tensor([1.,0.],dtype=torch.double))
        cross = torch.tensor([[1.,0.],[0.,0.]],dtype=torch.double)
        out = effective_fisher_information(identity,cross,nuisance)
        torch.testing.assert_close(out['information'],torch.eye(2,dtype=torch.double))
        self.assertEqual(out['nuisance_rank'].item(),1)
        self.assertEqual(out['identifiable_rank'].item(),2)
        empty = effective_fisher_information(identity, cross[:, :0], nuisance[:0, :0])
        torch.testing.assert_close(empty['information'], identity)
        self.assertEqual(empty['nuisance_rank'].item(), 0)
        with self.assertRaises(ValueError):
            effective_fisher_information(identity,torch.ones_like(cross),nuisance)

    def test_residual_permutation_bound_and_baseline(self):
        torch.manual_seed(3)
        net = SharedPairResidual(3,2,bound=.4).double()
        z = torch.randn(2,3,dtype=torch.double)
        means = torch.randn(2,4,3,dtype=torch.double)
        state = torch.randn(2,2,dtype=torch.double)
        q = torch.rand(2,3,dtype=torch.double)
        cov = torch.ones_like(means)
        self.assertTrue((net(z,means,state,q,cov)==0).all())
        torch.nn.init.normal_(net.network[-1].weight)
        r = net(z,means,state,q,cov)
        torch.testing.assert_close(r,-r.transpose(1,2))
        self.assertLessEqual(r.abs().max().item(),.4)
        p = torch.tensor([2,0,3,1])
        torch.testing.assert_close(net(z,means[:,p],state,q,cov[:,p]),r[:,p][:,:,p])
        r.square().sum().backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in net.parameters()))

    def test_projection_normal_equations_and_cycle(self):
        base = torch.tensor([[1.,-2.,3.]],dtype=torch.double,requires_grad=True)
        d = base[:,:,None]-base[:,None,:]
        w = torch.ones_like(d)
        torch.testing.assert_close(project_pairwise(base,d,w),base)
        cyclic = torch.tensor([[[0.,1.,-1.],[-1.,0.,1.],[1.,-1.,0.]]],dtype=torch.double,requires_grad=True)
        out = project_pairwise(base,cyclic,w,anchor=.7)
        residual = .7*(out-base)+(w*(out[:,:,None]-out[:,None,:]-cyclic)).sum(-1)
        torch.testing.assert_close(residual,torch.zeros_like(residual),atol=1e-12,rtol=0)
        p = torch.tensor([2,0,1])
        torch.testing.assert_close(project_pairwise(base[:,p],cyclic[:,p][:,:,p],w,anchor=.7),out[:,p])
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(base.grad).all() and torch.isfinite(cyclic.grad).all())


if __name__ == '__main__':
    unittest.main()
