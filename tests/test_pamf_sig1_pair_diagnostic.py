import unittest
import numpy as np
import torch
from experiments.pamf_channel_pl import sig1_pair_diagnostic as d


class PairDiagnosticTests(unittest.TestCase):
    def setUp(self):
        g=np.random.default_rng(124)
        self.z=[g.normal(size=(48,6))*3 for _ in range(3)]
        self.p=[d.probability(z) for z in self.z]
        y=g.integers(6,size=48);self.t=np.full((48,6),.01/6);self.t[np.arange(48),y]+=.99

    def test_peer_probability_true_CE_dot_identity(self):
        a,b=d.gradients('G_peer_probability',*self.p,self.z[1],self.z[2])
        actual=(a*(self.p[1]-self.t)).sum(1)+(b*(self.p[2]-self.t)).sum(1)
        np.testing.assert_allclose(actual,np.square(self.p[1]-self.p[2]).sum(1),atol=1e-14)

    def test_cosine_score_MSE_monotonicity(self):
        a,b=d.gradients('G_peer_cosine_scores',*self.p,self.z[1],self.z[2])
        dot=(a*(self.p[1]-self.t)).sum(1)+(b*(self.p[2]-self.t)).sum(1)
        self.assertGreaterEqual(dot.min(),-1e-14)
        za=torch.tensor(self.z[1],requires_grad=True);zb=torch.tensor(self.z[2],requires_grad=True)
        loss=.5*((za/30-zb/30)**2).mean(1).sum();ga,gb=torch.autograd.grad(loss,(za,zb))
        np.testing.assert_allclose(a,ga.tolist(),atol=1e-14);np.testing.assert_allclose(b,gb.tolist(),atol=1e-14)

    def test_raw_soft_can_oppose_shared_true_CE(self):
        p0=np.array([[.1,.9,0,0,0,0]]);pa=np.array([[.8,.2,0,0,0,0]])
        a,b=d.gradients('raw_soft',p0,pa,pa,np.zeros((1,6)),np.zeros((1,6)))
        target=np.array([[1.,0,0,0,0,0]])
        self.assertLess(float((a*(pa-target)).sum()+(b*(pa-target)).sum()),0)

    def test_unit_feature_leaf_Jacobian_matches_frozen_cosine_head(self):
        g=np.random.default_rng(136);h=g.normal(size=(9,160));h/=np.linalg.norm(h,axis=1,keepdims=True)
        w=g.normal(size=(6,160));delta=g.normal(size=(9,6))
        leaf=torch.tensor(h,requires_grad=True);weight=torch.tensor(w);before=weight.clone()
        z=30*torch.nn.functional.linear(torch.nn.functional.normalize(leaf,dim=1),torch.nn.functional.normalize(weight,dim=1))
        actual=torch.autograd.grad((z*torch.tensor(delta)).sum(),leaf)[0]
        np.testing.assert_allclose(d.feature_gradient(h,delta,w),actual.tolist(),rtol=1e-12,atol=1e-12)
        self.assertIsNone(weight.grad);self.assertTrue(torch.equal(weight,before))

    def test_invalid_shapes_and_nonfinite_rejected(self):
        for z in (np.ones((2,5)),np.ones(6),np.full((2,6),np.nan)):
            with self.assertRaises(ValueError):d.probability(z)
        with self.assertRaises(ValueError):d.gradients('unknown',*self.p,self.z[1],self.z[2])
        with self.assertRaises(ValueError):d.feature_gradient(np.ones((2,159)),np.ones((2,6)),np.ones((6,160)))

    def test_CPU_source_runtime_forbids_target_or_GPU(self):
        rt=dict(run_id=d.RUN,project='/p',python='python',run_root='/p/runs/'+d.RUN,release_root='/p/releases/'+d.RELEASE,
            context='/p/releases/'+d.RELEASE+'/context.json',runtime_file='/p/releases/'+d.RELEASE+'/runtime.json',code_commit='c',
            frozen_sources={},allowed_gpus=[],launch_owner='codex-pamf-sig1-rejected-pairs-20261026')
        d.validate_runtime(rt)
        for change in (dict(truth='x'),dict(allowed_gpus=[0]),dict(run_root='/other')):
            with self.assertRaises(ValueError):d.validate_runtime(dict(rt,**change))


if __name__=='__main__':unittest.main()
