import unittest
from copy import deepcopy
from unittest.mock import patch
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.h_quality_objective import quality,physical_H_delta,METHODS,recipe
from experiments.pamf_channel_pl.receiver_balance_objective import semantic_reallocation
from tests import test_pamf_receiver_balance_source as fixture
old_context=fixture.context;old_runtime=fixture.runtime


class HQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_weights_are_detached_and_modes_have_exact_pressure(self):
        z=torch.tensor([[8.,0,0,0,0,0],[0,8.,0,0,0,0],[0.,0,0,0,0,0]],requires_grad=True);y=torch.zeros(3,dtype=torch.long)
        torch.testing.assert_close(quality(z,y,'fixed_half'),torch.full((3,),.5))
        torch.testing.assert_close(quality(z,y,'gate'),torch.tensor([1.,0.,0.]))
        w=quality(z,y,'soft');self.assertFalse(w.requires_grad);self.assertEqual(float(w[0]),1);self.assertAlmostEqual(float(w[1]),.05)
        self.assertAlmostEqual(float(w[2]),1/3,places=6)

    def inputs(self):
        torch.manual_seed(17);return [torch.randn(8,6,requires_grad=True) for _ in range(6)],torch.arange(8)%6,torch.tensor([1,1,0,1,1,1,0,1],dtype=torch.bool),torch.arange(8)%2==0,torch.ones(8)

    def test_delta_only_changes_accepted_physical_H_CE_not_raw_ordinary(self):
        z,y,a,h,w=self.inputs();delta,details=physical_H_delta(z,y,a,h,w,'soft')
        gradients=torch.autograd.grad(delta,z,allow_unused=True)
        for i in (0,1,4,5):self.assertTrue(gradients[i] is None or torch.equal(gradients[i],torch.zeros_like(z[i])))
        for i in (2,3):self.assertTrue(torch.equal(gradients[i][~(a&h)],torch.zeros_like(gradients[i][~(a&h)])))
        self.assertEqual(sum(d['eligible'] for d in details),2*int((a&h).sum()))

    def test_exact_weighted_scalar_and_detached_derivative_all_modes(self):
        for mode in METHODS.values():
            z,y,a,h,w=self.inputs();w[a]=torch.tensor([.5,1.5,1.,1.,.5,1.5]);rb,before,after=semantic_reallocation(z,y,a,h,w)
            delta,_=physical_H_delta(z,y,a,h,w,mode);actual=after+delta
            ce=[F.cross_entropy(v,y,reduction='none',label_smoothing=.01) for v in z]
            mixed=[]
            for i in (2,3):mixed.append(ce[i]*torch.where(h,quality(z[i],y,mode),torch.ones_like(w)))
            def kl(x,b):
                p,q=x.log_softmax(1),b.log_softmax(1);return .5*((p.exp()-q.exp())*(p-q)).sum(1)
            packet=.25*(ce[0]+ce[1])+.5*kl(z[0],z[1])+.25*(mixed[0]+mixed[1])+.5*(~h)*kl(z[2],z[3])+.125*(ce[4]+ce[5])+.25*kl(z[4],z[5])
            expected=(packet*a*w).mean()/1.25
            torch.testing.assert_close(actual,expected)
            ga=torch.autograd.grad(actual,z,retain_graph=True);gb=torch.autograd.grad(expected,z)
            for x,b in zip(ga,gb):torch.testing.assert_close(x,b,atol=1e-7,rtol=1e-5)
            self.assertGreaterEqual(float(actual.detach()),0)

    def test_no_accepted_H_means_zero_delta(self):
        z,y,a,h,w=self.inputs();a[:]=False
        delta,details=physical_H_delta(z,y,a,h,w,'gate');self.assertEqual(float(delta.detach()),0)
        self.assertTrue(all(d['eligible']==d['nonzero']==0 for d in details))

    def test_invalid_or_trainable_weights_rejected(self):
        z,y,a,h,w=self.inputs()
        with self.assertRaises(ValueError):physical_H_delta(z,y,a,h,w.requires_grad_(),'soft')
        with self.assertRaises(ValueError):quality(z[0]*torch.nan,y,'gate')
        with self.assertRaises(ValueError):quality(z[0],y,'foreign')

    def test_configs_preserve_raw_gate_full_L_own_E20_and_reject_foreign_inputs(self):
        from experiments.pamf_channel_pl import h_quality_source as s
        previous=fixture.ReceiverBalanceSourceTests();previous.setup_configs()
        ctx=old_context()
        with patch.object(s,'read',return_value=previous.previous),patch.object(s.reference,'validate_config',return_value=previous.previous):
            for method in METHODS:
                c=s.template(ctx,'commit',method);s.validate_config(c);self.assertEqual(c['H_quality'],recipe(method))
                self.assertEqual(c['plain_PL_threshold'],.95);self.assertEqual(c['epochs'],50);self.assertEqual(c['shared_SIG1_E20'],previous.previous['shared_SIG1_E20'])
                for change in (dict(target_inputs={}),dict(source_G={}),dict(plain_PL_threshold=.8),dict(sig_time_views=3),dict(epochs=40)):
                    with self.assertRaises(ValueError):s.validate_config(dict(c,**change))

    def test_context_and_runtime_reject_G_target_extra_gpu_or_root_changes(self):
        from experiments.pamf_channel_pl import h_quality_source as s
        c=old_context();s.validate_context(c)
        for key in ('G','truth','target_inputs'):
            with self.assertRaises(ValueError):s.validate_context(dict(c,**{key:'forbidden'}))
        rt=old_runtime();rt.update(run_id=s.RUN,launch_owner=s.OWNER,run_root='/project/runs/'+s.RUN,release_root='/project/releases/'+s.RELEASE,
            context='/project/releases/'+s.RELEASE+'/context.json',runtime_file='/project/releases/'+s.RELEASE+'/runtime.json')
        s.validate_runtime(rt)
        for change in (dict(allowed_gpus=[0]),dict(run_root='/foreign'),dict(G={})):
            with self.assertRaises(ValueError):s.validate_runtime(dict(rt,**change))


if __name__=='__main__':unittest.main()
