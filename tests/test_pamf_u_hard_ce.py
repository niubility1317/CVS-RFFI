import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.u_hard_ce_objective import hard_CE_delta,METHODS,recipe
from experiments.pamf_channel_pl.receiver_balance_objective import semantic_reallocation
from tests import test_pamf_receiver_balance_source as fixture


class HardUCETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def inputs(self):
        torch.manual_seed(711);return [torch.randn(8,6,requires_grad=True) for _ in range(6)],torch.arange(8)%6,torch.tensor([1,1,0,1,1,1,0,1],dtype=torch.bool),torch.arange(8)%2==0,torch.ones(8)

    def test_exact_linear_identity_matches_six_hard_CE_losses_and_gradients(self):
        z,y,a,h,w=self.inputs();w[a]=torch.tensor([.5,1.5,1.,1.,.5,1.5])
        rb,before,after=semantic_reallocation(z,y,a,h,w);delta=hard_CE_delta(z,y,a,w)
        coefficients=(.25,.25,.25,.25,.125,.125)
        expected=sum(c*((F.cross_entropy(v,y,reduction='none')-F.cross_entropy(v,y,reduction='none',label_smoothing=.01))*a*w).mean()/1.25 for v,c in zip(z,coefficients))
        torch.testing.assert_close(delta,expected,atol=2e-7,rtol=1e-4)
        ga=torch.autograd.grad(delta,z,retain_graph=True);gb=torch.autograd.grad(expected,z,retain_graph=True)
        for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=1e-7,rtol=1e-4)
        def kl(x,b):
            p,q=x.log_softmax(1),b.log_softmax(1);return .5*((p.exp()-q.exp())*(p-q)).sum(1)
        ce=[F.cross_entropy(v,y,reduction='none') for v in z]
        packet=sum(c*v for c,v in zip(coefficients,ce))+.5*kl(z[0],z[1])+.5*(~h)*kl(z[2],z[3])+.25*kl(z[4],z[5])
        full=(packet*a*w).mean()/1.25;actual=after+delta;torch.testing.assert_close(actual,full)
        ga=torch.autograd.grad(actual,z,retain_graph=True);gb=torch.autograd.grad(full,z)
        for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=1e-7,rtol=1e-4)

    def test_rejected_packets_receive_zero_target_change(self):
        z,y,a,h,w=self.inputs();delta=hard_CE_delta(z,y,a,w);grad=torch.autograd.grad(delta,z)
        for g in grad:self.assertTrue(torch.equal(g[~a],torch.zeros_like(g[~a])))
        a[:]=False;self.assertEqual(float(hard_CE_delta(z,y,a,w).detach()),0)

    def test_high_confidence_hard_CE_restores_upward_pseudo_margin(self):
        y=torch.tensor([0,0]);a=torch.tensor([True,True]);w=torch.ones(2)
        z=[torch.tensor([[12.,0,0,0,0,0],[12.,0,0,0,0,0]],requires_grad=True) for _ in range(6)]
        h=torch.tensor([True,False]);_,_,smooth=semantic_reallocation(z,y,a,h,w);hard=smooth+hard_CE_delta(z,y,a,w)
        gs=torch.autograd.grad(smooth,z,retain_graph=True);gh=torch.autograd.grad(hard,z)
        self.assertTrue(all(float(v[0,0])>0 for v in gs));self.assertTrue(all(float(v[0,0])<0 for v in gh))

    def test_forbidden_truth_like_labels_trainable_weights_or_nonfinite_rejected(self):
        z,y,a,h,w=self.inputs()
        with self.assertRaises(ValueError):hard_CE_delta(z,y,a,w.requires_grad_())
        w=w.detach();y[0]=-1
        with self.assertRaises(ValueError):hard_CE_delta(z,y,a,w)
        y[0]=0;z[0]=z[0]*torch.nan
        with self.assertRaises(ValueError):hard_CE_delta(z,y,a,w)

    def test_immutable_source_config_keeps_L_raw_gate_budget_parent_and_head(self):
        from experiments.pamf_channel_pl import u_hard_ce_source as s
        previous=fixture.ReceiverBalanceSourceTests();previous.setup_configs();ctx=fixture.context()
        with patch.object(s,'read',return_value=previous.previous),patch.object(s.reference,'validate_config',return_value=previous.previous):
            method=next(iter(METHODS));c=s.template(ctx,'commit',method);s.validate_config(c)
            self.assertEqual(c['U_semantic_target'],recipe(method));self.assertEqual(c['plain_PL_threshold'],.95)
            for key in ('shared_SIG1_E20','clean_H','hard_pl_joint_contrast','hard_pl_gradient_guard','sig_time_views','default_head_joint'):
                self.assertEqual(c[key],previous.previous[key])
            for change in (dict(G={}),dict(target_inputs={}),dict(epochs=40),dict(sig_time_views=3),dict(plain_PL_threshold=.8)):
                with self.assertRaises(ValueError):s.validate_config(dict(c,**change))


if __name__=='__main__':unittest.main()
