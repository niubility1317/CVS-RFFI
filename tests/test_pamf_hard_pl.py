import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_ssl_research.hard_pl_objective import masked_pair,mechanism_loss,HardPLObjective


class HardPLMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026102314)

    def test_rejected_rows_have_no_CE_or_RDrop_gradient(self):
        a=torch.randn(6,6,requires_grad=True);b=torch.randn(6,6,requires_grad=True)
        y=torch.arange(6);accepted=torch.tensor([True,False,True,False,False,True])
        loss,ce,kl=masked_pair(a,b,y,accepted,1.)
        loss.backward()
        self.assertTrue(torch.equal(a.grad[~accepted],torch.zeros_like(a.grad[~accepted])))
        self.assertTrue(torch.equal(b.grad[~accepted],torch.zeros_like(b.grad[~accepted])))
        expected=.5*(F.cross_entropy(a,y,reduction='none',label_smoothing=.01)+F.cross_entropy(b,y,reduction='none',label_smoothing=.01))
        self.assertTrue(torch.allclose(ce,(expected*accepted).mean()))
        self.assertGreater(float(kl.detach()),0.)

    def test_As_uses_two_CE_and_zero_independent_KL(self):
        a=torch.randn(8,6,requires_grad=True);b=torch.randn(8,6,requires_grad=True)
        loss,ce,kl=masked_pair(a,b,torch.arange(8)%6,torch.ones(8,dtype=torch.bool),0.)
        self.assertTrue(torch.equal(loss,ce));self.assertGreater(float(kl.detach()),0.)

    def test_no_accepted_rows_are_finite_differentiable_zero(self):
        a=torch.randn(4,6,requires_grad=True);b=torch.randn(4,6,requires_grad=True)
        loss,_,_=masked_pair(a,b,torch.arange(4),torch.zeros(4,dtype=torch.bool),1.)
        loss.backward();self.assertEqual(float(loss.detach()),0.)
        self.assertEqual(float(a.grad.norm()+b.grad.norm()),0.)

    def test_empty_partition_is_differentiable_zero(self):
        a=torch.empty(0,6,requires_grad=True);b=torch.empty(0,6,requires_grad=True)
        loss,_,_=masked_pair(a,b,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.bool),0.)
        loss.backward();self.assertEqual(float(loss.detach()),0.)

    def test_CE_dose_matches_one_masked_CE_after_normalization(self):
        # Even when accepted records occur only in one half, equal logits in
        # all six passes must give exactly one masked CE dose, not 1.25.
        for n in (12,124,256):
            a=torch.randn(n,6,requires_grad=True);y=torch.arange(n)%6
            physical=torch.arange(n)%2==0;accepted=physical.clone()
            value,detail=mechanism_loss(a,a,a,a,a,a,y,accepted,physical,a.sum()*0.)
            expected=(F.cross_entropy(a,y,reduction='none',label_smoothing=.01)*accepted).mean()
            self.assertTrue(torch.allclose(value,expected,atol=1e-6))
            self.assertEqual(float(detail['ordinary_KL']),0.)
            self.assertEqual(float(detail['raw_KL']),0.)
            self.assertEqual(float(detail['As_KL_weight']),0.)

    def test_mismatched_gate_or_partition_is_rejected(self):
        a=torch.randn(6,6);y=torch.arange(6);gate=torch.ones(6,dtype=torch.bool)
        with self.assertRaises(ValueError):masked_pair(a,a,y,gate.float(),1.)
        with self.assertRaises(ValueError):mechanism_loss(a,a,a,a,a,a,y,gate,gate,a.sum()*0.)


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(4,6);self.student_calls=0
        with torch.no_grad():self.head.weight.zero_();self.head.bias.zero_();self.head.bias[0]=7.
        self._cd_U_seen=self._cd_U_accepted=0
    def forward(self,x):return self.head(x.flatten(1))
    def forward_identity_only(self,x,domain_labels=None):
        self.student_calls+=1;return dict(tx_logits=self(x))


class HardPLControlTests(unittest.TestCase):
    def test_control_is_exact_twice_plain_PL_and_restores_training_mode(self):
        torch.manual_seed(7);model=MiniModel().train();x=torch.randn(8,2,2);domain=torch.zeros(8,dtype=torch.long)
        model._cd_U_loader=[(x,torch.full((8,),-1),domain,dict(rx_i=torch.ones(8,dtype=torch.long)))]
        model._cd_U_iter=iter(model._cd_U_loader)
        objective=HardPLObjective(dict(hard_pl=dict(method='HPL2',threshold=.95,weight=2.)))
        with torch.no_grad():confidence,y=model.eval()(x).softmax(1).max(1)
        model.train();expected=2*(F.cross_entropy(model(x),y,reduction='none')*(confidence>=.95)).mean()
        actual=objective.loss(model,'cpu')
        self.assertTrue(torch.equal(actual,expected));self.assertTrue(model.training)
        self.assertEqual(model.student_calls,1);self.assertEqual(objective.seen,8)
        self.assertEqual(model._cd_U_seen,8);self.assertEqual(objective.ema_updates,0)
        self.assertGreater(float(actual.detach()),0.);self.assertEqual(objective.accepted,8)

    def test_visible_U_truth_is_rejected_before_any_model_forward(self):
        model=MiniModel().train();model._cd_U_loader=[(torch.randn(4,2,2),torch.arange(4),torch.zeros(4),{})]
        model._cd_U_iter=iter(model._cd_U_loader)
        objective=HardPLObjective(dict(hard_pl=dict(method='HPL2',threshold=.95,weight=2.)))
        with self.assertRaisesRegex(ValueError,'truth'):objective.loss(model,'cpu')
        self.assertEqual(model.student_calls,0)

    def test_eval_failure_restores_original_mode(self):
        model=MiniModel().train();model._cd_U_loader=[(torch.randn(4,2,2),torch.full((4,),-1),torch.zeros(4),{})]
        model._cd_U_iter=iter(model._cd_U_loader)
        model.forward=lambda x:(_ for _ in ()).throw(RuntimeError('probe'))
        objective=HardPLObjective(dict(hard_pl=dict(method='HPL2',threshold=.95,weight=2.)))
        with self.assertRaisesRegex(RuntimeError,'probe'):objective.loss(model,'cpu')
        self.assertTrue(model.training)

    def test_six_U_passes_gate_contrast_and_remove_feature_hooks(self,no_uas=False):
        from experiments.cric_dura_comparison.f_model import contrast
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Module()
                self.id_backbone.encoder.core=nn.Module();core=nn.Module();self.id_backbone.encoder.core.id_backbone=core
                core.time_down=nn.Conv1d(2,4,1);core.t2=nn.Module();core.t2.pool=nn.Conv1d(4,4,1);core.cls_head=nn.Linear(4,6)
                self._feature_l_steps=1;self._cd_U_seen=self._cd_U_accepted=self.student_calls=0;self.inputs=[]
                self._cd_practical=lambda x,meta,epoch,step:x*1.5
            def forward(self,x):
                if not self.training:
                    logits=torch.zeros(len(x),6);logits[torch.arange(6),torch.arange(6)]=7.;return logits
                core=self.id_backbone.encoder.core.id_backbone
                h=core.t2.pool(core.time_down(x));return core.cls_head(F.dropout(h.mean(-1),p=.1,training=True))
            def forward_identity_only(self,x,domain_labels=None):
                self.student_calls+=1;self.inputs.append(x.detach().clone());return dict(tx_logits=self(x))
        torch.manual_seed(17);model=Model().train();x=torch.randn(12,2,8)
        if no_uas:
            model._cd_practical=lambda *args:(_ for _ in ()).throw(AssertionError('U channel must not execute'))
        model._cd_U_loader=[(x,torch.full((12,),-1),torch.zeros(12,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*6)))]
        model._cd_U_iter=iter(model._cd_U_loader)
        objective=HardPLObjective(dict(hard_pl=dict(method='HPLT1U',threshold=.95,weight=2.),
            step_offset=6660,source_rxs=[1,3],hard_pl_smoke=True,hard_pl_no_uas=no_uas))
        seen=[]
        def capture(maps,y,rx):
            seen.append((len(y),{key:len(h) for key,h in maps.items()}));return contrast(maps,y,rx)
        with patch('experiments.pamf_ssl_research.hard_pl_objective.contrast',side_effect=capture):
            loss=objective.loss(model,'cpu');loss.backward()
        self.assertEqual(model.student_calls,6);self.assertEqual(objective.accepted,6)
        self.assertEqual(objective.student_views,72);self.assertEqual(objective.physical_seen,0 if no_uas else 6)
        self.assertEqual(objective.physical_student_views,0 if no_uas else 12);self.assertEqual(len(seen),2)
        self.assertEqual(objective.raw_As_seen,6 if no_uas else 0)
        self.assertTrue(all(n==6 and set(v.values())=={6} for n,v in seen))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        self.assertFalse(objective.capture_active);self.assertTrue(torch.isfinite(loss))
        if no_uas:
            mask=(torch.arange(12)+6661).remainder(2).bool()
            self.assertTrue(torch.equal(model.inputs[2][mask],x[mask]))
            self.assertTrue(torch.equal(model.inputs[2][~mask],model.inputs[0][~mask]))
            self.assertTrue(objective.smoke_U_As_input_raw)

    def test_no_U_channel_keeps_six_passes_and_exact_raw_As_input(self):
        self.test_six_U_passes_gate_contrast_and_remove_feature_hooks(no_uas=True)


if __name__=='__main__':unittest.main()
