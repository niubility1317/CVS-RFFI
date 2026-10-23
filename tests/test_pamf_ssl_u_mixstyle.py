from copy import deepcopy
import unittest
import torch

from experiments.pamf_ssl_research.u_mixstyle import configuration, validate
from experiments.pamf_ssl_research.u_mixstyle_objective import protected_partners, mix_moments, UnlabeledMixStyleObjective
from experiments.pamf_ssl_research.plan import configuration as scaffold, SEEDS
from tests.test_pamf_ssl_research import template
from tests.test_pamf_ssl_paired_u import Student


class StyleStudent(Student):
    def __init__(self):
        super().__init__()
        shallow=torch.nn.Conv1d(2,2,1)
        with torch.no_grad():
            shallow.weight.copy_(torch.eye(2).view(2,2,1));shallow.bias.zero_()
        self.levels=torch.nn.ModuleList([shallow,torch.nn.Identity()])
    def get_submodule(self,name):
        return self.levels[0 if name.endswith('time_down') else 1]
    def forward_identity_only(self,x,domain_labels):
        for layer in self.levels:x=layer(x)
        return super().forward_identity_only(x,domain_labels)


def setup():
    parent=scaffold(template(),'RXPSM',SEEDS[0],'/p','old')
    c=configuration(parent,'/p','new');c['u_mixstyle_smoke']=True
    student=StyleStudent();x=torch.randn(8,2,256)
    meta=dict(rx_i=torch.tensor([1,3,4,6,8,1,3,4]),physical_id=torch.arange(8)*17+5)
    domain=torch.arange(8);student._cd_U_loader=[(x,torch.full((8,),-1),domain,meta)]
    student._cd_U_iter=iter(student._cd_U_loader)
    from experiments.pamf_ssl_research.objective import Objective
    inherited=Objective(student,parent).state_dict()
    inherited.update(ema_updates=6660,seen=963900,physical_seen=481950,calls=3774)
    obj=UnlabeledMixStyleObjective(student,c);obj.load_state_dict(inherited)
    obj.current_epoch,obj.current_step=31,6661;obj.labeled=(x,torch.zeros(8,dtype=torch.long),meta['rx_i'])
    return student,obj,c,x,meta,domain


class UnlabeledMixStyleTest(unittest.TestCase):
    def test_partner_is_same_prediction_other_source_rx_and_both_confident(self):
        p=torch.tensor([[.9,.1],[.8,.2],[.1,.9],[.2,.8],[.85,.15],[.95,.05]],requires_grad=True)
        w=torch.tensor([1.,.9,1.,.8,.2,1.],requires_grad=True)
        rx=torch.tensor([1,3,1,3,4,1])
        for _ in range(10):
            partner,valid=protected_partners(p,w,rx,[1,3,4,6,8],.5)
            self.assertEqual(valid.tolist(),[True,True,True,True,False,True])
            self.assertTrue(bool((p.argmax(1)[valid]==p.argmax(1)[partner[valid]]).all()))
            self.assertTrue(bool((rx[valid]!=rx[partner[valid]]).all()))
            self.assertTrue(bool((w[partner[valid]]>=.5).all()))
            self.assertEqual(partner[4].item(),4)
            self.assertFalse(partner.requires_grad)
            self.assertFalse(valid.requires_grad)

    def test_mixed_moments_are_literal_convex_combinations_and_content_gradient_only(self):
        h=torch.tensor([[[0.,2.,4.,6.]],[[10.,14.,18.,22.]]],requires_grad=True)
        partner=torch.tensor([1,0]);valid=torch.tensor([True,False]);lam=torch.tensor([.25,.5]).view(2,1,1)
        out=mix_moments(h,partner,valid,lam,1e-6)
        mu=h.detach().mean(-1,keepdim=True)
        sigma=(h.detach().var(-1,unbiased=False,keepdim=True)+1e-6).sqrt()
        expected=(h-mu)/sigma*(lam*sigma+(1-lam)*sigma[partner])+lam*mu+(1-lam)*mu[partner]
        torch.testing.assert_close(out[0],expected[0])
        torch.testing.assert_close(out[1],h[1])
        out[0].sum().backward()
        self.assertGreater(float(h.grad[0].norm()),0.)
        self.assertEqual(float(h.grad[1].norm()),0.)

    def test_constant_features_and_no_valid_partner_stay_finite_and_unchanged(self):
        h=torch.ones(4,3,8,requires_grad=True)
        p=torch.ones(4,1);rx=torch.ones(4,dtype=torch.long)
        partner,valid=protected_partners(p,torch.ones(4),rx,[1,3,4,6,8],.5)
        self.assertFalse(bool(valid.any()))
        out=mix_moments(h,partner,valid,torch.zeros(4,1,1),1e-6)
        torch.testing.assert_close(out,h)
        out.sum().backward()
        self.assertTrue(bool(torch.isfinite(h.grad).all()))

    def test_non_source_rx_and_different_recipe_or_target_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError,'source RX'):
            protected_partners(torch.ones(2,1),torch.ones(2),torch.tensor([1,9]),[1,3,4,6,8],.5)
        parent=scaffold(template(),'RXPSM',SEEDS[0],'/p','old')
        c=configuration(parent,'/p','new')
        self.assertEqual(c['u_mixstyle']['alpha'],.1)
        self.assertEqual(c['u_mixstyle']['probability_per_level'],.5)
        self.assertNotIn('u_cric',c)
        self.assertNotIn('paired_u',c)
        bad=deepcopy(c);bad['u_mixstyle']['alpha']=.2
        with self.assertRaisesRegex(ValueError,'U-MixStyle recipe'):validate(bad)
        bad=deepcopy(c);bad['truth']='/target/truth'
        with self.assertRaisesRegex(ValueError,'source handles'):validate(bad)

    def test_only_current_u_student_hooks_change_features_and_no_extra_parameters(self):
        student,obj,c,x,meta,domain=setup();before=set(student.state_dict())
        loss=obj.loss(student,'cpu');loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(obj.u_mixstyle_calls,1)
        self.assertEqual(obj.u_mixstyle_seen,8)
        self.assertEqual(obj.u_mixstyle_eligible,8)
        self.assertEqual(obj.u_mixstyle_applied,{'time_down':8,'t2.pool':8})
        self.assertTrue(all(h.grad is not None and bool(torch.isfinite(h.grad).all()) for h in obj._smoke_maps.values()))
        self.assertTrue(all(p.grad is None for p in obj.teacher.parameters()))
        self.assertFalse(obj._style_context_active)
        self.assertTrue(all(not layer._forward_hooks for layer in student.levels))
        self.assertEqual(set(student.state_dict()),before)
        student.forward_identity_only(x,domain)
        torch.testing.assert_close(student.last_inputs,x)
        obj.after_update(student);saved=obj.state_dict()
        restored=UnlabeledMixStyleObjective(student,c);restored.load_state_dict(saved)
        self.assertEqual(restored.ema_updates,6661)
        self.assertEqual(restored.seen,963908)
        self.assertEqual(restored.physical_seen,481954)
        self.assertEqual(restored.u_mixstyle_applied['t2.pool'],8)

    def test_hidden_u_truth_is_rejected_before_student_transform(self):
        student,obj,c,x,meta,domain=setup()
        student._cd_U_iter=iter([(x,torch.zeros(8,dtype=torch.long),domain,meta)])
        with self.assertRaisesRegex(ValueError,'U TX truth exposed'):obj.loss(student,'cpu')
        self.assertEqual(obj.u_mixstyle_calls,0)
        self.assertTrue(all(not layer._forward_hooks for layer in student.levels))

    def test_forward_failure_removes_hooks_without_counting_a_completed_call(self):
        student,obj,c,x,meta,domain=setup()
        def failed(*args,**kwargs):raise RuntimeError('intentional student failure')
        student.forward_identity_only=failed
        with self.assertRaisesRegex(RuntimeError,'intentional student failure'):
            obj.unlabeled_student_forward(student,x,domain,torch.ones(8,6)/6,torch.ones(8),meta['rx_i'])
        self.assertEqual(obj.u_mixstyle_calls,0)
        self.assertFalse(obj._style_context_active)
        self.assertTrue(all(not layer._forward_hooks for layer in student.levels))


if __name__=='__main__':unittest.main()
