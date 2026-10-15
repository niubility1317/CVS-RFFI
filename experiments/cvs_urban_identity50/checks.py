import copy,unittest
import torch
from torch.nn import functional as F
from experiments.cvs_urban_sigtime50 import design as base
from experiments.cvs_urban_sigtime50.model import build
from . import features as f
from . import objective as o


class IdentityChecks(unittest.TestCase):
    def model(self):
        m=build(base.config(next(r for r in base.rows() if r['arm']=='C3')),'cpu')
        with torch.no_grad():m.id_backbone.encoder.response_gain.fill_(.07)
        return m

    def test_actual_joint_matches_in_train_and_eval_without_added_parameters(self):
        m=self.model();before=set(m.state_dict());x=torch.randn(4,2,256)
        for mode in (True,False):
            m.train(mode);out,z=f.satellite_forward(m,x)
            self.assertTrue(torch.equal(z,out['id_feat_joint']))
            self.assertEqual(set(m.state_dict()),before)
            self.assertFalse(any(module._forward_hooks for module in m.modules()))

    def test_each_auxiliary_has_zero_W_gain_gradient_and_real_branch_gradient(self):
        for kind in ('direction','output_kl'):
            m=self.model().train();teacher=copy.deepcopy(m).eval().requires_grad_(False)
            x=torch.randn(4,2,256);target=f.frozen_teacher(teacher,x);labels=target['tx_logits'].argmax(1)
            out,z=f.satellite_forward(m,.7*x+.2*torch.randn_like(x))
            weight=o.labeled_reliability(target['tx_logits'],labels)
            loss=o.direction(z,target['id_feat_joint'],weight) if kind=='direction' else o.output_kl(z,target['tx_logits'],weight,m.id_backbone.encoder.core.id_backbone.cls_head)
            groups=f.parameter_groups(m);boundary=f.boundary_parameters(m)
            params=[p for ps in groups.values() for p in ps]+list(boundary.values())
            gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True);offset=0
            for name,ps in groups.items():
                own=gs[offset:offset+len(ps)];offset+=len(ps)
                self.assertGreater(sum(float(g.square().sum()) for g in own if g is not None),0.,(kind,name))
            for name,g in zip(boundary,gs[offset:]):
                self.assertTrue(g is None or float(g.abs().max())<1e-9,(kind,name))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            ce=F.cross_entropy(out['tx_logits'],labels);ce.backward()
            self.assertGreater(float(boundary['default_head'].grad.norm()),0.)
            for name,p in boundary.items():self.assertGreater(float(p.grad.norm()),0.,name)

    def test_direction_uses_direction_not_original_norm(self):
        x=torch.randn(4,160,requires_grad=True);target=torch.randn(4,160);w=torch.ones(4)
        self.assertTrue(torch.allclose(o.direction(x,target,w),o.direction(x*5,target*3,w),atol=1e-6))
        zero=torch.zeros(4,160,requires_grad=True);loss=o.direction(zero,target,w);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(zero.grad).all())

    def test_wrong_clean_teacher_disables_only_auxiliary(self):
        logits=torch.randn(4,6);labels=(logits.argmax(1)+1)%6
        w=o.labeled_reliability(logits,labels);self.assertEqual(float(w.sum()),0.)
        student=torch.randn(4,160,requires_grad=True)
        self.assertEqual(float(o.direction(student,torch.randn(4,160),w)),0.)
        with self.assertRaises(ValueError):o.labeled_reliability(logits,torch.full((4,),-1))

    def test_teacher_target_and_weight_cannot_receive_gradients(self):
        x=torch.randn(2,160,requires_grad=True)
        with self.assertRaises(ValueError):o.direction(x,torch.randn(2,160,requires_grad=True),torch.ones(2))
        with self.assertRaises(ValueError):o.direction(x,torch.randn(2,160),torch.ones(2,requires_grad=True))

    def test_low_confidence_student_is_not_a_gate(self):
        x=torch.randn(2,160,requires_grad=True);target=torch.randn(2,160)
        loss=o.direction(x,target,torch.ones(2));loss.backward()
        self.assertGreater(float(x.grad.norm()),0.)

    def test_static_source_coefficient_is_not_online_gradient_equalization(self):
        from .calibrate import coefficient
        c=coefficient([.2,.3,.4]);self.assertAlmostEqual(c['coefficient'],1/3)
        self.assertAlmostEqual(c['weighted_median_gradient_ratio'],.1)
        self.assertTrue(coefficient([1e-5])['clipped'])
        with self.assertRaises(ValueError):coefficient([0.,0.])


if __name__=='__main__':unittest.main()
