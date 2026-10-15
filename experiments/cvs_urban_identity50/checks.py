import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
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

    def test_new_native_hook_label_phase_has_teacher_and_no_U_metadata(self):
        from . import design as d,model as model_api
        from .observer import Observer
        for row in d.rows():
            c=d.config(row);m=model_api.build(c,'cpu').train()
            teacher=copy.deepcopy(m).eval().requires_grad_(False)
            x=torch.randn(4,2,256);y=f.frozen_teacher(teacher,x)['tx_logits'].argmax(1)
            meta=dict(rx_i=torch.tensor([1,3,4,6]),day_i=torch.ones(4,dtype=torch.long),
                eq_i=torch.ones(4,dtype=torch.long),base_index=torch.arange(4))
            clean=m(x,y_tx=y,grl_lambda=1.,return_aux=True,domain_labels=torch.zeros(4,dtype=torch.long))
            ce=F.cross_entropy(clean['tx_logits'],y,label_smoothing=.01)
            with tempfile.TemporaryDirectory() as td:
                obs=Observer(c);obs.initialize(m,c,Path(td))
                kw=dict(model=m,args=d.make_args(c,'cpu'),epoch=1,batch_idx=1,clean_iq=x,labels=y,
                    clean_meta=meta,clean_output=clean,clean_ce=ce,unlabeled_iq=None,unlabeled_meta=None,
                    teacher_prob=None,weak_mask=None,teacher_model=teacher)
                adj,metrics=obs.observe(**kw);loss=ce+adj;loss.backward()
                self.assertTrue(torch.isfinite(loss));self.assertEqual(obs.counts['satellite_L_packets'],4)
                self.assertEqual(obs.counts['teacher_L_clean_packets'],4)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                self.assertGreater(float(f.boundary_parameters(m)['default_head'].grad.norm()),0.)
                self.assertTrue(all(v==0 for v in obs.epoch_records[0]['auxiliary_boundary_norms'].values()))
                obs.trace.close()

    def test_frozen_scratch_budget_and_matching_physics(self):
        from . import design as d
        from .channel import PairedChannel
        from experiments.cvs_urban_sigtime50.channel import PairedChannel as OldChannel
        old=base.config(next(r for r in base.rows() if r['arm']=='C3'))
        x=torch.randn(3,2,256);meta=dict(rx_i=torch.tensor([1,3,8]),day_i=torch.ones(3,dtype=torch.long),
            eq_i=torch.ones(3,dtype=torch.long),base_index=torch.arange(3))
        expected,em=OldChannel(old).apply(x,meta,4,7,eta=.05)
        for row in d.rows():
            c=d.config(row);a=d.make_args(c,'cpu')
            self.assertEqual((a.epochs,a.label_epochs,a.pseudo_epochs),(50,33,17))
            self.assertTrue(a.from_scratch);self.assertFalse(a.baseline_ckpt or a.teacher_ckpt)
            self.assertFalse(c['margin'] or c['urban_ssl']);self.assertEqual(len(c['frozen_eta']),50)
            actual,am=PairedChannel(c).apply(x,meta,4,7,eta=c['frozen_eta'][3])
            self.assertTrue(torch.equal(expected,actual));self.assertEqual(em,am)
            invalid=copy.deepcopy(c);invalid['auxiliary']['coefficient']*=2
            with self.assertRaises(ValueError):d.validate(invalid)

    def test_incomplete_predictions_never_open_truth(self):
        from . import design as d,evaluate
        row=d.rows()[0]['row_id']
        with tempfile.TemporaryDirectory() as td,patch.object(d,'BASE',Path(td)),patch.object(d,'TRUTH',Path(td)/'forbidden_truth'):
            with self.assertRaises(FileNotFoundError):evaluate.score(row)
            self.assertFalse((Path(td)/row/'scoring').exists())

    def test_registered_matrix_and_publisher_call_actual_new_module(self):
        from . import design as d,publish
        spec=d.read(d.ROOT/'experiments/cvs_urban_identity50/registration.json')
        self.assertEqual([r['config'] for r in spec['rows']],[d.config(r) for r in d.rows()])
        self.assertIn('experiments.cvs_urban_identity50.dispatch',publish.REMOTE)
        self.assertIn('experiments/cvs_urban_identity50/registration.json',publish.REMOTE)
        self.assertNotIn('experiments.cvs_urban_sigtime50.dispatch',publish.REMOTE)


if __name__=='__main__':unittest.main()
