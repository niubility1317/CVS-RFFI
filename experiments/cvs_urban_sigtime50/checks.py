import copy,unittest
import torch
from torch.nn import functional as F
from . import design as d
from .teacher import acceptance,weak_iq,calibrate
from .objective import soft_urban_ce


class FixedTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor(8.),requires_grad=False)
        self.register_buffer('training_calls',torch.tensor(0))
    def forward_identity_only(self,x):
        if self.training:self.training_calls.add_(1)
        z=torch.zeros(len(x),6);z[:,0]=self.scale
        return dict(tx_logits=z)
    def forward(self,x,**kwargs):return self.forward_identity_only(x)


class E50Checks(unittest.TestCase):
    def test_release_entrypoint_imports_and_targets_e50(self):
        from . import publish
        self.assertEqual(publish.configuration()['run'],d.RUN)
        self.assertIn("release/'experiments/cvs_urban_sigtime50/registration.json'",publish.REMOTE)
        self.assertIn("release/'experiments/cvs_urban_sigtime50'",publish.REMOTE)
        with self.assertRaises(ValueError):publish.configuration(True)

    def test_fixed_budget_recipe_and_all_seven_views(self):
        self.assertEqual(len(d.rows()),4)
        for row in d.rows():
            c=d.config(row);a=d.make_args(c,'cpu')
            self.assertEqual((a.epochs,a.label_epochs,a.pseudo_epochs,c['updates']),(50,33,17,11100))
            self.assertTrue(a.from_scratch);self.assertFalse(a.baseline_ckpt);self.assertFalse(a.teacher_ckpt)
            self.assertEqual(c['feature_parameters'],195041);self.assertEqual(len(c['test_views']),7)
            self.assertEqual(c['channel'],d.previous.config(d.previous.rows()[0])['channel'])

    def test_wrong_budget_head_and_formal_channel_are_rejected(self):
        c=d.config(d.rows()[0])
        for key,value in [('epochs',200),('default_head','new classifier'),('parent_checkpoint','E30.pth')]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):d.validate(bad)
        bad=copy.deepcopy(c);bad['channel']['reference_snr_db']=40
        with self.assertRaises(ValueError):d.validate(bad)

    def test_extra_teacher_supervises_low_confidence_urban_student(self):
        p=torch.tensor([[.99,.002,.002,.002,.002,.002]])
        mask=acceptance(p,'clean_confidence')
        student=torch.zeros(1,6,requires_grad=True)
        loss=soft_urban_ce(student,p,mask);loss.backward()
        self.assertTrue(mask.item());self.assertGreater(float(student.grad.norm()),0.)
        self.assertEqual(float(student.softmax(1).max()),float(torch.tensor(1/6)))

    def test_same_sample_weak_disagreement_is_rejected(self):
        p=torch.tensor([[.99,.002,.002,.002,.002,.002]])
        q=p.roll(1,1)
        self.assertFalse(acceptance(p,'same_sample_weak_agreement',weak_prob=q).item())
        with self.assertRaises(ValueError):acceptance(p.requires_grad_(),'clean_confidence')

    def test_weak_stream_repeats_only_for_same_named_seed(self):
        x=torch.ones(2,2,256)
        self.assertTrue(torch.equal(weak_iq(x,17),weak_iq(x,17)))
        self.assertFalse(torch.equal(weak_iq(x,17),weak_iq(x,18)))

    def test_source_teacher_calibration_freezes_state_and_restores_modes(self):
        model=FixedTeacher().train();x=torch.zeros(27000,2,2);y=torch.zeros(27000,dtype=torch.long)
        result=calibrate(model,[(x,y,None,{})],'cpu',dict(acceptance='clean_confidence',confidence_threshold=.95))
        self.assertTrue(result['passed']);self.assertEqual(result['count'],27000)
        self.assertEqual(int(model.training_calls),0);self.assertTrue(model.training)
        self.assertIsNone(model.scale.grad)

    def test_source_calibration_cannot_score_hidden_U(self):
        model=FixedTeacher();x=torch.zeros(2,2,2);y=torch.full((2,),-1)
        with self.assertRaisesRegex(ValueError,'legal V'):
            calibrate(model,[(x,y,None,{})],'cpu',dict(acceptance='clean_confidence',confidence_threshold=.95))

    def test_e30_update_count_does_not_close_e50(self):
        from .source import require_budget
        require_budget(11100,11100)
        with self.assertRaises(ValueError):require_budget(6660,6660)

    def test_actual_structure_and_joint_default_head_gradient(self):
        from .model import build
        from experiments.cvs_urban_sigtime.model import build as old_build
        c=d.config(next(r for r in d.rows() if r['arm']=='C3'))
        m=build(c,'cpu');old=old_build(d.previous.config(next(r for r in d.previous.rows() if r['arm']=='C3' and r['model_seed']==d.SEEDS[0])),'cpu')
        self.assertEqual(set(m.state_dict()),set(old.state_dict()))
        self.assertTrue(all(torch.equal(v,old.state_dict()[k]) for k,v in m.state_dict().items()))
        m.train();loss=F.cross_entropy(m.forward_identity_only(torch.randn(4,2,256))['tx_logits'],torch.arange(4));loss.backward()
        named=dict(m.named_parameters())
        for key in ['id_backbone.encoder.core.id_backbone.cls_head.weight','id_backbone.encoder.sequence_pool.views.0.local_projection.weight']:
            self.assertTrue(named[key].requires_grad);self.assertGreater(float(named[key].grad.norm()),0.)


if __name__=='__main__':unittest.main()
