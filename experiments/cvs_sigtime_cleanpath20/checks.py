import copy,unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.cvs_feature_sequence50.features import SequenceReference
from experiments.cvs_reference_identity.model import ReferenceCVS
from .mechanism import install

class Block(nn.Module):
    def __init__(self):super().__init__();self.drop=nn.Dropout(.1)
    def forward(self,x):return self.drop(x)
class Pool(nn.Module):
    def forward(self,h):self.observed=h;return h.mean((1,2))[:,None].expand(-1,160)

def fixture():
    ref=SequenceReference.__new__(SequenceReference);nn.Module.__init__(ref)
    ref.core=nn.Module();ref.core.id_backbone=nn.Module();ref.core.id_backbone.t2=Block();ref.sequence_pool=Pool()
    return ref
def base_features(ref,x):
    ref.post=ref.core.id_backbone.t2(x)
    return torch.zeros(len(x),160,device=x.device)

class PathTraining(unittest.TestCase):
    def test_clean_signature_keeps_original_main_mask_and_rng(self):
        original=fixture().train();clean=copy.deepcopy(original);install(clean,'CLEAN_SIG');x=torch.ones(16,6,32)
        with patch.object(ReferenceCVS,'features',base_features):
            torch.manual_seed(17);original.features(x);state=torch.random.get_rng_state()
            torch.manual_seed(17);clean.features(x)
        self.assertTrue(torch.equal(clean.post,original.post));self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(clean.sequence_pool.observed,x));self.assertFalse(torch.equal(original.sequence_pool.observed,x))
        self.assertTrue(bool((clean.post==0).any()))
    def test_signature_gradient_is_not_detached_or_dropout_masked(self):
        model=fixture().train();install(model,'CLEAN_SIG');x=torch.ones(8,6,32,requires_grad=True)
        with patch.object(ReferenceCVS,'features',base_features):model.features(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue(bool((x.grad!=0).all()))
        self.assertTrue(torch.equal(x.grad,x.grad[:,:,:1].expand_as(x.grad)))
    def test_eval_equal_and_state_head_unmodified(self):
        original=fixture().eval();x=torch.randn(8,6,32)
        for mode in ('ELEMENT','NONE','CLEAN_SIG'):
            model=copy.deepcopy(original);before=copy.deepcopy(model.state_dict());install(model,mode)
            with patch.object(ReferenceCVS,'features',base_features):
                self.assertTrue(torch.equal(original.features(x),model.features(x)))
            self.assertTrue(all(torch.equal(value,model.state_dict()[name]) for name,value in before.items()))
            self.assertEqual(set(model.state_dict()),set(before))
    def test_hook_removed_after_error_and_no_call_local_state_persists(self):
        model=fixture();install(model,'CLEAN_SIG')
        with patch.object(ReferenceCVS,'features',side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):model.features(torch.ones(2,6,32))
        self.assertEqual(len(model.core.id_backbone.t2.drop._forward_pre_hooks),0)
        self.assertNotIn('h',model.__dict__)
    def test_fixed_original_site_and_mode(self):
        with self.assertRaises(ValueError):install(fixture(),'LOCKED')
        model=fixture();model.core.id_backbone.t2.drop=nn.Dropout(.2)
        with self.assertRaises(ValueError):install(model,'CLEAN_SIG')
    def test_fixed_seed_no_target_and_recipe(self):
        from . import design as d
        for key,value in (('model_seed',2026092701),('test_views',['clean']),('head_fit_updates',1)):
            c=copy.deepcopy(d.config('NONE'));c[key]=value
            with self.assertRaises(ValueError):d.validate(c)
    def test_primary_and_control_confirmation_are_separate(self):
        from . import design as d
        def row(errors,ce):return dict(source_health=True,exports=dict(FINAL=dict(count=1260,errors=errors,mean_CE=ce)))
        r=dict(ELEMENT=row(500,2.),NONE=row(400,1.),CLEAN_SIG=row(450,1.5));q=d.qualify(r)
        self.assertFalse(q['CLEAN_SIG_primary_qualified']);self.assertTrue(q['NONE_independent_confirmation_passed'])
        r['CLEAN_SIG']=row(399,.9);self.assertTrue(d.qualify(r)['CLEAN_SIG_primary_qualified'])
        r['CLEAN_SIG']['exports']['FINAL']['count']=100
        with self.assertRaises(ValueError):d.qualify(r)

if __name__=='__main__':unittest.main(verbosity=2)
