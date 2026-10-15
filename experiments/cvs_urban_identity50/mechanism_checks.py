import unittest
import torch
from . import mechanism as m
from . import design as d,model as models


class MechanismChecks(unittest.TestCase):
    def test_wrong_TX_is_excluded_from_correct_pair_and_no_classifier_fit(self):
        labels=torch.tensor([0,1,2,3,4,5])
        x=torch.zeros(6,160);x[:,:6]=torch.eye(6)
        r=m.identity_distances(x,x,x,labels)
        self.assertTrue(torch.equal(r['clean_pair_cosine'],torch.ones(6)))
        self.assertTrue(torch.equal(r['closest_wrong_TX_cosine'],torch.zeros(6)))
        self.assertTrue(torch.equal(r['correct_minus_wrong_cosine'],torch.ones(6)))
        self.assertTrue(torch.equal(r['unit_pair_distance'],torch.zeros(6)))
        with self.assertRaises(ValueError):m.identity_distances(x,x,x,torch.full((6,),-1))

    def test_actual_frozen_path_is_state_preserving_and_default_logits_match(self):
        model=models.build(d.config(d.rows()[0]),'cpu').eval().requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()};x=torch.randn(6,2,256)
        with torch.no_grad():
            value=m.forward(model,x,batch=3);original=model.forward_identity_only(x)
        self.assertTrue(torch.allclose(value['joint'],original['id_feat_joint'],atol=1e-6))
        self.assertTrue(torch.allclose(value['logits'],original['tx_logits'],atol=1e-6))
        self.assertEqual(set(value['branches']),{'base','PA','reference','signature'})
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        self.assertFalse(any(module._forward_hooks for module in model.modules()))

    def test_error_repairs_and_new_errors_use_the_same_fixed_clean_set(self):
        items=[]
        for fixed,old,new in ((True,False,True),(True,True,False),(False,False,True)):
            items.append(dict(fixed_A_clean_correct=fixed,A_satellite_correct=old,correct=new,margin=0.,
                **{k:0. for k in ('clean_pair_cosine','EMA_pair_cosine','closest_wrong_TX_cosine','correct_minus_wrong_cosine','unit_pair_distance')}))
        r=m.summarize(items)
        self.assertEqual(r['A_errors_repaired'],1);self.assertEqual(r['A_correct_new_errors'],1)
        self.assertEqual(r['fixed_A_clean_correct_count'],2);self.assertEqual(r['fixed_A_retention'],.5)


if __name__=='__main__':unittest.main()
