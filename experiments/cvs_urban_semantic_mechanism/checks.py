import copy,unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from .probe import reference_gap,auxiliary,gradient_record,summarize,KEYS,PairedChannel
from experiments.cvs_urban_semantic50.objective import mask
from experiments.cvs_urban_branch50 import branches as b
from experiments.cvs_urban_branch50.model import build
from experiments.cvs_urban_identity50.features import boundary_parameters


class Checks(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(137);n=8
        return dict(y=torch.tensor([0,1,0,1,0,1,0,1]),rx=torch.tensor([1,1,3,3,4,4,6,6]),
            day=torch.ones(n,dtype=torch.long),a=torch.randn(n,160),k=torch.randn(n,160))

    def test_matched_reference_gap_matches_exact_triplets_and_rejects_illegal_source(self):
        v=self.inputs();actual=reference_gap(v['a'],v['k'],v['y'],v['rx'],v['day'])
        sim=F.normalize(v['a'],dim=1)@F.normalize(v['k'],dim=1).T
        valid=mask(v['y'],v['rx'],v['day']);count=valid.sum((1,2))
        exact=((sim[:,:,None]-sim[:,None,:])*valid).sum((1,2))/count.clamp_min(1)
        self.assertTrue(torch.allclose(actual['gap'],exact,atol=1e-6))
        self.assertTrue(torch.equal(actual['triplets'],count))
        missing=reference_gap(v['a'],v['k'],v['y'],torch.ones(8,dtype=torch.long),v['day'])
        self.assertEqual(float(missing['gap'].abs().sum()),0.)
        for labels,rx in ((torch.full((8,),-1),v['rx']),(v['y'],torch.full((8,),14))):
            with self.assertRaises(ValueError):reference_gap(v['a'],v['k'],labels,rx,v['day'])

    def test_actual_paired_clean_anchor_gradients_and_frozen_state(self):
        torch.set_num_threads(2);v=self.inputs();model=build(d.training.parent.config(d.training.parent.rows()[0]),'cpu').eval()
        before={k:t.clone() for k,t in model.state_dict().items()}
        out,values=b.forward(model,torch.randn(16,2,256));clean={k:t[:8] for k,t in values.items()};urban={k:t[8:] for k,t in values.items()}
        target=dict(logits=out['tx_logits'][:8].detach(),features={k:clean[k].detach() for k in KEYS})
        recipe=dict(semantic=dict(coefficients={a:dict(coefficient=.2) for a in ('D3','D4')}),
            positive=dict(coefficients=dict(J=dict(coefficient=.3))))
        for arm in ('D3','D4'):
            extra,_=auxiliary(model,arm,clean,urban,target,v['y'],v['rx'],v['day'],torch.ones(8),recipe)
            gc,gu=torch.autograd.grad(extra,[clean['J'],urban['J']],allow_unused=True,retain_graph=True)
            if arm=='D3':self.assertIsNone(gc)
            else:self.assertGreater(float(gc.norm()),0.)
            self.assertGreater(float(gu.norm()),0.)
            grads=torch.autograd.grad(extra,list(boundary_parameters(model).values()),allow_unused=True,retain_graph=True)
            self.assertTrue(all(t is None or float(t.abs().max())==0. for t in grads))
        record=gradient_record(model,out,values,target,v['y'],v['rx'],v['day'],torch.ones(8),'D4',recipe,'practical_low_urban',0)
        self.assertGreater(record['global_vs_satellite']['auxiliary_weighted_norm'],0.)
        self.assertTrue(all(value==0. for value in record['auxiliary_boundary_norms'].values()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(t,model.state_dict()[k]) for k,t in before.items()))

    def test_source_eligibility_rejects_unfrozen_or_target_contact(self):
        def read(path):
            name=path.name
            if name=='completion.json':return dict(config={},epoch=49,optimizer_steps=0,target_access=False,head_fit_updates=0)
            if name=='source_frozen.json':return dict(config={},epoch=49,selection='not_final',target_access=False)
            if name=='inner_source_contract.json':return {}
            raise AssertionError(name)
        with patch.object(d,'read',read):
            with self.assertRaisesRegex(ValueError,'source final50'):d.eligible_rows()

    def test_fixed_A0_set_repairs_and_destruction_are_separate(self):
        rows=[]
        for fixed,base,correct in ((True,False,True),(True,True,False),(True,False,False),(False,False,True)):
            rows.append(dict(fixed_A0_clean_correct=fixed,A0_satellite_correct=base,correct=correct,margin=.1,
                urban_to_clean_reference_gap=.2,clean_to_urban_reference_gap=.3,
                identity={k:dict(clean_pair_cosine=.4) for k in KEYS}))
        result=summarize(rows)
        self.assertEqual(result['count'],4);self.assertEqual(result['fixed_A0_clean_correct_count'],3)
        self.assertEqual(result['A0_errors_repaired'],1);self.assertEqual(result['A0_correct_new_errors'],1)
        self.assertAlmostEqual(result['fixed_A0_retention'],1/3);self.assertEqual(result['accuracy'],.5)
        self.assertEqual(d.FORWARD_PACKETS,36722)

    def test_common_engine_accepts_its_actual_original_A0_configuration(self):
        engine=PairedChannel(d.training.parent.config(d.training.parent.rows()[0]))
        self.assertEqual(set(engine.configs),set(d.training.SCENES))
        for cfg in engine.configs.values():
            self.assertEqual(cfg.processing_route,'residual');self.assertFalse(cfg.equalization_enabled)
        self.assertTrue(str(engine.ch.__file__).replace('\\','/').endswith('adv3b02_xuc/code/leo_practical/channel.py'))


if __name__=='__main__':unittest.main()
