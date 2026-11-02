import importlib
import unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.phase1_dura_feedback.model import pack_context
from experiments.phase1_dura_tripleblend.test_model import FakeExpert


class Reader(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(512,6);self.eval().requires_grad_(False)
    def features(self,x):return x.flatten(1)
    def classify_features(self,x):return self.head(x)


class AdapterTest(unittest.TestCase):
    def adapter(self):
        try:return importlib.import_module('experiments.phase1_dura_tripleblend.train_source')
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_tripleblend.train_source':self.fail('three-input fusion training adapter is not implemented')
            raise

    def test_source_loss_unpacks_anchor_and_trains_gate(self):
        torch.manual_seed(99);torch.set_num_threads(2);a=self.adapter()
        from experiments.phase1_dura_tripleblend.model import TripleBlend
        m=TripleBlend(FakeExpert());r=Reader();x=torch.randn(3,2,256);anchor=x+.1*torch.randn_like(x);clean=x+.2*torch.randn_like(x)
        packed=pack_context(anchor,torch.randn(3,160),torch.randn(3,6));out=m(x,packed,torch.full((3,),-2.))
        loss,_=a.triple_source_objective(r,out,packed,clean,r.features(clean).detach(),torch.tensor([4,4,6]));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.gate_output.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in r.parameters()));self.assertTrue(all(p.grad is None for p in m.expert.parameters()))

    def test_adapter_freezes_all_components_and_restores_hooks(self):
        a=self.adapter();base=a.base
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA','load_components','input_noise','predict_batch','extra_frozen_components','extra_frozen_model_components')
        old=tuple(getattr(base,k) for k in names)
        def fail():
            m=nn.Module();m.expert=nn.Linear(2,2);d=object();reader=object()
            self.assertTrue(base.USES_DURA);self.assertEqual(base.extra_frozen_components(d,reader),(d,reader));self.assertEqual(base.extra_frozen_model_components(m),(m.expert,))
            raise RuntimeError('synthetic failure')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):a.main()
        self.assertEqual(old,tuple(getattr(base,k) for k in names))


if __name__=='__main__':unittest.main()
