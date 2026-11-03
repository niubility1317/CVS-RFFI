import importlib
import unittest
from unittest.mock import patch
import torch
from torch import nn


class FrozenReader(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(512,6);self.eval().requires_grad_(False)
    def features(self,x):return x.flatten(1)
    def classify_features(self,x):return self.head(x)


class DenoiserAdapterTest(unittest.TestCase):
    def adapter(self):
        try:return importlib.import_module('experiments.phase1_dura_posterior_denoiser.train_source')
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_posterior_denoiser.train_source':self.fail('flow training adapter is not implemented')
            raise

    def test_joint_loss_trains_restorer_without_reader_updates(self):
        torch.manual_seed(33);torch.set_num_threads(2);a=self.adapter();m=a.make_model();r=FrozenReader()
        x=torch.randn(3,2,256);anchor=x+.1*torch.randn_like(x);clean=x+.2*torch.randn_like(x);lv=torch.full((3,),-2.);groups=torch.tensor([4,4,6])
        out=m(x,anchor,lv);loss,terms=a.denoiser_training_objective(r,out,anchor,clean,r.features(clean).detach(),groups,x,lv)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(terms['denoising']))
        loss.backward();self.assertTrue(torch.isfinite(m.gate_output.weight.grad).all());self.assertGreater(float(m.gate_output.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in r.parameters()));a._current[0]=None

    def test_adapter_installs_and_restores_flow_training_contract(self):
        a=self.adapter();base=a.base
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','training_source_objective','USES_DURA','load_components','input_noise','predict_batch','extra_frozen_components')
        before=tuple(getattr(base,k) for k in names)
        def fail():
            self.assertIs(base.training_source_objective,a.denoiser_training_objective)
            self.assertTrue(base.USES_DURA)
            raise RuntimeError('synthetic failure')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):a.main()
        self.assertEqual(before,tuple(getattr(base,k) for k in names));self.assertIsNone(a._current[0])


if __name__=='__main__':unittest.main()
