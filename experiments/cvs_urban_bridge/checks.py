"""Coupled-physics and actual SIG_TIME graph checks; no data or fitted head."""
from dataclasses import replace
import copy,unittest
import numpy as np
import torch
from .views import coupled
from . import signature
from experiments.cvs_urban_sigtime.channel import imports
from experiments.cvs_urban_branch50 import design as parent
from experiments.cvs_urban_branch50.model import build


class Physics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch,cls.residual=imports();rng=np.random.default_rng(91)
        cls.x=rng.normal(size=256)+1j*rng.normal(size=256);cls.x/=np.sqrt(np.mean(abs(cls.x)**2))

    def config(self,eq=False):
        cfg=self.ch.Config(fs_hz=25e6,scenario='practical_low_urban',processing_route='residual',
            mode='post_sync',equalization_enabled=eq)
        return cfg,self.ch.receiver_for_session(cfg,2026092701,'rx1:day2:eq1')

    def test_full_exact_and_no_gain_or_state_reestimate(self):
        cfg,receiver=self.config();before=copy.deepcopy(receiver)
        expected,meta=self.residual.ResidualChannel(cfg,531,receiver).process(self.x)
        waves,actual,proof,traces=coupled(self.x,cfg,531,receiver)
        self.assertTrue(np.array_equal(waves['full'],expected));self.assertEqual(receiver,before)
        self.assertEqual(actual['agc_gain_db'],meta['agc_gain_db']);self.assertTrue(proof['frozen_fields_identical'])
        self.assertFalse(proof['output_RMS_renormalized']);self.assertFalse(proof['original_recording_noise_removed'])
        self.assertAlmostEqual(proof['bridge_shared_signal_to_added_noise_db']-proof['full_signal_to_added_noise_db'],6.,places=10)
        gain=10**(meta['agc_gain_db']/20)
        self.assertTrue(np.array_equal(waves['no_added_noise'],(traces['pre_agc_signal']*gain).astype(np.complex64)))
        self.assertFalse(np.array_equal(waves['bridge_shared'],waves['bridge_independent']))

    def test_unit_noise_matches_full_and_rejects_bad_dose(self):
        cfg,receiver=self.config()
        waves,_,_,_=coupled(self.x,cfg,533,receiver,noise_improvement_db=0.)
        self.assertTrue(np.array_equal(waves['bridge_shared'],waves['full']))
        for value in (-1.,float('nan'),31.):
            with self.assertRaises(ValueError):coupled(self.x,cfg,533,receiver,noise_improvement_db=value)
        with self.assertRaises(ValueError):coupled(self.x,replace(cfg,mode='raw'),533,receiver)

    def test_EQ_colored_noise_and_context_support_are_preserved(self):
        cfg,receiver=self.config(True)
        waves,meta,proof,traces=coupled(self.x,cfg,535,receiver,quality_snr_db=30.)
        self.assertTrue(meta['channel_equalization_applied']);self.assertGreater(len(traces['equalizer_kernel']),1)
        spectrum=abs(np.fft.fft(traces['equalizer_kernel'],1024))**2
        self.assertGreater(spectrum.max()/max(spectrum.min(),1e-20),1.1)
        gain=10**(meta['agc_gain_db']/20);signal=traces['pre_agc_signal'];noise=traces['pre_agc_noise']
        expected=((signal+10**(-6/20)*noise)*gain).astype(np.complex64)
        self.assertTrue(np.array_equal(waves['bridge_shared'],expected));self.assertTrue(proof['EQ_noise_coloration_preserved'])
        cfg=replace(cfg,boundary='require_context')
        with self.assertRaises(ValueError):coupled(self.x,cfg,536,receiver)
        _,meta,proof,_=coupled(self.x,cfg,536,receiver,preceding_context=self.x,quality_snr_db=30.)
        self.assertEqual(meta['boundary'],'provided_context');self.assertTrue(proof['frozen_kernels_identical'])


class Signature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);cls.model=build(parent.config(parent.rows()[0]),'cpu').eval()

    def test_exact_semantic_groups_and_fixed_time_excluded(self):
        d=torch.arange(112,dtype=torch.float32)[None,:];g=signature.split(d)
        self.assertEqual([g[k].shape[-1] for k in signature.KEYS],[6,15,6])
        used=set(range(6))|set(signature.UU)|set(signature.UT)
        self.assertEqual(used,set(range(28))-{6})
        x=torch.zeros(2,112,requires_grad=True);target=torch.ones_like(x)
        loss,_=signature.grouped_loss(x,target,{k:1. for k in signature.KEYS})
        gradient=torch.autograd.grad(loss,x)[0].reshape(2,4,28)
        self.assertTrue(torch.equal(gradient[:,:,6],torch.zeros(2,4)))
        scales=signature.teacher_scales(torch.zeros(2,112));self.assertTrue(all(v>=1e-3-1e-9 for v in scales.values()))

    def test_actual_local_and_t2_graph_reaches_no_W_gain_or_end_projection(self):
        model=copy.deepcopy(self.model);teacher=copy.deepcopy(model).requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()};x=torch.randn(3,2,256)
        with torch.no_grad(),signature.capture(teacher) as t:teacher.forward_identity_only(x)
        with signature.capture(model) as s:model.forward_identity_only(x+.2*torch.randn_like(x))
        loss,_=signature.grouped_loss(s['descriptor'],t['descriptor'],signature.teacher_scales(t['descriptor']))
        named=list(model.named_parameters());grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
        norms={name:0. if g is None else float(g.norm()) for (name,_),g in zip(named,grads)}
        self.assertTrue(any(v>0 for k,v in norms.items() if 't2.' in k))
        self.assertTrue(any(v>0 for k,v in norms.items() if 'local_projection.weight' in k))
        self.assertTrue(all(v==0 for k,v in norms.items() if 'output_projection.' in k or 'gain' in k or k.endswith('cls_head.weight')))
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        logits=model.forward_identity_only(x)['tx_logits'];y=torch.arange(3)
        head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
        self.assertGreater(float(torch.autograd.grad(torch.nn.functional.cross_entropy(logits,y),head)[0].norm()),0.)


if __name__=='__main__':unittest.main()
