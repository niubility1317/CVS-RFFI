"""Focused arithmetic, protocol-negative, real network and native concat checks."""
import tempfile,unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .objective import paired_margin,soft_urban_ce
from .channel import PairedChannel,imports,identities
from .controls import compare

def cfg(arm):return d.config(next(r for r in d.rows() if r['arm']==arm))
def meta(n):
    return dict(base_index=torch.arange(n),rx_i=torch.ones(n,dtype=torch.long),
        day_i=torch.ones(n,dtype=torch.long),eq_i=torch.ones(n,dtype=torch.long))

class Checks(unittest.TestCase):
    def test_margin_detached_and_scale(self):
        clean=torch.tensor([[4.,1.,0.],[0.,4.,1.]],requires_grad=True)
        sat=torch.zeros_like(clean,requires_grad=True);labels=torch.tensor([0,1]);q=torch.tensor([.2,1.])
        loss=paired_margin(clean,sat,labels,q,scale=1.);loss.backward()
        self.assertIsNone(clean.grad);self.assertGreater(float(sat.grad.abs().sum()),0)
        self.assertAlmostEqual(float(loss),float(paired_margin(clean*30,sat*30,labels,q,scale=30.)))
        self.assertEqual(float(paired_margin(clean,clean,labels,q,scale=1.)),0.)
    def test_all_real_labels_have_satellite_CE(self):
        x=torch.zeros(2,3,requires_grad=True);y=torch.tensor([0,2])
        q=torch.zeros(2);c=torch.eye(3)[:2].requires_grad_()
        (F.cross_entropy(x,y)+paired_margin(c,x,y,q)).backward()
        self.assertTrue(bool((x.grad.abs().sum(1)>0).all()))
    def test_wrong_clean_reference_zero(self):
        clean=torch.tensor([[0.,9.,0.]]);sat=torch.zeros(1,3,requires_grad=True)
        self.assertEqual(float(paired_margin(clean,sat,torch.tensor([0]),torch.ones(1))),0.)
    def test_hidden_metadata_no_TX(self):
        ids,sessions=identities(meta(3));self.assertEqual(len(ids),3);self.assertEqual(len(set(sessions)),1)
        self.assertNotIn('tx',sessions[0]);self.assertRaises(ValueError,identities,dict(rx_i=[1]))
    def test_refresh_and_stable_hardware(self):
        x=torch.randn(3,2,256,generator=torch.Generator().manual_seed(91));ch=PairedChannel(cfg('C1'))
        a,am=ch.apply(x,meta(3),1,1);b,bm=ch.apply(x,meta(3),2,1)
        self.assertFalse(torch.equal(a,b));self.assertEqual(am[0]['receiver'],bm[0]['receiver'])
        self.assertEqual({r['processing_route'] for r in am},{'residual'})
        self.assertFalse(any(r['channel_equalization_applied'] for r in am))
        self.assertRaises(ValueError,ch.apply,x,meta(3),1,1,eta=.7)
    def test_original_residual_control_parity_and_EQ_noise(self):
        ch,r=imports();rng=np.random.default_rng(42);x=rng.normal(size=256)+1j*rng.normal(size=256);x,_=ch.normalize_input(x)
        for eq in (False,True):
            c=ch.Config(fs_hz=25e6,scenario='practical_mid_urban',processing_route='residual',equalization_enabled=eq)
            receiver=ch.receiver_for_session(c,9,'rx1/day1');values,m,original=compare(x,c,90,receiver)
            self.assertEqual(values['full'].shape,(256,));self.assertEqual(m['no_noise/fixed']['receiver_lock_status'],original['receiver_lock_status'])
            self.assertEqual(m['no_noise/total']['receiver_lock_status'],'locked')
            self.assertLess(m['full']['boundary_interior_relative_error'],1e-5)
            if not eq:self.assertAlmostEqual(m['full']['noise_coloration_dynamic_range_db'],0.)
    def test_frozen_protocol_negative(self):
        c=cfg('C3');d.validate(c)
        for key,value in [('epochs',200),('parent_checkpoint','E200.pth'),('separate_head_fit',True),('test_views',['clean'])]:
            changed=dict(c);changed[key]=value;self.assertRaises(ValueError,d.validate,changed)
        changed=dict(c,channel=dict(c['channel'],reference_snr_db=40));self.assertRaises(ValueError,d.validate,changed)
    def test_soft_source_teacher_never_urban_student_gate(self):
        logits=torch.zeros(3,6,requires_grad=True);prob=F.one_hot(torch.tensor([1,2,3]),6).float()
        loss=soft_urban_ce(logits,prob,torch.tensor([True,False,True]));loss.backward()
        self.assertGreater(float(logits.grad[[0,2]].abs().sum()),0);self.assertEqual(float(logits.grad[1].abs().sum()),0)
        ch=PairedChannel(cfg('C4'));_,records=ch.apply(torch.randn(12,2,256),meta(12),21,1,view='U')
        self.assertTrue(all(r['scenario'] in ('practical_mid_urban','practical_low_urban') for r in records))
    def test_real_model_paired_update_and_native_concat(self):
        from .model import build
        from .observer import Observer
        from experiments.cvs_phase1_overlay.model import native_modules
        torch.set_num_threads(2);c=cfg('C3');m=build(c,'cpu').train();x=torch.randn(8,2,256);y=torch.arange(8)%6
        with tempfile.TemporaryDirectory() as tmp:
            obs=Observer(c);obs.initialize(m,c,Path(tmp));a=d.make_args(c,'cpu')
            out=m.forward_identity_only(x);ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            adjustment,logs=obs.observe(model=m,args=a,epoch=1,batch_idx=1,clean_iq=x,labels=y,clean_meta=meta(8),
                clean_output=out,clean_ce=ce,unlabeled_iq=None,unlabeled_meta=None,teacher_prob=None,weak_mask=None)
            old=[p.detach().clone() for p in obs.params];(ce+adjustment).backward()
            opt=torch.optim.AdamW(m.parameters(),lr=.0002);opt.step()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in obs.params))
            self.assertTrue(all(not torch.equal(p,z) for p,z in zip(obs.params,old)));obs.trace.close()
            (Path(tmp)/'baseline').mkdir()
            bc=cfg('B0');bo=Observer(bc);bo.initialize(m,bc,Path(tmp)/'baseline');n=native_modules();original=n._prepare_concat_sat_batch_for_training
            with bo.native_scope(n):
                bo.meta=meta(8);xx,yy,dd,view,info=n._prepare_concat_sat_batch_for_training(None,x,y,None,args=d.make_args(bc,'cpu'),epoch=1,batch_idx=1)
                self.assertEqual(xx.shape,(16,2,256));self.assertTrue(torch.equal(yy[:8],yy[8:]));self.assertEqual(info['expanded'],1.)
                ll=torch.randn(16,6,requires_grad=True)
                self.assertAlmostEqual(float(F.cross_entropy(ll,yy)),float((F.cross_entropy(ll[:8],y)+F.cross_entropy(ll[8:],y))/2),places=6)
            self.assertIs(n._prepare_concat_sat_batch_for_training,original);bo.trace.close()

if __name__=='__main__':unittest.main()
