"""Focused native operator/protocol-negative checks, no model or real IQ."""
import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .core import paired,stream,state,equal_state
from .metrics import summarize
P=Path(sys.argv.pop(1)) if len(sys.argv)>1 else Path(__file__).resolve().parents[2]/'experiments/adv3b02_xuc/code/DataAugmentation.py'
s=importlib.util.spec_from_file_location('native_taper46_checks',P);native=importlib.util.module_from_spec(s);sys.modules[s.name]=native;s.loader.exec_module(native)
def augment(p=1):
 return native.RFFIAugmentor(p_dac=0,p_pa=.14,enable_class_signature=False,p_time_shift=.35,max_time_shift=32,p_amp_scale=.45,amp_min=.9,amp_max=1.1,p_phase_rot=.45,p_cfo=.35,cfo_max=.0004,p_phase_noise=.3,phase_noise_sigma_max=.006,p_awgn=.4,snr_min_db=20,snr_max_db=36,p_multipath=.18,mp_taps_min=2,mp_taps_max=4,mp_delay_max=4,p_dc_offset=.3,dc_offset_max=.02,p_bandedge_taper=p,taper_alpha_min=.02,taper_alpha_max=.1)
def fixture():
 y=np.arange(6300)%6;rx=np.repeat(np.array(d.RX),1260);raw=np.eye(6,dtype=np.float32)[y]*30;raw=np.stack([raw,raw]);full=np.repeat(raw[:,None],2,axis=1);bypass=full.copy();trigger=np.ones((2,6300),bool)
 return raw,full,bypass,y,rx,trigger
class Checks(unittest.TestCase):
 def setUp(self):
  self.x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(514));self.y=torch.arange(128)%6
 def test_exact_native_full_and_RNG(self):
  aug=augment();before=state('cpu');f,b,meta=paired(native,aug,self.x,self.y,61,14)
  self.assertTrue(equal_state(before,state('cpu')))
  with stream(61,14,'cpu'):expected=aug(self.x.clone(),labels=self.y,no_pa=False)
  self.assertTrue(torch.equal(f,expected));self.assertTrue(meta['triggered']);self.assertGreater(float((f-b).square().sum()),0)
 def test_actual_window_geometry(self):
  w=native._tukey_window(256,.08,'cpu',torch.float32);self.assertEqual(float(w[0]),0);self.assertEqual(float(w[-1]),0);self.assertTrue(torch.equal(w[80:160],torch.ones(80)));self.assertTrue(((w>=0)&(w<=1)).all())
 def test_nontrigger_zero_effect(self):
  f,b,m=paired(native,augment(1e-30),self.x,self.y,63,0);self.assertFalse(m['triggered']);self.assertTrue(torch.equal(f,b))
 def test_exception_restores_native_and_RNG(self):
  aug=augment();orig=native._tukey_window;before=state('cpu')
  class Failure:
   enable_class_signature=False;p_dac=0;p_bandedge_taper=1
   def __call__(self,x,**kw):native._tukey_window(256,.08,'cpu',torch.float32);raise RuntimeError('synthetic')
  with self.assertRaises(RuntimeError):paired(native,Failure(),self.x,self.y,1,9)
  self.assertIs(native._tukey_window,orig);self.assertTrue(equal_state(before,state('cpu')))
 def test_protocol_and_input_negatives(self):
  for field,value in [('L',5040),('source_RX_holdout',1),('U_used',True),('target_access',True),('expected_NN',1),('ordinary_epoch',20),('new_head',True),('automatic_training',True)]:
   c=d.config();c[field]=value
   with self.assertRaises(ValueError):d.validate(c)
  for x,y in [(self.x[:,:,:255],self.y),(self.x*float('nan'),self.y),(self.x,self.y.float()),(self.x,self.y+6)]:
   with self.assertRaises(ValueError):paired(native,augment(),x,y,1,0)
  aug=augment();aug.enable_class_signature=True
  with self.assertRaises(ValueError):paired(native,aug,self.x,self.y,1,0)
 def test_qualification_positive_and_no_gain(self):
  raw,f,b,y,rx,tr=fixture()
  for r in d.RX:
   ix=np.flatnonzero(rx==r)[:10];f[:,:,ix,:]=0;f[:,:,ix,(y[ix]+1)%6]=30
  out=summarize(raw,f,b,y,rx,tr);self.assertTrue(out['source_qualified']);self.assertEqual(out['cohort_count'],6300)
  out=summarize(raw,b,b,y,rx,tr);self.assertFalse(out['source_qualified'])
 def test_not_one_RX_or_draw_rescue(self):
  raw,f,b,y,rx,tr=fixture();ix=np.flatnonzero(rx==1)[:80];f[:,:,ix,:]=0;f[:,:,ix,(y[ix]+1)%6]=30
  self.assertFalse(summarize(raw,f,b,y,rx,tr)['checks']['each_model_at_least4positive_RX'])
 def test_saved_nontrigger_rejected(self):
  raw,f,b,y,rx,tr=fixture();tr[:]=False;b[0,0,0,0]+=1
  with self.assertRaises(ValueError):summarize(raw,f,b,y,rx,tr)
if __name__=='__main__':unittest.main()
