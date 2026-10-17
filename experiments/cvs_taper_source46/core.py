"""Consume the exact native gate and draws; bypass only the time window."""
from contextlib import contextmanager
import random
import numpy as np
import torch
@contextmanager
def stream(seed,call,device):
 py=random.getstate();nr=np.random.get_state();dev=torch.device(device);ds=[dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type=='cuda' else []
 with torch.random.fork_rng(devices=ds):
  try:
   s=(int(seed)+1000003*int(call))%(2**32-1);random.seed(s);np.random.seed(s);torch.manual_seed(s)
   if ds:torch.cuda.manual_seed_all(s)
   yield
  finally:random.setstate(py);np.random.set_state(nr)
def state(device):
 d=torch.device(device)
 return random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state(d).clone() if d.type=='cuda' else None
def equal_state(a,b):
 return a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]) and ((a[3] is None and b[3] is None) or (a[3] is not None and b[3] is not None and torch.equal(a[3],b[3])))
def paired(module,aug,x,y,seed,call):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not torch.isfinite(x).all() or y.shape!=(len(x),) or y.dtype!=torch.long or ((y<0)|(y>=6)).any():raise ValueError('Finite labeled IQ256 only')
 if aug.enable_class_signature or aug.p_dac!=0 or not 0<aug.p_bandedge_taper<=1:raise ValueError('Actual ordinary/no class-signature/DAC only')
 if not hasattr(module,'_tukey_window'):raise ValueError('Native Tukey function missing')
 original=module._tukey_window;tapes=[];ends=[];results=[]
 try:
  for bypass in (False,True):
   tape=[]
   def window(*args,**kwargs):
    w=original(*args,**kwargs)
    if w.shape!=(256,) or not torch.isfinite(w).all() or (w<0).any() or (w>1).any():raise ValueError('Invalid native window')
    tape.append(w.detach().clone());return torch.ones_like(w) if bypass else w
   module._tukey_window=window
   with stream(seed,call,x.device):
    out=aug(x.clone(),labels=y,no_pa=False);ends.append(state(x.device))
   if out.shape!=x.shape or not torch.isfinite(out).all():raise FloatingPointError('Invalid ordinary IQ')
   results.append(out);tapes.append(tape)
 finally:module._tukey_window=original
 if len(tapes[0])!=len(tapes[1]) or len(tapes[0])>1 or not all(torch.equal(a,b) for a,b in zip(*tapes)) or not equal_state(*ends):raise ValueError('Native gate/window or other random draws differ')
 if not tapes[0] and not torch.equal(*results):raise ValueError('Nontrigger must be exact zero effect')
 return results[0],results[1],dict(triggered=bool(tapes[0]),window_points_below_one=int((tapes[0][0]<1).sum()) if tapes[0] else 0,random_draws_equal=True)
def decisions(model,x):
 z=model.forward_identity_only(x)['tx_logits']
 if z.shape!=(len(x),6) or not torch.isfinite(z).all():raise FloatingPointError('Finite unchanged six-class default logits required')
 return z
