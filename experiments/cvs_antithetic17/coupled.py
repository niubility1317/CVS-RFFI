"""Couple native Gaussian noise drivers before unchanged receiver processing."""
import math
import numpy as np
from . import design as d
def tensor_input(array):
 import torch
 a=np.asarray(array,dtype=np.float32)
 return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape)
class ExactNoise:
 def __init__(self,draws):
  a=np.asarray(draws,dtype=np.float64)
  if a.shape!=(256,2) or not np.isfinite(a).all():raise ValueError('Finite original256x2 Gaussian driver required')
  self.draws=a.copy();self.calls=0
 def normal(self,*args,**kw):
  if args or kw!={'size':(256,2)} or self.calls:raise ValueError('Exactly one native noise draw required')
  self.calls+=1;return self.draws.copy()
def process(ch,residual,cfg,hw,x,seed,draws):
 if cfg.processing_route!='residual' or cfg.mode!='post_sync' or cfg.equalization_enabled or cfg.reference_snr_db!=30. or cfg.boundary!='reflect':raise ValueError('Original complete NoEQ/reference30/reflect required')
 obs=residual.ResidualChannel(cfg,seed,hw);driver=ExactNoise(draws);obs.latent.rngs['noise']=driver;y,m=obs.process(x)
 if driver.calls!=1 or m['channel_equalization_applied'] or not np.isfinite(y).all():raise ValueError('Native finite complete observation required')
 return y,m
def three(ch,residual,cfg,hw,x,seed,e1,e2):
 values=[process(ch,residual,cfg,hw,x,seed,e) for e in (e1,e2,-e1)]
 a,b,c=(m for _,m in values)
 fields=('geometry','receiver','state_start','state_end','quality_snr_db','receiver_lock_status','residual_scale','output_frequency_hz','residual_kernel_energy','channel_equalization_applied','phase_tracking_applied')
 for field in fields:
  if a[field]!=b[field] or a[field]!=c[field]:raise ValueError('Non-noise dependency changed:'+field)
 if not math.isclose(a['measured_added_noise_power_pre_agc'],c['measured_added_noise_power_pre_agc'],rel_tol=1e-12,abs_tol=1e-15):raise ValueError('Sign pair changed preAGC noise power')
 return values
def variance(gram,repeats):
 g=np.asarray(gram,dtype=np.float64)
 if g.shape!=(3*repeats,3*repeats) or not np.isfinite(g).all():raise ValueError('Complete gradient Gram required')
 out={}
 for name,other in [('independent',1),('antithetic',2)]:
  a=np.zeros((repeats,3*repeats))
  for i in range(repeats):a[i,3*i]=a[i,3*i+other]=.5
  k=a@g@a.T;v=float((np.trace(k)-k.sum()/repeats)/(repeats-1))
  if v < -1e-9*max(1.,float(np.trace(k))):raise ValueError('Invalid variance')
  out[name]=max(0.,v)
 if out['independent']<=0:raise ValueError('Variance reference zero;no evidence')
 return dict(out,ratio=out['antithetic']/out['independent'])
