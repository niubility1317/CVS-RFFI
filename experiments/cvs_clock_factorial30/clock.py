"""A bounded affine sampling grid; no wrap seam, gain/RMS change or TX label input."""
import numpy as np
import torch
from . import design as d

class Clock:
 def __init__(self):self.calls=0;self.packets=0;self.max_ppm=0.;self.max_grid_offset=0.;self.squared_change_sum=0.
 def apply(self,x,meta,epoch,step):
  if x.shape!=(64,2,256) or not bool(torch.isfinite(x).all()) or epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1:raise ValueError('Only finite assigned L64 in the cumulative E21..30/step4441..6660')
  keys=('eq_i','rx_i','day_i','sig_i','base_index')
  if any(k not in meta or len(meta[k])!=64 for k in keys):raise ValueError('Original physical metadata required;no labels accepted')
  v={k:meta[k].detach().cpu().tolist() for k in keys}
  if set(v['rx_i'])-set(d.FIT) or set(v['day_i'])-{1,2,3} or set(v['eq_i'])!={1}:raise ValueError('Original full source roles required')
  from leo_practical.channel import stable_seed
  ppm=[]
  for i in range(64):
   pid=tuple(int(v[k][i]) for k in keys);g=np.random.default_rng(stable_seed(d.CLOCK['seed'],'clock/G/e'+str(epoch)+'/step'+str(step),pid));ppm.append(float(g.uniform(-80.,80.)))
  y=resample(x,torch.tensor(ppm,device=x.device,dtype=torch.float64))
  self.calls+=1;self.packets+=len(x);self.max_ppm=max(self.max_ppm,max(map(abs,ppm)));self.max_grid_offset=max(self.max_grid_offset,max(map(abs,ppm))*1e-6*127.5);self.squared_change_sum+=float((y-x).square().sum())
  return y
 def summary(self):return dict(calls=self.calls,packets=self.packets,max_abs_ppm=self.max_ppm,max_abs_grid_offset_samples=self.max_grid_offset,mean_IQ_MSE=self.squared_change_sum/max(1,self.packets*512),parameters=dict(d.CLOCK),new_TX_impairments=False,global_RMS=False,CFO=False)

from experiments.cvs_clock_factorial20.clock import resample
