import numpy as np
import subprocess,sys
from pathlib import Path
import torch
from torch import nn
from .probe import Probe,descriptor
from .scatter import halves,scatter

def main():
 subprocess.run([sys.executable,'-X','utf8','-B','-c','from statistics import median; assert median([1,2,3])==2'],cwd=Path(__file__).parent,check=True)
 y=np.tile(np.repeat(np.arange(6),20),15);rx=np.repeat(np.repeat([1,3,4,6,8],3),120);day=np.tile(np.repeat([1,2,3],120),5);ids=[str(i).zfill(5) for i in range(1800)];sp=halves(y,rx,day,ids,20261019821)
 x=np.repeat(y[:,None],144,axis=1).astype(float);r=scatter(x,sp);assert abs(r['ratio']-1)<1e-12 and r['positive_RX_day_count']==15
 assert scatter(np.ones((1800,144)),sp)['ratio']==0
 x=np.zeros((1800,144))
 for a,b,groups in sp:
  for label,(i,j) in enumerate(groups):x[i]=label;x[j]=-label
 assert scatter(x,sp)['cross_half_TX_scatter_sum']<0
 core=nn.Module();core.time_fuse=nn.Identity();core.time_down=nn.AvgPool1d(2);p=Probe(core);g=torch.arange(2*48*256,dtype=torch.float32).reshape(2,48,256)/10000;z=core.time_down(core.time_fuse(g));values=p.take();assert np.asarray(values['pooled']).shape==(2,144) and np.asarray(values['removed']).shape==(2,144) and p.calls==1;p.close()
 negative=0
 for fn in (lambda:scatter(np.zeros((1800,143)),sp),lambda:halves(y,rx,day,ids[:-1],1),lambda:halves(y[:-1],rx[:-1],day[:-1],ids[:-1],1),lambda:scatter(np.full((1800,144),np.nan),sp)):
  try:fn()
  except ValueError:negative+=1
  else:raise AssertionError('Malformed source diagnostic accepted')
 print(dict(status='PASS',direct_script_stdlib_median_regression=True,synthetic_full_model_NN=0,synthetic_pool_activations=2,real_checkpoint=0,real_IQ=0,target_access=False,negative_tests=negative,analytic_identity_scatter_ratio=1,negative_scatter_preserved=True))
if __name__=='__main__':main()
