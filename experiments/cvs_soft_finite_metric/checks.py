import argparse,json
from pathlib import Path
from copy import deepcopy
import torch
from .metric import fit,logits
from . import design as d
def run():
 torch.set_num_threads(2);torch.manual_seed(55);z=torch.randn(120,160);z=torch.nn.functional.normalize(z,dim=1);y=torch.arange(120)%6;delta=.01*torch.randn(4,120,160)
 s=fit(z,y,delta);zero=fit(z,y,torch.zeros_like(delta))
 if not torch.allclose(zero['MAHALANOBIS'],zero['SOFT_FINITE'],rtol=1e-8,atol=1e-8) or zero['hard_rank']!=0:raise AssertionError('Zero uncertainty limit')
 for arm in ('MAHALANOBIS','SOFT_FINITE','HARD_FINITE'):
  if torch.linalg.eigvalsh((s[arm]+s[arm].T)/2).min()<-1e-8:raise AssertionError('Non-PSD metric')
  if not torch.isfinite(logits(z,s,arm)).all():raise AssertionError('Nonfinite classifier')
 p=torch.tensor([2,0,5,1,4,3]);inv=p.argsort();other=fit(z,inv[y],delta)
 for arm in ('COSINE_PROTO','MAHALANOBIS','SOFT_FINITE','HARD_FINITE'):
  if not torch.allclose(logits(z,s,arm)[:,p],logits(z,other,arm),rtol=1e-7,atol=1e-7):raise AssertionError('Class label permutation changed classifier')
 for key,value in [('target_access',True),('U_access',True),('identity_updates',1),('soft_covariance_multiplier',2.)]:
  c=deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:pass
  else:raise AssertionError('Illegal metric config')
 return dict(status='PASS',class_permutation=True,PSD=True,zero_uncertainty_limit=True,negative_cases=4)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))
