import argparse,json
from pathlib import Path
from copy import deepcopy
import torch
from . import design as d
def run():
 torch.manual_seed(33);z=torch.randn(120,160);z=torch.nn.functional.normalize(z,dim=1);y=torch.arange(120)%6;mu=torch.stack([z[y==k].mean(0) for k in range(6)]);p=torch.tensor([2,0,5,1,4,3]);inv=p.argsort();other=torch.stack([z[inv[y]==k].mean(0) for k in range(6)])
 a=z@torch.nn.functional.normalize(mu,dim=1).T;b=z@torch.nn.functional.normalize(other,dim=1).T
 if not torch.allclose(a[:,p],b):raise AssertionError('Prototype class permutation')
 for row in d.rows():
  c=d.config(row);d.validate(c)
  if c['source_loader_config']['model_seed']!=row['model_seed'] or c['source_loader_config']['unlabeled'] or c['checkpoint_sources']:raise AssertionError('Source model seed/roles')
 for key,value in [('model_seed',2026092701),('target_access',True),('source_U',True),('epochs',21)]:
  c=deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:pass
  else:raise AssertionError('Illegal confirmation config')
 return dict(status='PASS',rows=10,source_seed_configs_exact=True,prototype_permutation=True,negative_cases=4)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))
