import argparse,json,math
from pathlib import Path
from copy import deepcopy
import torch
from . import design as d
from .model import build,groups
from experiments.cvs_nocfo_foundation.model import build as native
def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(54);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;rows=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));m=build(c,x.cpu(),device);pg,n=groups(m,c);m.eval();reference=native(c['source_loader_config'],device).eval()
  with torch.no_grad():a=reference(x);b=m(x)
  if not torch.allclose(a,b,rtol=1e-5,atol=1e-5) or m._current_q is not None:raise AssertionError('Zero-init FiLM must be native identity and clear packet context')
  m.train();loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward()
  params=[p for g in pg for p in g['params']]
  if not any(p.grad is not None and p.grad.abs().sum()>0 for p in params) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise AssertionError('FiLM source gradients')
  if arm!='CE30':
   if [v['level'] for v in m.layer_info]!=list(d.LEVELS):raise AssertionError('Actual shallow/middle layers missing')
   if not all(gen[-1].weight.grad is not None and gen[-1].weight.grad.abs().sum()>0 for gen in m.maps.values()):raise AssertionError('Actual shallow/middle conditional maps not trained')
  rows.append(dict(arm=arm,parameters=sum(p.numel() for p in params),layers=m.layer_info))
 for key,value in [('held_rx',99),('source_U',True),('epochs',20),('training_target_access',True)]:
  c=deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:pass
  else:raise AssertionError('Illegal quality sourceconfig')
 return dict(status='PASS',device=device,rows=rows,initial_identity=True,all_classes_shared=True,scale_bounds=[math.exp(-.2),math.exp(.2)],shift_bound=.1,negative_cases=4)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
