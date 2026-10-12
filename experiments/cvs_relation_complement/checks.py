import argparse,json
from pathlib import Path,PurePosixPath
from copy import deepcopy
from unittest.mock import patch
import torch
from . import design as d
from .model import build,groups,relations
from experiments.cvs_nocfo_foundation.model import build as native

def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(71);x=torch.randn(32,2,256,device=device);y=torch.arange(32,device=device)%6;rows=[]
 for row in d.rows():
  expected=str(PurePosixPath(d.PROJECT)/'runs'/'20261012-phase1-quality-film-source-manysig-s2026092701-r01'/d.baseline_id(row['held_rx']))
  if d.config(row)['baseline_ref']!=expected:raise AssertionError('Preregistered comparison path must match Linux runtime on every host')
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm))
  with patch('torch.load',side_effect=AssertionError('No pretrained construction')):m=build(c,x,device)
  pg,names=groups(m,c);params=[p for g in pg for p in g['params']]
  if len({id(p) for p in params})!=len(params) or {id(p) for p in params}!={id(p) for p in m.parameters() if p.requires_grad}:raise AssertionError('Optimizer ownership')
  ref=native(c['source_loader_config'],device).eval();m.eval()
  with torch.no_grad():
   if not torch.allclose(m(x),ref(x),atol=1e-5,rtol=1e-5):raise AssertionError('Initial native equivalence')
   v=relations(x,m.projection,arm);phase=.7;gain=1.6;z=torch.complex(x[:,0],x[:,1])*gain*torch.exp(x.new_tensor(phase)*1j);xp=torch.stack((z.real,z.imag),1)
   changed=relations(xp,m.projection,arm)
   if arm=='COMPLEX_REL_BRANCH' and not torch.allclose(v,changed,atol=2e-5,rtol=2e-5):raise AssertionError('Weak gain/phase relation invariance')
   if arm=='RAW_REL_BRANCH' and torch.allclose(v,changed,atol=1e-5,rtol=1e-5):raise AssertionError('Ordinary relation negative control')
   if not torch.isfinite(m(torch.zeros_like(x))).all():raise AssertionError('Zero packet must remain evaluated')
  opt=torch.optim.AdamW(pg,lr=.0002)
  for step in range(2):
   m.train();opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward()
   if m.gain.grad is None or not m.gain.grad.abs().sum() or not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):raise AssertionError('Fusion gradients')
   if step==1 and not any(p.grad is not None and p.grad.abs().sum()>0 for p in m.branch.parameters()):raise AssertionError('Actual branch gets trained')
   opt.step()
  rows.append(dict(arm=arm,parameters=sum(p.numel() for p in params),initial_native=True,branch_gradient=True,no_checkpoint_load=True))
 for key,value in [('held_rx',99),('source_U',True),('epochs',20),('training_target_access',True)]:
  c=deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:pass
  else:raise AssertionError('Illegal complementary source config')
 if rows[0]['parameters']!=rows[1]['parameters']:raise AssertionError('Branch capacity mismatch')
 return dict(status='PASS',device=device,rows=rows,negative_cases=4,weak_channel_invariance=True,whole_model_invariant=False,zero_packet_finite=True,POSIX_baseline_refs=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
