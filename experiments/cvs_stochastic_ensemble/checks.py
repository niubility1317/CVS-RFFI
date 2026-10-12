import argparse,json
from pathlib import Path,PurePosixPath
from unittest.mock import patch
from copy import deepcopy
import torch
from . import design as d
from .model import build,build_member,groups,probabilities
from experiments.cvs_nocfo_foundation.model import build as native
def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(4);x=torch.randn(16,2,256,device=device);y=torch.arange(16,device=device)%6;models=[];records=[]
 for row in d.rows():
  if d.config(row)['baseline_ref']!=str(PurePosixPath(d.PROJECT)/'runs'/'20261012-phase1-quality-film-source-manysig-s2026092701-r01'/d.baseline_id(row['held_rx'])):raise AssertionError('Portable source comparison path')
 for member in (1,2):
  c=d.config(next(r for r in d.rows() if r['member']==member))
  with patch('torch.load',side_effect=AssertionError('No initialization checkpoint')):m=build(c,x,device)
  pg,names=groups(m,c);pp=[p for g in pg for p in g['params']]
  if len({id(p) for p in pp})!=len(pp) or {id(p) for p in pp}!={id(p) for p in m.parameters() if p.requires_grad}:raise AssertionError('Native member optimizer ownership')
  loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward()
  if not any(p.grad is not None and p.grad.abs().sum()>0 for p in pp) or not all(torch.isfinite(p.grad).all() for p in pp if p.grad is not None):raise AssertionError('Member gradients')
  models.append(m);records.append(dict(member=member,initialization_seed=c['member_init_seed'],parameters=sum(p.numel() for p in pp)))
 trainable={n:p for n,p in models[0].named_parameters() if p.requires_grad};other=dict(models[1].named_parameters())
 if all(torch.equal(p,other[n]) for n,p in trainable.items()):raise AssertionError('Actual trainable identity tensors require distinct member initialization; frozen nuisance differences do not count')
 c=d.config(d.rows()[0]);c=dict(c,member=0,member_init_seed=d.SEED);zero=build_member(c,device).eval();ref=native(c['source_loader_config'],device).eval()
 with torch.no_grad():
  if not torch.allclose(zero(x),ref(x),atol=1e-5,rtol=1e-5):raise AssertionError('Member0 must match native architecture and initializer')
  z=[m.eval()(x) for m in (zero,*models)];p=probabilities(z)
  if not torch.allclose(p.sum(1),torch.ones(len(p),device=device),atol=1e-6):raise AssertionError('Probability mixture')
  perm=torch.tensor([3,0,5,2,1,4],device=device)
  if not torch.allclose(probabilities([v[:,perm] for v in z]),p[:,perm]):raise AssertionError('All classes share same ensemble rule')
  if not torch.allclose(probabilities([z[0]]*3),z[0].softmax(1)):raise AssertionError('Identical members identity')
 for key,value in [('held_rx',99),('source_U',True),('epochs',20),('training_target_access',True),('member_init_seed',d.SEED)]:
  c=deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:pass
  else:raise AssertionError('Illegal ensemble source configuration')
 return dict(status='PASS',device=device,members=records,negative_cases=5,portable_paths=True,member0_native_equivalence=True,equal_probabilities=True,class_permutation=True,no_checkpoint_initialization=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
