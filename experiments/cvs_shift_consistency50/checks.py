import copy,json
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .objective import objective,statistics_penalty,shift,capture
from experiments.cvs_e20_cross_style.style import _paths
from .source import expected_counts,validate_counts

def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Expected forbidden config/role/budget rejection')

def run():
 torch.set_num_threads(2)
 for row in d.rows():
  c=d.config(row);assert c['epochs']==50 and c['updates']==11100
  counts=expected_counts(c);counts['accepted_U']=0;validate_counts(c,counts,56700 if c['source_U'] else 0)
  bad=dict(counts,identity_updates=11099);reject(lambda:validate_counts(c,bad,56700 if c['source_U'] else 0))
 c=d.config(d.rows()[0]);bad=copy.deepcopy(c);bad['epochs']=51;reject(lambda:d.validate(bad));bad=copy.deepcopy(c);bad['source_U']=True;reject(lambda:d.validate(bad))
 x=torch.arange(10).reshape(1,1,10).float();assert torch.equal(shift(x,2)[...,2:],x[...,:-2]) and not shift(x,2)[...,:2].any();assert torch.equal(shift(x,-2)[...,:-2],x[...,2:])
 g=torch.Generator().manual_seed(509);maps={'time_down':torch.randn(4,48,128,generator=g),'t2.pool':torch.randn(4,96,32,generator=g)}
 moved={k:shift(v,8//(2 if k=='time_down' else 8)) for k,v in maps.items()};penalty,_=statistics_penalty(maps,moved,8);assert abs(float(penalty))<1e-10
 delta=2;moved={k:(v*.9+.1).requires_grad_(True) for k,v in maps.items()};original={k:v.requires_grad_(True) for k,v in maps.items()};penalty,detail=statistics_penalty(original,moved,delta);penalty.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in original.values())
 x=torch.randn(8,2,256,generator=g);y=torch.arange(8)%6;records=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm))
  with patch('torch.load',side_effect=AssertionError('Scratch training must not load weights')):model=build(c,'cpu')
  pg,names=groups(model,c);model.eval();model.zero_grad(set_to_none=True)
  warm,wm,_=objective(model,x,y,None,None,None,None,c,1);assert wm['L_model_forwards']==1 and wm['teacher_forward_calls']==0 and wm['shift_statistics']==0
  class Bank:
   calls=0
   def update(self,ids,pseudo,confidence,passes):
    assert not confidence.requires_grad and not pseudo.requires_grad;self.calls+=1
    return torch.ones(len(ids),device=confidence.device,dtype=torch.bool),dict(synthetic=True)
  bank=Bank();ux=x.clone() if c['source_U'] else None
  loss,m,info=objective(model,x,y,ux,[f'synthetic-U-{i}' for i in range(8)] if ux is not None else None,torch.ones(8,dtype=torch.long) if ux is not None else None,bank if ux is not None else None,c,3331)
  loss.backward();assert torch.isfinite(loss) and sum(p.numel() for z in pg for p in z['params'])==177025
  assert m['L_model_forwards']==(1 if arm=='CE50' else 2) and m['teacher_forward_calls']==(2 if c['source_U'] else 0)
  assert model.head.weight.shape==(6,160) and not model._identity_input_policy['explicit_CFO_identity_input']
  if c['shift_statistics']['enabled']:
   assert [v['level'] for v in info['layers']]==list(d.LEVELS) and m['shift_statistics']>0
   with capture(model) as original:model(x)
   with capture(model) as moved:model(shift(x,2))
   stat_loss,_=statistics_penalty(original,moved,2)
   modules,paths=_paths(model);prefix=paths['time_down'].rsplit('.',1)[0];identity=modules[prefix]
   grad=torch.autograd.grad(stat_loss,[identity.t1.dw.weight,identity.t2.dw.weight],allow_unused=True)
   assert all(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in grad)
  if not c['source_U']:reject(lambda:objective(model,x,y,x,['x']*8,torch.ones(8),bank,c,3331))
  if c['source_U']:assert bank.calls==1 and m['U_packets']==8
  records.append(dict(arm=arm,L_forwards=m['L_model_forwards'],teacher_forwards=m['teacher_forward_calls'],statistics_layers=[v['level'] for v in info['layers']],U_truth_read=False,ordinarycos160=True))
 return dict(status='PASS',records=records,fixed50_and11100_budget=True,epoch51_rejected=True,noncircular_shift=True,exact_integer_grid_common_region_equivalence=True,shallow_middle_gradient_live=True,own_teacher_detached=True,U_forbidden_in_controls=True,training_checkpoint_not_loaded=True,target_access=False)

if __name__=='__main__':print(json.dumps(run()))
