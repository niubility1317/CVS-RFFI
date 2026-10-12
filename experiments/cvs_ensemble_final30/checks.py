import argparse,json
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import torch
from . import design as d
from .model import build_member,groups,ProbabilityEnsemble

def run(device='cpu'):
 from .contracts import verified_component_contract
 reference=dict(role_ids={'L_s':['synthetic-L'],'V':['synthetic-V']});actual=dict(reference,native_role_comparison='EXACT_MATCH')
 if verified_component_contract([actual,dict(actual)],reference)!=actual:raise AssertionError('Runtime source role evidence must be preserved')
 for bad in (dict(reference),dict(actual,role_ids={'L_s':['wrong-source'],'V':['synthetic-V']})):
  try:verified_component_contract([bad,actual],reference)
  except ValueError:pass
  else:raise AssertionError('Missing runtime evidence or wrong role must block')
 torch.set_num_threads(2);torch.manual_seed(1);x=torch.randn(16,2,256,device=device);y=torch.arange(16,device=device)%6;rows=[]
 row=d.rows()[0];members=[]
 for member in (1,2):
  c=d.member_config(row,member);d.validate_member(c)
  with patch('torch.load',side_effect=AssertionError('No new initialization weights')):m=build_member(c,device)
  pg,_=groups(m,c);pp=[p for g in pg for p in g['params']];loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward()
  if not any(p.grad is not None and p.grad.abs().sum()>0 for p in pp) or any(not torch.isfinite(p.grad).all() for p in pp if p.grad is not None):raise AssertionError('Actual final member gradients')
  rows.append(dict(member=member,init_seed=c['member_init_seed'],trainable_parameters=sum(p.numel() for p in pp)));members.append(m.eval())
 p0={n:p for n,p in members[0].named_parameters() if p.requires_grad};p1=dict(members[1].named_parameters())
 if all(torch.equal(p,p1[n]) for n,p in p0.items()):raise AssertionError('Actual identity initializer streams must differ')
 c=dict(d.member_config(row,1),member=0,member_init_seed=row['model_seed']);zero=build_member(c,device).eval()
 with torch.no_grad():
  output=ProbabilityEnsemble([zero,*members]).eval()(x)
  expected=torch.stack([m(x).softmax(1) for m in (zero,*members)]).mean(0)
  if not torch.allclose(output.exp(),expected,atol=1e-6) or output.shape!=(16,6):raise AssertionError('Final fixed mixture identity')
 for key,value in [('source_U',True),('training_target_access',True),('epochs',20)]:
  c=deepcopy(d.member_config(row,1));c[key]=value
  try:d.validate_member(c)
  except ValueError:pass
  else:raise AssertionError('Illegal final member config')
 return dict(status='PASS',device=device,primary_seed_count=3,members=rows,exact_mixture=True,no_checkpoint_initialization=True,negative_cases=5,runtime_contract_preserved=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
