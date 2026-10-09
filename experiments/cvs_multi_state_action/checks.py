"""Control-plane, physical role and view checks; no query/truth input."""
import argparse
from pathlib import Path
import torch
from experiments.cvs_multi_state_action import design as d
from experiments.cvs_multi_state_action.runner import joint,subset,composition_indices
from experiments.cvs_multi_state_action.views import channel_batch,WEAK
from experiments.cvs_multi_action_audit.checks import channel_bridge_check

def checks(device='cpu'):
 torch.set_num_threads(2)
 assert len(d.rows())==36 and len(d.audit_rows())==4
 for r in d.rows():
  c=d.config(r);d.validate(c);a=d.make_args(c,device)
  assert a.from_scratch and not a.baseline_ckpt and not a.teacher_ckpt
  assert a.pseudo_temporal_mode=='batch_neighbor' and a.phase1_lr_schedule=='cosine'
  assert a.concat_sat_ce_only and a.lambda_sat_cls==.68 and a.lambda_sat_cons==0
  assert a.a1_source_screen_only and a.a1_periodic_target_start==0
  try:d.validate(dict(c,target_access=True))
  except ValueError:pass
  else:raise AssertionError('Changed protocol config accepted')
 x=torch.randn(12,2,256);data=dict(x=x,y=torch.arange(12)%6,rx=torch.arange(12)%2,day=torch.ones(12).long(),ids=list(map(str,range(12))),condition=['clean']*12)
 j=joint(data)
 assert j['ids'][:12]==j['ids'][12:] and j['condition'][12:]==['source_practical_mid']*12
 assert subset(j,j['y']!=0)['y'].eq(0).sum()==0
 ci=composition_indices(j)
 assert {key[3] for key in ci}=={'clean','source_practical_mid'} and {key[0] for key in ci}==set(range(6))
 for scene in WEAK:
  y,cfg=channel_batch(x,scene,torch.Generator().manual_seed(41))
  y2,_=channel_batch(x,scene,torch.Generator().manual_seed(41))
  assert y.shape==x.shape and torch.isfinite(y).all() and torch.equal(y,y2)
 channel_bridge_check(device)
 return dict(status='PASS',registered_rows=36,audit_rows=4,source_contract_preserved=True,
  role_preserving_views=True,weak_views=WEAK,no_query=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);a=p.parse_args();v=checks(a.device);d.write(a.output,v);print(v)
