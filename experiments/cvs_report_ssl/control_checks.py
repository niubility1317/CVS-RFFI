"""Reachability checks on native architecture; real parents in remote preflight."""
import argparse,copy,io,time
from unittest.mock import patch
import torch
from . import design as d
from .model import ReportIdentity,Teacher,groups
from .auxiliary import PhysicalHeads,repetition_plan,linear_objective,repetition_objective
from .source import objective,load_source
from experiments.cvs_stage2_ssl.frequency_optim import FrequencyAdamW

def graphs(base,L,U,seed):
 records=[]
 for arm in d.ARMS:
  c=d.config(dict(arm=arm,model_seed=seed,row_id=f'{arm}-s{seed}'))
  model=ReportIdentity(copy.deepcopy(base),c['enabled']['nonlinear']);model.train();pg,names=groups(model,c)
  heads=PhysicalHeads(c['enabled']['linear'],c['enabled']['repetition'])
  teacher=Teacher(model,.99) if arm!='C0' else None
  optimizer=FrequencyAdamW(model.base,pg,3e-6,c['optimizer']['weight_decay']) if c['enabled']['frequency'] else torch.optim.AdamW(pg,lr=1e-4)
  # Transient control copies only; no production run artifacts or identity updates.
  indices=torch.arange(8);ui=torch.arange(8)
  # Production auxiliary schedule is L32+U32; use 32 actual observations here.
  indices=torch.arange(32);ui=torch.arange(32)
  loss,parts,weighted,detail=objective(c,model,heads,L,indices,U,ui,teacher,None,892,'cpu')
  assert torch.isfinite(loss);loss.backward()
  assert any(p.grad is not None and float(p.grad.norm())>0 for p in model.id_backbone.parameters())
  assert all(p.grad is None for n,p in model.base.named_parameters() if not n.startswith('id_backbone.'))
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())
  before=copy.deepcopy(model.state_dict());optimizer.step()
  assert any(not torch.equal(before[n],p) for n,p in model.state_dict().items())
  if teacher:teacher.update(model);assert teacher.updates==1
  saved=copy.deepcopy(model.state_dict());restored=copy.deepcopy(model);restored.load_state_dict(saved,strict=True)
  with torch.no_grad():
   model.eval();restored.eval();assert torch.equal(model(L['x'][:2]),restored(L['x'][:2]))
  records.append(dict(arm=arm,loss=float(loss.detach()),mechanisms=detail,checkpoint_prediction_equal=True,domain_forward=False,transient_identity_updates=1))
 return records

def condition_check():
 from .conditional import ConditionalTeacher
 # Synthetic separable feature oracle verifies sufficient statistics and reads.
 # Real native snapshot is checked at source construction, not claimed here.
 class Oracle(torch.nn.Module):
  def forward(self,x):return x[:,0,:6]*20
 class T:pass
 t=T();t.model=Oracle()
 rx=torch.arange(5).repeat_interleave(1260);day=torch.arange(3).repeat_interleave(420).repeat(5);y=torch.arange(6).repeat_interleave(70).repeat(15)
 x=torch.zeros(6300,2,256);x[:,0,:6]=torch.nn.functional.one_hot(y,6).float();x[:,0,6]=rx*.01;x[:,0,7]=day*.01
 L=dict(x=x,rx=rx,day=day,y=y);U={k:v.clone() for k,v in L.items() if k!='y'}
 def feature(m,v):return torch.cat((v[:,0,:8],torch.ones(len(v),1)),1)
 with patch('experiments.cvs_report_ssl.conditional.features',side_effect=feature):
  ct=ConditionalTeacher(t,L,U,'cpu')
  q=ct.probabilities(x[:70],rx[:70],day[:70]);assert torch.isfinite(q).all() and bool((q.argmax(1)==y[:70]).all())
 assert float(ct.alphas.max())<=.2 and torch.all(ct.variances>0)
 assert all(abs(v-1/6)<1e-7 for v in ct.priors.flatten().tolist())
 assert 'y' not in U
 return dict(separate_L_U_statistics=True,fixed_L_priors=True,diagonal_shrinkage=True,alpha_cap=.2,statistical_feature_oracle_only=True)

def check():
 from .physical_checks import check as physical_check
 from experiments.cvs_stage2_decouple.design import make_args
 from experiments.cvs_phase1_stack.runtime import installed
 physical=physical_check();seed=d.SEEDS[0];pc=d.parent_config(seed);args=make_args(pc,'cpu');args.num_domains,args.input_len=5,256
 with installed(pc) as native:base=native.build_baseline_model(args,torch.device('cpu'))
 base._stage2_baseline_args=vars(args)
 x=torch.randn(64,2,256);L=dict(x=x,y=torch.arange(64)%6,rx=torch.arange(64)%5,day=torch.arange(64)%3)
 U={k:v for k,v in L.items() if k!='y'}
 result=graphs(base,L,U,seed);conditional=condition_check()
 from . import evaluate,dispatch
 with patch.object(evaluate,'snapshot',return_value=[d.config(d.rows()[0])]),patch.object(evaluate,'prediction_preflight',side_effect=ValueError('Incomplete prediction')),patch.object(d,'read',side_effect=AssertionError('Truth must stay closed')):
  try:evaluate.score()
  except ValueError:pass
  else:raise AssertionError('Incomplete predictions scored')
 assert dispatch.choose_gpu({0:dict(pids={1,2,3},free_mb=20000)}) is None
 for row in d.rows():d.validate(d.config(row))
 return dict(status='PASS',physical=physical,graphs=result,conditional=conditional,truth_last_rejection=True,total_registered_rows=24,target_access=False,production_updates=0,transient_copy_updates=8)

def remote_preflight():
 from experiments.cvs_stage2_decouple.common import build_model
 started=time.perf_counter();records=[]
 for seed in d.SEEDS:
  c=d.config(dict(arm='C8',model_seed=seed,row_id=f'C8-s{seed}'))
  L,U,V,contract,provenance=load_source(c);base=build_model(c['parent_contract'],'cpu')
  records.append(dict(seed=seed,parent_provenance=provenance,graphs=graphs(base,L,U,seed),U_count=len(U['ids']),V_count=len(V['ids']),old_auxiliary_caches_reused=False))
 return dict(status='PASS',rows=records,target_access=False,production_updates=0,transient_copy_updates=24,elapsed_seconds=time.perf_counter()-started)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(2);value=remote_preflight() if a.remote_preflight else check();d.write(a.output,value);print('PASS report SSL native reachability')
