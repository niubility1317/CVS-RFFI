"""Bounded source-only integration and actual-parent preflight checks."""
import argparse,copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .objective import objective
from .source import expected_counts
from experiments.cvs_early_geometry.checks import fixture


def fixture_parent(c):
 from experiments.cvs_early_geometry import design as old
 from experiments.cvs_early_geometry.model import build as fresh
 oc=old.config(next(r for r in old.rows() if r['arm']=='CE' and r['model_seed']==c['model_seed']))
 base=fresh(oc,'cpu').base
 provenance=dict(status='VERIFIED',checkpoint=c['parent_checkpoint'],source_roles='EXACT_MATCH',training_target_access=False,epoch=20,optimizer_steps=4440,seed=c['model_seed'],scratch_only=True,ancestors=[],auxiliary_inheritance=False)
 return dict(model=base.state_dict(),baseline_args=base._stage2_baseline_args),provenance


def check():
 torch.set_num_threads(2);torch.manual_seed(81);L,U=fixture();li=torch.arange(32)
 c0=d.config(d.rows()[0]);parent,provenance=fixture_parent(c0);results=[];warm=None
 from .warmup import head_warmup
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));d.validate(c);json.dumps(c)
  with patch('experiments.cvs_e20_budget_fishr.model.verify',return_value=provenance),patch('torch.load',return_value=parent):model=build(c,'cpu')
  assert all(torch.equal(v,parent['model'][k]) for k,v in model.base.state_dict().items())
  assert not hasattr(model,'teacher')
  if arm=='MULTI':
   with tempfile.TemporaryDirectory() as temp:warm=head_warmup(c,model,L,'cpu',Path(temp))
   assert all(torch.equal(v,parent['model'][k]) for k,v in model.base.state_dict().items())
  pg,_=groups(model,c);model.train();rng=torch.get_rng_state()
  early,_,_,detail=objective(c,model,None,L,li,None,None,444,'cpu')
  torch.set_rng_state(rng);plain=F.cross_entropy(model(L['x'][li]),L['y'][li]);assert torch.equal(early,plain)
  calls=[];original=model.forward_identity_only
  def counted(x):calls.append(len(x));return original(x)
  model.forward_identity_only=counted
  value,parts,weighted,detail=objective(c,model,None,L,li,U if c['unlabeled'] else None,li if c['unlabeled'] else None,1332,'cpu')
  expected=(3 if c['unlabeled'] else 1)+(2 if c['multilevel']['mixstyle_l'] else 1 if c['multilevel']['enabled'] else 0)
  assert len(calls)==expected,(arm,len(calls),expected)
  if c['multilevel']['enabled']:
   for name in ('aux_ce','fishr') if c['multilevel']['fishr_mode']!='none' else ('aux_ce',):
    grad=torch.autograd.grad(weighted[name],pg[0]['params'],retain_graph=True,allow_unused=True)
    assert sum(float(g.abs().sum()) for g in grad if g is not None)>0,(arm,name)
  value.backward();assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  assert model.training and all(m.training for m in model.modules())
  del model.forward_identity_only
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'own.pt'
   payload=dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,epoch=20,total_epochs=40,optimizer_steps=4440,parent_checkpoint=c['parent_checkpoint'],parent_provenance=provenance,checkpoint_selection='final_only',target_access=False,run_id=c['run_id'],candidate_id=c['row_id'])
   torch.save(payload,path);restored=build(c,'cpu',path);model.eval();restored.eval()
   with torch.no_grad():assert torch.equal(model(L['x'][:4]),restored(L['x'][:4]))
   torch.save(dict(payload,parent_checkpoint='wrong'),path)
   try:build(c,'cpu',path)
   except ValueError:pass
   else:raise AssertionError('Wrong parent accepted')
  results.append(dict(arm=arm,forwards=len(calls),expected=expected_counts(c),details=detail))
 for row in d.rows():
  c=d.config(row);assert d.learning_rate(c,1)==.0002 and abs(d.learning_rate(c,20)-1e-6)<1e-12
 for key,value in [('epochs',40),('model_initialization','E200'),('target_access',True)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Changed configuration accepted')
 from .dispatch import require_scored
 marker=dict(status='SCORED_COMPLETE',models=24,views=1,query_count_per_view=168000,prediction_count=4032000,metric_records=336,day_metric_records=96,truth_last=True,independent_recount='VERIFIED')
 require_scored(marker,24)
 try:require_scored(dict(marker,prediction_count=168000),24)
 except ValueError:pass
 else:raise AssertionError('Incomplete matrix accepted')
 return dict(status='PASS',synthetic_parent_strict_load=True,unchanged_parent_before_updates=True,ordinary_CE_exact=True,U_has_no_y=True,checkpoint_roundtrip=True,wrong_parent_rejected=True,head_warmup=warm,rows=results,matrix=24,target_access=False)


def preflight():
 from .source import load_source
 from experiments.cvs_e20_foundation.verify import verify
 from experiments.cvs_equivariant_identity.precision import numerical_context
 result=check();evidence=[]
 with numerical_context(d.FULL_FP32_POLICY):
  for seed in d.SEEDS:
   reference=verify(seed,build_smoke=True)
   c=d.config(next(r for r in d.rows() if r['arm']=='JOINT' and r['model_seed']==seed))
   L,U,V,contract=load_source(c);model=build(c,'cpu');model.train();idx=torch.arange(32)
   value,_,_,detail=objective(c,model,None,L,idx,U,idx,1332,'cpu');value.backward()
   assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   evidence.append(dict(seed=seed,L=len(L['ids']),U=len(U['ids']),V=len(V['ids']),real_source_backward=True,E20_parent=reference,training_parent_loaded=True,details=detail))
 result.update(actual_source_evidence=evidence,target_access=False);return result


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-preflight',action='store_true');a=p.parse_args();d.write(a.output,preflight() if a.remote_preflight else check())
