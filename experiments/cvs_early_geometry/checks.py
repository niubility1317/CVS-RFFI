import argparse,copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .objective import objective,weak
from .source import expected_counts
from experiments.cvs_stage2_rx_actions.receiver import ReceiverActions

def fixture():
 cells=torch.cartesian_prod(torch.arange(6),torch.tensor([1,3,4,6,8]),torch.tensor([1,2,3])).repeat_interleave(70,dim=0)
 x=torch.randn(len(cells),2,256)
 L=dict(x=x,y=cells[:,0],rx=cells[:,1],day=cells[:,2],ids=[f'L{i}' for i in range(len(x))])
 U={k:v for k,v in L.items() if k!='y'};U=dict(U,ids=[f'U{i}' for i in range(len(x))])
 return L,U

def check():
 torch.set_num_threads(2);torch.manual_seed(81)
 from .multilevel_checks import run as check_multilevel
 multilevel=check_multilevel()
 from experiments.cvs_u_geometry_audit.checks import check as geometry_checks
 geometry_checks();L,U=fixture();li=torch.arange(32);first=None;results=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));d.validate(c);json.dumps(c)
  # Scratch creation must be possible even when every checkpoint load is forbidden.
  with patch('torch.load',side_effect=AssertionError('Scratch attempted checkpoint load')):model=build(c,'cpu')
  state={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith('base.')}
  if first is None:first=state
  else:assert all(torch.equal(v,first[k]) for k,v in state.items())
  pg,_=groups(model,c);bank=ReceiverActions(c,L);model.train();rng=torch.get_rng_state()
  early,parts,_,detail=objective(c,model,bank,L,li,None,None,444,'cpu')
  torch.set_rng_state(rng);plain=F.cross_entropy(model(L['x'][li]),L['y'][li]);assert torch.equal(early,plain)
  assert detail['U_packets']==detail['physical_L']==0
  calls=[];original=model.forward_identity_only
  def counted(x):calls.append(len(x));return original(x)
  model.forward_identity_only=counted
  value,parts,weighted,detail=objective(c,model,bank,L,li,U if c['unlabeled'] else None,li if c['unlabeled'] else None,1332,'cpu')
  expected_forwards=(4 if c['rx_action']['enabled'] else 3 if c['unlabeled'] else 1)+(2 if c['multilevel']['mixstyle'] else 1 if c['multilevel']['enabled'] else 0)
  assert len(calls)==expected_forwards,(arm,len(calls),expected_forwards)
  if c['losses']['feature']:
   g=torch.autograd.grad(weighted['feature'],pg[0]['params'],retain_graph=True,allow_unused=True)
   assert sum(float(x.abs().sum()) for x in g if x is not None)>0
  if c['losses']['margin']:
   g=torch.autograd.grad(weighted['margin'],pg[0]['params'],retain_graph=True,allow_unused=True)
   assert sum(float(x.abs().sum()) for x in g if x is not None)>0
  if c['multilevel']['enabled']:
   for name in ('aux_ce','fishr') if c['multilevel']['fishr'] else ('aux_ce',):
    grad=torch.autograd.grad(weighted[name],pg[0]['params'],retain_graph=True,allow_unused=True)
    assert sum(float(x.abs().sum()) for x in grad if x is not None)>0,(arm,name)
  value.backward();assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  assert model.training and all(m.training for m in model.modules())
  del model.forward_identity_only
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'own.pt';torch.save(dict(model=model.state_dict(),config=c,epoch=20,optimizer_steps=4440,parent_checkpoint=None,checkpoint_selection='final_only',target_access=False),path)
   restored=build(c,'cpu',path)
   model.eval();restored.eval()
   with torch.no_grad():assert torch.equal(model(L['x'][:4]),restored(L['x'][:4]))
  results.append(dict(arm=arm,forwards=len(calls),feature_gradient=bool(c['losses']['feature']),margin_gradient=bool(c['losses']['margin']),expected=expected_counts(c)))
 for row in d.rows():
  c=d.config(row);assert d.learning_rate(c,1)==.0002 and abs(d.learning_rate(c,20)-1e-6)<1e-12
 for key,value in [('epochs',40),('model_initialization','E200'),('target_access',True)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Changed configuration accepted')
 from .dispatch import require_scored
 marker=dict(status='SCORED_COMPLETE',models=30,views=1,query_count_per_view=168000,prediction_count=5040000,metric_records=420,day_metric_records=120,truth_last=True,independent_recount='VERIFIED')
 require_scored(marker,30)
 try:require_scored(dict(marker,prediction_count=168000),30)
 except ValueError:pass
 else:raise AssertionError('Incomplete matrix accepted')
 return dict(status='PASS',scratch_loads_no_checkpoint=True,identical_seed_initializations=True,ordinary_CE_exact=True,U_has_no_y=True,checkpoint_roundtrip=True,geometry=geometry_checks(),multilevel=multilevel,rows=results,matrix=30)

def preflight():
 from .source import load_source
 from experiments.cvs_e20_foundation.verify import verify
 from experiments.cvs_equivariant_identity.precision import numerical_context
 result=check();evidence=[]
 with numerical_context(d.FULL_FP32_POLICY):
  for seed in d.SEEDS:
   reference=verify(seed,build_smoke=True)
   c=d.config(next(r for r in d.rows() if r['arm']=='MIX_FISHR_SM' and r['model_seed']==seed))
   L,U,V,contract=load_source(c);model=build(c,'cpu');pg,_=groups(model,c);bank=ReceiverActions(c,L);model.train();idx=torch.arange(32)
   value,parts,weighted,detail=objective(c,model,bank,L,idx,U,idx,1332,'cpu');value.backward()
   assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   evidence.append(dict(seed=seed,L=len(L['ids']),U=len(U['ids']),V=len(V['ids']),real_source_backward=True,E20_reference=reference,training_parent_loaded=False))
 result.update(actual_source_evidence=evidence,target_access=False);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-preflight',action='store_true');a=p.parse_args();d.write(a.output,preflight() if a.remote_preflight else check())
