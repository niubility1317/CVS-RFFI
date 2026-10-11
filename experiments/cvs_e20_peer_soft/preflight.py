"""Bounded source-only integration and exact original E20 input verification."""
import argparse,copy,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from .model import build,groups
from .source import expected_counts
from .provenance import expected_sources,teacher_path


def local():
 from experiments.cvs_e20_budget_fishr.checks import fixture_parent
 parent,proof=fixture_parent(d.config(d.rows()[0]));rows=[]
 for row in d.rows():
  c=d.validate(d.config(row));counts=expected_counts(c)
  assert counts['identity_updates']==4440 and counts['auxiliary_steps']==(999 if c['multilevel']['enabled'] else 0)
  assert counts['fishr_steps']==(999 if c['multilevel']['fishr_levels'] else 0)
  assert counts['mixstyle_U_steps']==(3996 if c['multilevel']['mixstyle_u'] else 0)
  assert counts['U_packets']==(3996*128 if c['unlabeled'] else 0)
  assert d.learning_rate(c,1)==.0002 and abs(d.learning_rate(c,20)-1e-6)<1e-12
  rows.append(dict(row_id=row['row_id'],counts=counts))
 # Strict native parent and own checkpoint roundtrip for every distinct arm.
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm))
  with patch('experiments.cvs_e20_peer_soft.model.verify',return_value=proof),patch('torch.load',return_value=parent):model=build(c,'cpu')
  assert all(torch.equal(v,parent['model'][k]) for k,v in model.base.state_dict().items())
  assert not hasattr(model,'teacher');groups(model,c)
  with tempfile.TemporaryDirectory() as temp:
   teachers={str(s):dict(proof,seed=s,checkpoint=teacher_path(s)) for s in c['teacher']['seeds']}
   path=Path(temp)/'own.pt';payload=dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,epoch=20,total_epochs=40,optimizer_steps=4440,parent_checkpoint=c['parent_checkpoint'],parent_provenance=proof,teacher_provenance=teachers,checkpoint_sources=expected_sources(c),checkpoint_selection='final_only',target_access=False,run_id=c['run_id'],candidate_id=c['row_id'])
   torch.save(payload,path);restored=build(c,'cpu',path);model.eval();restored.eval()
   with torch.no_grad():assert torch.equal(model(torch.zeros(2,2,256)),restored(torch.zeros(2,2,256)))
   torch.save(dict(payload,parent_checkpoint='wrong'),path)
   try:build(c,'cpu',path)
   except ValueError:pass
   else:raise AssertionError('Invalid E20 source accepted')
   if teachers:
    torch.save(dict(payload,teacher_provenance={}),path)
    try:build(c,'cpu',path)
    except ValueError:pass
    else:raise AssertionError('Missing independent teacher ancestry accepted')
 for key,val in [('epochs',40),('target_access',True),('model_initialization','E200')]:
  c=d.config(d.rows()[0]);c[key]=val
  try:d.validate(c)
  except ValueError:pass
  else:raise AssertionError('Invalid registered config accepted')
 return dict(status='PASS',rows=rows,parent_weights_exact=True,own_checkpoint_roundtrip=True,wrong_parent_rejected=True,target_access=False)


def preflight():
 from .source import load_source
 from .objective import objective
 from experiments.cvs_e20_foundation.verify import verify
 from experiments.cvs_equivariant_identity.precision import numerical_context
 result=local();evidence=[]
 spec=d.read(d.ROOT/'registration/experiment.json')
 assert spec['run_id']==d.RUN and [r['config'] for r in spec['rows']]==[d.config(r) for r in d.rows()]
 with numerical_context(d.FULL_FP32_POLICY):
  for seed in d.SEEDS:
   proof=verify(seed,build_smoke=True);c=d.config(next(r for r in d.rows() if r['arm']=='PEER_JOINT' and r['model_seed']==seed))
   L,U,V,contract=load_source(c);model=build(c,'cpu');model.train();idx=torch.arange(32)
   from experiments.cvs_e20_routed_fishr import design as parent_design
   from experiments.cvs_e20_routed_fishr.model import build as parent_build
   probabilities=[]
   for teacher in c['teacher']['seeds']:
    pc=parent_design.config(next(r for r in parent_design.rows() if r['arm']=='CE' and r['model_seed']==teacher))
    peer=parent_build(pc,'cpu');peer.eval()
    with torch.no_grad():probabilities.append(peer(U['x'][idx]).softmax(1))
    del peer
   left,right=probabilities;q=(left+right)/2;logq=q.clamp_min(1e-12).log()
   bank=dict(q=q,ids=U['ids'][:32],teacher_seeds=c['teacher']['seeds'],peer_disagreement=left.argmax(1)!=right.argmax(1),peer_JS=.5*((left*(left.clamp_min(1e-12).log()-logq)).sum(1)+(right*(right.clamp_min(1e-12).log()-logq)).sum(1)))
   loss,_,weighted,detail=objective(c,model,bank,L,idx,U,idx,1332,'cpu')
   named=list(model.base.parameters())
   aux=torch.autograd.grad(weighted['aux_ce'],named,retain_graph=True,allow_unused=True)
   assert all(g is None or bool((g==0).all()) for g in aux)
   loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   assert torch.allclose(weighted['KL'],weighted['KL_same']+weighted['KL_disagree'],atol=1e-7)
   evidence.append(dict(seed=seed,L=len(L['ids']),U=len(U['ids']),V=len(V['ids']),U_role=U['role'],U_truth_hidden='y' not in U,V_backward=False,real_LU_backward=True,head_CE_identity_zero=True,E20_parent=proof,details=detail,real_peer_models=c['teacher']['seeds']))
 result.update(actual_source_evidence=evidence);return result


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-preflight',action='store_true');a=p.parse_args();torch.set_num_threads(2);d.write(a.output,preflight() if a.remote_preflight else local())
