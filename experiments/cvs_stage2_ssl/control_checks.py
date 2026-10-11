"""Bounded graph/protocol integration; source-only remote preflight."""
import argparse
import time
import torch
from . import design as d
from .source import forward
from experiments.cvs_stage2_hybrid.semisupervised import EMAIdentity,UnlabeledStream,VisitPassBank,make_plan,validate_unlabeled


def graphs(model,cache,bank,actions,device,seed=None,uc=None):
 seed=d.SEEDS[0] if seed is None else seed
 from experiments.cvs_stage2_decouple.common import parameter_groups
 from experiments.cvs_stage2_decouple.source_counterfactual_audit import RNGState
 from experiments.cvs_stage2_hybrid.optim import TemporarySAM
 model.train();teacher=EMAIdentity(model)
 if uc is None:uc=dict(x=cache['x'].clone(),ids=['U-'+str(i) for i in range(len(cache['x']))],role='U_s',preprocessing={},
  rx=cache['rx'],day=cache['day'],domain=cache['rx'],eq=torch.ones(len(cache['x']),dtype=torch.long),sig=torch.arange(len(cache['x'])),base_index=torch.arange(len(cache['x'])))
 validate_unlabeled(uc);stream=UnlabeledStream(len(uc['ids']),d.SEEDS[0]);records=[];calls=[]
 hooks=[m.dom_backbone.register_forward_hook(lambda *args:calls.append(1)) for m in (model,teacher.model)]
 try:
  for arm in d.ARMS:
   c=d.config(dict(arm=arm,model_seed=seed,row_id=f'{arm}-s{seed}'));groups,names=parameter_groups(model)
   for g in groups:g['peak']=c['optimizer']['encoder_peak_lr' if g['name']=='encoder' else 'fusion_head_peak_lr']
   assert d.learning_rate(c,444,groups[0]['peak'])==c['optimizer']['encoder_peak_lr']
   relation=bank.sample_relations(torch.Generator().manual_seed(127));indices=torch.cat((torch.arange(96),relation['indices']))
   ui,up=stream.batch(256);history=VisitPassBank() if arm!='S0' else None
   plan=make_plan(teacher,uc,ui,up,888,d.SEEDS[0],c['semi']['mode'],history);plan['rx']=uc['rx'][ui].to(device);plan['ids']=(ui+len(cache['x'])).to(device)
   # Force the acceptance branch with student's own synthetic prediction,
   # never truth. Production gates remain entirely unchanged.
   from experiments.cvs_stage2_geometry.features import identity
   with torch.no_grad():plan['pseudo']=identity(model,plan['x_strong'])[0].argmax(1);plan['base_mask']=torch.ones(256,dtype=torch.bool,device=device)
   before={n:p.detach().clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True);rng=RNGState()
   loss,w,parts,details=forward(c,model,cache,indices,relation,888,device,actions,plan)
   assert torch.isfinite(loss) and 'pseudo' in parts and 'entropy' in parts;loss.backward()
   history_before=dict(history.entries) if history else None;teacher_calls=teacher.calls
   if c['enabled']['sam']:
    sam=TemporarySAM(groups,.02);after=RNGState();sam.perturb();model.zero_grad(set_to_none=True);rng.restore()
    try:second,*_=forward(c,model,cache,indices,relation,888,device,actions,plan);second.backward()
    finally:sam.restore();after.restore()
   assert teacher.calls==teacher_calls and teacher.updates==0 and (history is None or history.entries==history_before)
   assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for g in groups for p in g['params'])
   assert all(p.grad is None for p in teacher.model.parameters())
   records.append(dict(arm=arm,loss=float(loss.detach()),accepted=details['semi']['accepted'],encoder_peak=groups[0]['peak'],fusion_peak=groups[1]['peak']))
  assert not calls
 finally:
  for hook in hooks:hook.remove()
 return records


def check():
 from experiments.cvs_stage2_hybrid.semisupervised_checks import check as core_check
 core=core_check()
 from experiments.cvs_stage2_decouple.design import make_args
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_stage2_geometry.receiver_checks import synthetic_cache
 from experiments.cvs_stage2_geometry.receiver import fit_receiver,load_receiver
 from experiments.cvs_stage2_hybrid import actions as a
 pc=d.parent_config(d.SEEDS[0]);args=make_args(pc,'cpu');args.num_domains,args.input_len=5,256
 with installed(pc) as native:model=native.build_baseline_model(args,torch.device('cpu'))
 cache=synthetic_cache();cache['x']=torch.randn(360,2,256);bank=load_receiver(fit_receiver(cache,d.SEEDS[0]))
 payload=dict(version=a.VERSION,contract=a.contract(),seed=d.SEEDS[0],networks={k:a.new_net(k,a.a._generator(d.SEEDS[0]+i),'cpu').state_dict() for i,k in enumerate(a.a.KINDS)})
 records=graphs(model,cache,bank,a.FrozenBalanced(payload,'cpu'),'cpu')
 assert len(d.rows())==48
 for row in d.rows():d.validate(d.config(row))
 from . import evaluate as e
 from unittest.mock import patch
 with patch.object(e,'snapshot',return_value=[d.config(d.rows()[0])]),patch.object(e,'prediction_preflight',side_effect=ValueError('Incomplete')),patch.object(d,'read',side_effect=AssertionError('truth must remain closed')):
  try:e.score()
  except ValueError:pass
  else:raise AssertionError('Incomplete predictions scored')
 return dict(status='PASS',target_access=False,core=core,graphs=records,SAM_teacher_history_once=True,teacher_no_grad=True,domain_forwards=0,identity_updates=0,truth_last=True)


def remote_preflight():
 from .data import build
 from .dispatch import require_preparations
 from experiments.cvs_stage2_decouple.common import build_model,verify_parent
 from experiments.cvs_stage2_geometry.receiver import load_receiver
 from experiments.cvs_stage2_hybrid.actions import FrozenBalanced
 from experiments.cvs_stage2_hybrid import design as h
 started=time.perf_counter();torch.set_num_threads(2);u=build();require_preparations();records=[]
 for seed in d.SEEDS:
  pc=d.parent_config(seed);provenance=verify_parent(pc);model=build_model(pc,'cpu');folder=d.AUDIT_BASE/f's{seed}'
  packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False);payload=torch.load(h.RELEASE_BASE/'actions'/f's{seed}.pt',map_location='cpu',weights_only=False)
  if payload['seed']!=seed or payload['source_ids']!=packed['labeled']['ids'] or payload['parent_provenance']!=provenance:raise ValueError('Reused action provenance differs')
  uc=torch.load(d.UNLABELED_CACHE,map_location='cpu',weights_only=False);validate_unlabeled(uc,packed['contract']['role_ids']['U_s'])
  bank=load_receiver(torch.load(folder/'receiver.pt',map_location='cpu',weights_only=False))
  tests=graphs(model,packed['labeled'],bank,FrozenBalanced(payload,'cpu'),'cpu',seed,uc)
  records.append(dict(seed=seed,parent=provenance,checks=tests))
 return dict(status='PASS',source_unlabeled=u,parents=records,identity_updates=0,target_access=False,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);args=p.parse_args()
 torch.set_num_threads(2);result=remote_preflight() if args.remote_preflight else check();d.write(args.output,result);print('PASS SSL graph, EMA, U traversal, gates, stronger LR, truth-last')
