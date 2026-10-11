"""Focused source-only correctness checks; no target inputs or scores."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import json
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d


def check():
 from .objectives import cross_rx_contrastive,manifold_mix,head_logits,cross_rx_feature_mix
 from .optim import TemporarySAM,WeightAverage
 from . import dispatch as q,evaluate as e
 torch.set_num_threads(2)
 z=F.normalize(torch.randn(128,160),dim=1).requires_grad_();y=torch.arange(128)%6;rx=torch.arange(128)//6%5;ids=torch.arange(128)
 head=SimpleNamespace(weight=torch.randn(6,160,requires_grad=True),scale=30.)
 model=SimpleNamespace(id_backbone=SimpleNamespace(encoder=SimpleNamespace(core=SimpleNamespace(id_backbone=SimpleNamespace(cls_head=head)))))
 loss,stats=cross_rx_contrastive(z,y,rx,ids);loss.backward();assert stats['active_anchors']>0 and torch.isfinite(z.grad).all() and z.grad.norm()>0
 assert torch.allclose(manifold_mix(model,z,y,torch.arange(127,-1,-1),1.),F.cross_entropy(head_logits(model,z),y))
 empty,_=cross_rx_contrastive(z[:2],torch.zeros(2,dtype=torch.long),torch.zeros(2,dtype=torch.long),ids[:2]);assert empty==0
 duplicate,_=cross_rx_contrastive(z[:2],torch.zeros(2,dtype=torch.long),torch.arange(2),torch.zeros(2,dtype=torch.long));assert duplicate==0
 yy=y.clone();rr=rx.clone();yy[96:112]=0;rr[96:104]=0;rr[104:112]=1
 relation=dict(relations=[dict(valid=True,positions=[list(range(8)),list(range(8,16))])])
 mixed,detail=cross_rx_feature_mix(model,z,yy,rr,ids,relation,torch.Generator().manual_seed(8));assert torch.isfinite(mixed) and detail['pairs']==8
 try:cross_rx_feature_mix(model,z,yy,torch.zeros_like(rr),ids,relation,torch.Generator().manual_seed(8))
 except ValueError:pass
 else:raise AssertionError('Same-RX mixing accepted')
 m=torch.nn.Linear(4,2);opt=torch.optim.AdamW(m.parameters());sam=TemporarySAM(opt.param_groups,.02)
 x=torch.randn(3,4);before={k:v.clone() for k,v in m.state_dict().items()};m(x).square().mean().backward();sam.perturb()
 opt.zero_grad(set_to_none=True);m(x).square().mean().backward();sam.restore()
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and len(opt.state)==0
 opt.step();avg=WeightAverage(m,[n for n,p in m.named_parameters()]);expected=[]
 for epoch in (32,34,36,38,40):
  with torch.no_grad():m.weight.fill_(epoch)
  expected.append(m.weight.detach().clone());avg.add(m,epoch)
 avg.apply(m,[32,34,36,38,40]);torch.testing.assert_close(m.weight,torch.stack(expected).mean(0))
 assert len(d.rows())==33 and len({r['row_id'] for r in d.rows()})==33
 for row in d.rows():
  c=d.validate(d.config(row));assert c['updates']==8880 and not any(c[k] for k in ('teacher','unlabeled','satellite','target_access'))
  try:d.validate(dict(c,epochs=39))
  except ValueError:pass
  else:raise AssertionError('Changed budget accepted')
 assert q.choose_gpu({0:dict(pids={1,2,3,4},free_mb=99999)}) is None
 assert q.choose_gpu({0:dict(pids={1,2,3},free_mb=99999)}) is None
 assert q.choose_gpu({0:dict(pids={1,2},free_mb=99999)})==0
 with patch.object(e,'snapshot',return_value=[d.config(d.rows()[0])]),patch.object(e,'prediction_preflight',side_effect=ValueError('incomplete')),patch.object(d,'read',side_effect=AssertionError('Truth must stay closed')):
  try:e.score()
  except ValueError:pass
  else:raise AssertionError('Incomplete predictions scored')
 return dict(status='PASS',target_access=False,checks=['contrastive_gradient_and_duplicate_exclusion','soft_mix_endpoint','cross_RX_same_TX_mix','SAM_exact_restore_one_update','fixed_five_snapshot_mean','33_config_matrix','capacity_three','truth_last'])


def remote_preflight():
 from . import actions as a
 from .source import forward
 from .optim import TemporarySAM
 from experiments.cvs_stage2_geometry import actions as old
 from experiments.cvs_stage2_geometry.receiver import load_receiver
 from experiments.cvs_stage2_decouple.common import build_model,verify_parent,parameter_groups,IdentityView
 from experiments.cvs_stage2_decouple.source_counterfactual_audit import ReadOnlyState,RNGState
 from experiments.cvs_phase1_stack.recover import verify_contract
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from .dispatch import preparation_ready
 torch.set_num_threads(2);started=time.perf_counter();records=[]
 output=d.RELEASE_BASE/'actions';output.mkdir(exist_ok=False)
 with torch.random.fork_rng(devices=[]),numerical_context(d.FULL_FP32_POLICY):
  for seed in d.SEEDS:
   if not preparation_ready(seed):raise ValueError('Existing source preparation incomplete')
   pc=d.parent_config(seed);provenance=verify_parent(pc);folder=d.AUDIT_BASE/f's{seed}'
   packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False);verify_contract(packed['contract'],d.read(d.SOURCE))
   cache=packed['labeled']
   for key,role,count in [('labeled','L_s',6300),('validation','V',27000)]:
    ids=packed[key]['ids']
    if len(ids)!=count or len(set(ids))!=count or sorted(ids)!=sorted(packed['contract']['role_ids'][role]):raise ValueError('Cached source roles differ')
   if set(cache['ids'])&set(packed['validation']['ids']):raise ValueError('Overlapping source roles')
   model=build_model(pc,'cpu')
   if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):raise ValueError('Native BatchNorm invalid for registered SAM/average')
   fit_start=time.perf_counter();payload=a.fit(IdentityView(model),cache,seed,'cpu');payload['fit_seconds']=time.perf_counter()-fit_start
   payload['parent_provenance']=provenance;torch.save(payload,output/f's{seed}.pt')
   d.write(output/f's{seed}.json',dict(seed=seed,parent=provenance,contract=payload['contract'],identity_state=payload['identity_state'],
    auxiliary_updates=payload['auxiliary_updates'],source_occurrences=payload['source_occurrences'],fit_seconds=payload['fit_seconds'],fit_logs=payload['fit_logs'],target_access=False))
   bank=load_receiver(torch.load(folder/'receiver.pt',map_location='cpu',weights_only=False))
   if bank.payload['seed']!=seed or bank.payload['source_ids']!=cache['ids']:raise ValueError('Relation provenance differs')
   relation=bank.sample_relations(torch.Generator().manual_seed(seed+723));indices=torch.cat((torch.arange(96),relation['indices']))
   old_payload=torch.load(folder/'actions.pt',map_location='cpu',weights_only=False)
   if old_payload['seed']!=seed or old_payload['source_ids']!=cache['ids']:raise ValueError('Old coverage payload provenance differs')
   coverage=old.load_actions(old_payload,'cpu');balanced=a.FrozenBalanced(payload,'cpu')
   with torch.no_grad():
    frozen_parent=IdentityView(model).eval()
    payload_diagnostics=old._source_diagnostics(frozen_parent,cache,balanced.networks,seed,'cpu',128)
   d.write(output/f's{seed}_diagnostics.json',payload_diagnostics)
   checks=[]
   for arm in d.ARMS:
    c=d.config(dict(arm=arm,model_seed=seed,row_id=f'{arm}-s{seed}'));groups,_=parameter_groups(model);model.train()
    with torch.random.fork_rng(devices=[]):
     calls=[];hook=model.dom_backbone.register_forward_hook(lambda *args:calls.append(1))
     try:
      before={n:p.detach().clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
      rng=RNGState();loss,weighted,parts,details=forward(c,model,cache,indices,relation,888,'cpu',balanced if arm=='H10' else coverage)
      loss.backward();sam=None
      if c['enabled']['sam']:
       sam=TemporarySAM(groups,c['sam']['rho']);after=RNGState();sam.perturb();model.zero_grad(set_to_none=True);rng.restore()
       try:
        second,*_=forward(c,model,cache,indices,relation,888,'cpu',coverage);second.backward()
       finally:sam.restore();after.restore()
      norm=sum(p.grad.detach().square().sum() for group in groups for p in group['params'] if p.grad is not None).sqrt()
      if calls or not torch.isfinite(loss) or not torch.isfinite(norm) or not norm>0:raise ValueError('Native hybrid forward/backward invalid')
      if any(not torch.equal(before[n],p) for n,p in model.named_parameters()):raise ValueError('Preflight changed identity')
      checks.append(dict(arm=arm,loss=float(loss.detach()),gradient_norm=float(norm),domain_calls=0,optimizer_steps=0,sam_second_backward=c['enabled']['sam']))
     finally:hook.remove()
   records.append(dict(seed=seed,parent=provenance,auxiliary_updates=payload['auxiliary_updates'],source_occurrences=payload['source_occurrences'],native_checks=checks))
   del model,packed,cache,payload,bank,balanced,coverage
 done=dict(status='FROZEN_BALANCED_ACTIONS',seeds=list(d.SEEDS),target_access=False,identity_updates=0,auxiliary_updates=600,source_occurrences=75600,contract=a.contract())
 d.write(output/'complete.json',done)
 return dict(status='PASS',source_only=True,target_access=False,identity_updates=0,formal_parent_models=3,records=records,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
 if args.output and args.output.exists():raise FileExistsError('Preserve preflight')
 result=check()
 if args.remote_preflight:result=dict(status='PASS',synthetic=result,remote=remote_preflight())
 if args.output:d.write(args.output,result)
 print(json.dumps(result,indent=2))
