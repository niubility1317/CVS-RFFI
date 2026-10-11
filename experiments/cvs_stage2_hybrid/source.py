"""Fixed hybrid matrix: legal L_s only, one final E40/average per row."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from . import design as d
from .objectives import cross_rx_contrastive,cross_rx_feature_mix,manifold_mix
from .optim import TemporarySAM,WeightAverage
from .actions import FrozenBalanced
from experiments.cvs_stage2_geometry.features import identity
from experiments.cvs_stage2_geometry import actions as old_actions
from experiments.cvs_stage2_geometry.objectives import keep_loss,lt_loss
from experiments.cvs_stage2_geometry.receiver import load_receiver
from experiments.cvs_stage2_geometry.source import gradient_diagnostics,SourceIdentityView
from experiments.cvs_stage2_decouple.common import build_model,verify_parent,IdentityView,parameter_groups,clean,margins
from experiments.cvs_stage2_decouple.source import SourceStream,seed_all
from experiments.cvs_stage2_decouple.deterministic_forward import deterministic_identity
from experiments.cvs_stage2_decouple.source_counterfactual_audit import ReadOnlyState,RNGState
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags


def forward(c,model,cache,indices,relation,step,device,actions=None):
 x=cache['x'][indices].to(device);y=cache['y'][indices].to(device);rx=cache['rx'][indices].to(device);ids=indices.to(device)
 logits,z=identity(model,x,verify=step==1)
 zero=logits.sum()*0.;parts=dict(ce=F.cross_entropy(logits,y),contrastive=zero,cross_rx_mix=zero,manifold_mix=zero,finite_lt=zero,keep=zero)
 details={};g=torch.Generator().manual_seed(c['model_seed']+step*47)
 if c['enabled']['contrastive']:
  parts['contrastive'],details['contrastive']=cross_rx_contrastive(z,y,rx,ids,c['losses']['temperature'])
 if c['enabled']['cross_rx_mix'] and relation is not None:
  parts['cross_rx_mix'],details['cross_rx_mix']=cross_rx_feature_mix(model,z,y,rx,ids,relation,g)
 if c['enabled']['manifold_mix'] and step%4==0:
  perm=torch.randperm(len(y),generator=g).to(device)
  lam=float(np.random.RandomState(c['model_seed']+step*59).beta(c['mixing']['beta'],c['mixing']['beta']))
  parts['manifold_mix']=manifold_mix(model,z,y,perm,lam);details['manifold_mix']=dict(lambda_value=lam,pairs=128)
 if c['enabled']['actions'] and step%4==0:
  kind=c['action']['order'][(step//4-1)%4]
  policy='conditional' if c['enabled']['conditional_actions'] else 'coverage'
  ax,stats=actions.transform(x[-32:],kind,c['model_seed']+step*13,policy=policy)
  with deterministic_identity(model):
   cz,_=identity(model,x[-32:]);az,_=identity(model,ax)
  pm=cache['parent_margin'][indices[-32:]].to(device)
  parts['finite_lt']=lt_loss(cz,az,y[-32:],c['losses']['margin_required'])
  parts['keep']=keep_loss(cz,pm,y[-32:],c['losses']['margin_required'])
  details['actions']=dict(kind=kind,stats=stats,augmented_accuracy=float((az.argmax(1)==y[-32:]).float().mean()),
   deterministic_clean_margin=float(margins(cz,y[-32:]).detach().mean()),augmented_margin=float(margins(az,y[-32:]).detach().mean()))
 ramp=min(step/444.,1.)
 weighted={name:value if name=='ce' else ramp*c['losses'][name]*value for name,value in parts.items()}
 total=sum(weighted.values())
 details.update(main_accuracy=float((logits.argmax(1)==y).float().mean()),clean_margin=float(margins(logits,y).detach().mean()),ramp=ramp)
 return total,weighted,parts,details


def save_completion(c,out,result):
 d.write(Path(out)/'completion.json',clean(result))
 d.write(Path(out).parent/'source_frozen.json',dict(status='SOURCE_FROZEN',run_id=c['run_id'],row_id=c['row_id'],config=c,
  checkpoint=str(Path(out)/'final_ssdg.pth'),epoch=40,optimizer_steps=8880,target_access=False,target_scores_consumed=False,selection=c['checkpoint_selection']))


def checkpoint(c,model,optimizer,out,filename='final_ssdg.pth',averaging=None):
 path=out/filename
 torch.save(dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,run_id=c['run_id'],candidate_id=c['row_id'],
  epoch=40,optimizer_steps=8880,checkpoint_selection=c['checkpoint_selection'],parent_checkpoint=c['parent_checkpoint'],
  optimizer=optimizer.state_dict(),averaging=averaging,target_access=False,rng=torch.get_rng_state()),path)
 return path


def train(c,device='cuda:0'):
 from experiments.cvs_stage2_decouple.data import cached_loader
 from experiments.cvs_feature_disentangle.source import source_validation
 from experiments.cvs_feature_disentangle.validation import evaluate_source_stress
 from experiments.cvs_phase1_stack.recover import verify_contract
 d.validate(c);pc=d.parent_config(c['model_seed']);provenance=verify_parent(pc);prepared=Path(c['preparation'])
 pdone=d.read(prepared/'preparation_complete.json')
 if pdone['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or pdone['parent_contract']!=pc or pdone['target_access'] or pdone['identity_updates']!=0:raise ValueError('Preparation provenance differs')
 packed=torch.load(prepared/'source_cache.pt',map_location='cpu',weights_only=False)
 verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(pdone['source_contract'],packed['contract']);cache=packed['labeled']
 for role,key,n in [('L_s','labeled',6300),('V','validation',27000)]:
  ids=packed[key]['ids']
  if len(ids)!=n or len(set(ids))!=n or sorted(ids)!=sorted(packed['contract']['role_ids'][role]):raise ValueError('Source roles differ')
 if set(cache['ids'])&set(packed['validation']['ids']):raise ValueError('Overlapping source roles')
 bank=load_receiver(torch.load(prepared/'receiver.pt',map_location='cpu',weights_only=False))
 if bank.payload['seed']!=c['model_seed'] or bank.payload['source_ids']!=cache['ids']:raise ValueError('Relation sampler provenance differs')
 stream=SourceStream(len(cache['ids']),bank,c['model_seed']);vloader=cached_loader(packed['validation'],256)
 out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);seed_all(c['model_seed']);torch.set_num_threads(2);started=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY):
  model=build_model(pc,device)
  if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):raise ValueError('SAM/averaging protocol requires no BatchNorm; do not silently double-update buffers')
  groups,names=parameter_groups(model);optimizer=torch.optim.AdamW(groups,weight_decay=c['optimizer']['weight_decay'])
  sam=TemporarySAM(groups,c['sam']['rho']) if c['enabled']['sam'] else None
  average=WeightAverage(model,[name for values in names.values() for name in values]) if c['enabled']['averaging'] else None
  actions=None
  if c['enabled']['actions']:
   if c['enabled']['conditional_actions']:
    payload=torch.load(c['action']['payload'],map_location='cpu',weights_only=False)
    if payload['seed']!=c['model_seed'] or payload['source_ids']!=cache['ids']:raise ValueError('Balanced policy source differs')
    actions=FrozenBalanced(payload,device)
   else:
    payload=torch.load(prepared/'actions.pt',map_location='cpu',weights_only=False)
    if payload['seed']!=c['model_seed'] or payload['source_ids']!=cache['ids']:raise ValueError('Coverage source differs')
    actions=old_actions.load_actions(payload,device)
  resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),
   hardware=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
   backend_flags=actual_flags(),parameter_groups=names,trainable_parameters=sum(p.numel() for g in groups for p in g['params']),
   total_parameters=sum(p.numel() for p in model.parameters()),execution=dict(teacher=False,unlabeled_iterator=False,satellite_generator=False,
    receiver_loss=False,receiver_network_forward=False,relation_sampling=True,contrastive=c['enabled']['contrastive'],
    action_forward=c['enabled']['actions'],auxiliary_optimizer_during_identity=False,sam=c['enabled']['sam'],weight_average=c['enabled']['averaging']))
  for name,val in [('resolved_config.json',resolved),('initialization.json',provenance),('source_contract.json',packed['contract'])]:d.write(out/name,clean(val))
  print('RESOLVED '+json.dumps(clean(resolved)),flush=True)
  counts=dict(identity_updates=0,main_clean_packets=0,sam_extra_clean_packets=0,action_steps=0,contrastive_steps=0,cross_rx_mix_steps=0,
   manifold_mix_steps=0,relation_batches=0,teacher_calls=0,unlabeled_packets=0,satellite_calls=0,auxiliary_optimizer_steps=0)
  epochs=[];cuda=str(device).startswith('cuda')
  if cuda:torch.cuda.reset_peak_memory_stats(device)
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8') as log:
   for epoch in range(1,41):
    model.train();epoch_start=time.perf_counter();sums={};last_audit=None
    for local in range(1,223):
     step=(epoch-1)*222+local;indices,relation=stream.step(step)
     for group in groups:group['lr']=d.learning_rate(c,step,group['peak'])
     optimizer.zero_grad(set_to_none=True);rng=RNGState() if sam else None;t=time.perf_counter()
     loss,weighted,parts,details=forward(c,model,cache,indices,relation,step,device,actions)
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite hybrid loss')
     audit=gradient_diagnostics(weighted,parts,groups) if step%888==0 else {}
     if audit:last_audit=dict(step=step,**audit)
     loss.backward();sam_loss=None;sam_initial_norm=None
     if sam:
      after=RNGState();sam_initial_norm=sam.perturb()
      try:
       optimizer.zero_grad(set_to_none=True);rng.restore()
       extra,_,_,_=forward(c,model,cache,indices,relation,step,device,actions)
       if not torch.isfinite(extra):raise FloatingPointError('Nonfinite SAM neighborhood loss')
       extra.backward();sam_loss=float(extra.detach())
      finally:sam.restore();after.restore()
      counts['sam_extra_clean_packets']+=128
     norm=torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],float('inf'),error_if_nonfinite=True)
     optimizer.step()
     if cuda:torch.cuda.synchronize(device)
     counts['identity_updates']+=1;counts['main_clean_packets']+=128
     counts['action_steps']+=int('actions' in details);counts['contrastive_steps']+=int('contrastive' in details)
     counts['cross_rx_mix_steps']+=int('cross_rx_mix' in details);counts['manifold_mix_steps']+=int('manifold_mix' in details)
     counts['relation_batches']+=int(relation is not None)
     scalars={k:float(v.detach()) for k,v in parts.items()};scalars.update({'weighted_'+k:float(v.detach()) for k,v in weighted.items()})
     scalars.update(total_loss=float(loss.detach()),gradient_norm=float(norm),lr_encoder=groups[0]['lr'],lr_fusion_head=groups[1]['lr'],
      step_seconds=time.perf_counter()-t,main_accuracy=details['main_accuracy'],clean_margin=details['clean_margin'])
     for k,v in scalars.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(clean(dict(step=step,epoch=epoch,metrics=scalars,details=details,gradient_audit=audit,
      sam_neighborhood_loss=sam_loss,sam_initial_gradient_norm=sam_initial_norm)),allow_nan=False)+'\n')
    log.flush()
    if average and epoch in c['averaging']['snapshot_epochs']:average.add(model,epoch)
    with ReadOnlyState(model) as state:
     model.eval()
     with torch.no_grad():validation=source_validation(IdentityView(model),vloader,device)
    row=dict(epoch=epoch,optimizer_steps=counts['identity_updates'],epoch_time_s=time.perf_counter()-epoch_start,
     **{k:v/222 for k,v in sums.items()},**validation,**counts,gradient_audit=last_audit,
     average_snapshots=average.count if average else 0)
    epochs.append(clean(row));(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in epochs),encoding='utf-8')
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in epochs]
    with (out/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print('EPOCH '+json.dumps(clean(row)),flush=True)
  if counts['identity_updates']!=8880 or counts['relation_batches']!=1110:raise ValueError('Incomplete fixed identity budget')
  if c['enabled']['actions'] and counts['action_steps']!=2220:raise ValueError('Incomplete L/T schedule')
  training_seconds=time.perf_counter()-started;averaging=None
  if average:
   checkpoint(c,model,optimizer,out,'final_unaveraged.pth',dict(selected=False,reason='fixed average preregistered'))
   average.apply(model,c['averaging']['snapshot_epochs']);averaging=dict(count=average.count,epochs=average.epochs,parameters=list(sorted(average.names)))
   with ReadOnlyState(model):
    model.eval()
    with torch.no_grad():validation=source_validation(IdentityView(model),vloader,device)
  final=checkpoint(c,model,optimizer,out,averaging=averaging)
  model.load_state_dict(torch.load(final,map_location=device,weights_only=False)['model'],strict=True)
  evaluate_source_stress(SourceIdentityView(model),vloader,device,c,out)
  result=dict(status='SOURCE_TRAINED',epoch=40,optimizer_steps=8880,logged_steps=8880,config=c,checkpoint=str(final),
   checkpoint_selection=c['checkpoint_selection'],target_access=False,target_evaluated=False,final_source_metrics=validation,
   mechanism_execution=counts,parent_provenance=provenance,averaging=averaging,elapsed_seconds=time.perf_counter()-started,
   resources=dict(identity_training_and_epoch_validation_seconds=training_seconds,final_source_diagnostics_seconds=time.perf_counter()-started-training_seconds,
    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,trainable_parameters=resolved['trainable_parameters'],total_parameters=resolved['total_parameters'],
    main_identity_packets=counts['main_clean_packets'],extra_sam_packets=counts['sam_extra_clean_packets'],parent_training_epochs=200,incremental_epochs=40))
  save_completion(c,out,result);print('SOURCE_TRAINED '+json.dumps(clean(result)),flush=True)
 return result


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
 train(d.read(args.config),args.device)
