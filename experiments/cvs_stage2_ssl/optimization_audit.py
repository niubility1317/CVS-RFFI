"""Source-only Hz-coordinate audit; virtual optimizer copies never train identity."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_stage2_decouple.common import build_model,verify_parent,parameter_groups
from experiments.cvs_stage2_decouple.deterministic_forward import deterministic_identity
from experiments.cvs_stage2_geometry.features import identity

RUN='20261011-phase1-source-optimizer-units-manysig-m3-r01'
BASE=Path(d.PROJECT)/'runs'/RUN
LEVELS=(3e-5,1e-4,3e-4)


def virtual_update(value,gradient,lr,scale=1.):
 """One fresh AdamW update in u=f/scale, mapped back to raw Hz."""
 parameter=torch.nn.Parameter(value.detach().clone()/scale)
 before=parameter.detach().clone();mapped_before=before*scale
 parameter.grad=gradient.detach().clone()*scale
 optimizer=torch.optim.AdamW([parameter],lr=lr,weight_decay=1e-4,eps=1e-8,betas=(.9,.999))
 optimizer.step()
 return dict(delta=parameter.detach()*scale-mapped_before,roundtrip=mapped_before-value.detach(),coordinate_changed=parameter.detach()!=before)


def run(output,registration):
 torch.set_num_threads(2);output=Path(output);output.mkdir(exist_ok=False);spec=d.read(registration)
 if spec['run_id']!=RUN or len(spec['rows'])!=3:raise ValueError('Unregistered audit')
 source_contract=d.read(d.SOURCE);records=[];started=time.perf_counter()
 for seed in d.SEEDS:
  pc=d.parent_config(seed);provenance=verify_parent(pc);model=build_model(pc,'cpu');groups,_=parameter_groups(model)
  packed=torch.load(d.AUDIT_BASE/f's{seed}/source_cache.pt',map_location='cpu',weights_only=False);cache=packed['labeled']
  if len(cache['ids'])!=6300 or sorted(cache['ids'])!=sorted(source_contract['role_ids']['L_s']):raise ValueError('Source L roles differ')
  frequency={}
  for module_name,module in model.named_modules():
   if not module_name.startswith('id_backbone.'):continue
   if hasattr(module,'low_hz_') and hasattr(module,'band_hz_') and hasattr(module,'sample_rate'):
    for key in ('low_hz_','band_hz_'):
     name=module_name+'.'+key;parameter=getattr(module,key)
     if not parameter.requires_grad:raise ValueError('Expected trainable frequency parameter')
     frequency[name]=(parameter,float(module.sample_rate))
  if not frequency:raise ValueError('No actual Hz parameter found')
  baseline={name:p.detach().clone() for name,p in model.named_parameters()};calls=[]
  hook=model.dom_backbone.register_forward_hook(lambda *args:calls.append(1))
  metrics={name:dict(name=name,shape=list(p.shape),sample_rate=fs,norm=float(p.norm()),
   min=float(p.min()),max=float(p.max()),min_float32_spacing_Hz=float((torch.nextafter(p.detach(),torch.full_like(p,float('inf')))-p.detach()).min()),
   max_float32_spacing_Hz=float((torch.nextafter(p.detach(),torch.full_like(p,float('inf')))-p.detach()).max()),batches=[]) for name,(p,fs) in frequency.items()}
  losses=[]
  try:
   with deterministic_identity(model):
    for start in range(0,6300,128):
     x=cache['x'][start:start+128];y=cache['y'][start:start+128];logits,_=identity(model,x);loss=F.cross_entropy(logits,y)
     gradients=torch.autograd.grad(loss,[p for p,fs in frequency.values()]);losses.append(dict(count=len(x),loss=float(loss.detach())))
     for (name,(parameter,fs)),gradient in zip(frequency.items(),gradients):
      active=gradient!=0
      tests=[]
      for lr in LEVELS:
       for scale in (1.,fs):
        virtual=virtual_update(parameter,gradient,lr,scale);delta=virtual['delta']
        if not torch.isfinite(delta).all():raise ValueError('Nonfinite virtual source update')
        tests.append(dict(lr=lr,coordinate='raw_Hz' if scale==1 else 'frequency_over_fs',numel=parameter.numel(),
         zero_delta=int((delta==0).sum()),active_grad_numel=int(active.sum()),active_grad_zero_delta=int(((delta==0)&active).sum()),
         coordinate_changed=int(virtual['coordinate_changed'].sum()),active_grad_zero_coordinate_update=int((active&~virtual['coordinate_changed']).sum()),
         initialization_roundtrip_mean_abs_Hz=float(virtual['roundtrip'].abs().mean()),initialization_roundtrip_max_abs_Hz=float(virtual['roundtrip'].abs().max()),
         mean_abs_delta_Hz=float(delta.abs().mean()),max_abs_delta_Hz=float(delta.abs().max()),
         mean_abs_delta_over_fs=float(delta.abs().mean()/fs),gradient_mean_abs=float(gradient.abs().mean()),first_order_CE_change=float((gradient*delta).sum())))
      metrics[name]['batches'].append(dict(batch_start=start,source_count=len(x),tests=tests))
   if calls or any(not torch.equal(baseline[name],p) for name,p in model.named_parameters()):raise ValueError('Audit changed identity or invoked domain')
  finally:hook.remove()
  summaries=[]
  for name,metric in metrics.items():
   summary={k:v for k,v in metric.items() if k!='batches'};summary['comparisons']=[]
   for index in range(6):
    values=[batch['tests'][index] for batch in metric['batches']];numel=sum(v['numel'] for v in values);active=sum(v['active_grad_numel'] for v in values)
    summary['comparisons'].append(dict(lr=values[0]['lr'],coordinate=values[0]['coordinate'],parameter_batch_occurrences=numel,
     zero_update_fraction=sum(v['zero_delta'] for v in values)/numel,
     nonzero_gradient_but_zero_update_fraction=sum(v['active_grad_zero_delta'] for v in values)/active if active else None,
     mean_abs_delta_Hz=sum(v['mean_abs_delta_Hz']*v['numel'] for v in values)/numel,
     max_abs_delta_Hz=max(v['max_abs_delta_Hz'] for v in values),mean_first_order_CE_change=sum(v['first_order_CE_change'] for v in values)/len(values)))
    summary['comparisons'][-1].update(coordinate_change_fraction=sum(v['coordinate_changed'] for v in values)/numel,
     initialization_roundtrip_mean_abs_Hz=sum(v['initialization_roundtrip_mean_abs_Hz']*v['numel'] for v in values)/numel,
     initialization_roundtrip_max_abs_Hz=max(v['initialization_roundtrip_max_abs_Hz'] for v in values))
   summaries.append(summary)
  result=dict(seed=seed,parent_provenance=provenance,source_count=6300,source_batches=len(losses),
   source_CE=sum(v['loss']*v['count'] for v in losses)/6300,frequency_parameters=summaries,
   all_identity_parameters_l2_norm=float(sum(p.detach().square().sum() for g in groups for p in g['params']).sqrt()),
   non_frequency_identity_l2_norm=float(sum(p.detach().square().sum() for n,p in model.named_parameters() if p.requires_grad and n not in frequency).sqrt()),
   identity_state_exactly_unchanged=True,identity_updates=0,optimizer_state_attached_to_live_model=False,
   U_truth_read=False,U_samples_read=0,V_forward_or_fit=False,target_access=False,domain_forwards=0)
  d.write(output/f's{seed}_summary.json',result);d.write(output/f's{seed}_batch_details.json',dict(source_losses=losses,frequency=list(metrics.values())))
  records.append(result)
 result=dict(status='SOURCE_OPTIMIZER_UNITS_AUDIT_COMPLETE',run_id=RUN,rows=records,identity_updates=0,target_access=False,
  virtual_updates_only=True,budget='3 matched E200; full6300 L deterministic CE per seed; no trajectory/selection',elapsed_seconds=time.perf_counter()-started,
  caveat='Fresh first-step virtual AdamW per L batch, not the running SSL optimizer trajectory; no target/generalization claim')
 d.write(output/'completion.json',result)
 return result


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--registration',required=True);a=p.parse_args()
 result=run(a.output,a.registration);print(json.dumps(dict(status=result['status'],rows=len(result['rows']),identity_updates=0,target_access=False,elapsed_seconds=result['elapsed_seconds'])))
