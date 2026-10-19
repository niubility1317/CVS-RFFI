"""Focused actual native F graph, role-loader and old-gate checks."""
import copy,json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as me
from .data import build_context,NativeCachedRole
from .baseline import original_config
from experiments.cvs_f_supcon50.checks import fixture,W,S
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_phase1_stack.runtime import installed

def run(profile_path,metadata,output,device='cpu'):
 torch.set_num_threads(2);profile=d.read(profile_path);base=Path(output)/d.RUN;records=[];negative=0
 c=d.config(d.rows()[0],str(base),profile_path,metadata)
 for key,value in [('epochs',20),('label_epochs',20),('test_views',['practical_high']),('joint160_max_weight',.07),('class_weights',[1.5]*6),('separate_head_training',True)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Wrong original50 recipe accepted: '+key)
 a=d.make_args(c,device)
 with installed(c) as native:
  ctx=build_context(native,a,torch.device(device),profile,metadata)
  L=next(iter(ctx['train_loader']));U=next(iter(ctx['unlabeled_loader']));V=next(iter(ctx['val_loader']))
  if L[0].shape!=(128,2,256) or U[0].shape!=(256,2,256) or V[0].shape!=(256,2,256) or (U[1]!=-1).any() or U[3]['tx_label_visible'].any() or any(k in U[3] for k in ('tx','tx_i','y_tx','true_tx_i','tx_label')):raise ValueError('Cached native loaders expose wrong source roles')
  ds=ctx['unlabeled_loader'].dataset;order=np.argsort(ds.gate[:,2]);pair=None
  for i,j in zip(order[:-1],order[1:]):
   if int(ds.rx[i])==int(ds.rx[j]) and int(ds.day[i])==int(ds.day[j]) and int(ds.gate[i,0])==int(ds.gate[j,0]) and abs(int(ds.gate[i,1])-int(ds.gate[j,1]))<=2 and abs(int(ds.gate[i,2])-int(ds.gate[j,2]))<=2:pair=(int(i),int(j));break
  if pair is None:raise ValueError('No source U native neighbors available for gate check')
  mm={key:[ds[i][3][key] for i in pair] for key in ('rx_i','day_i','eq_i','sig_i','base_index')}
  if native.temporal_neighbor_agreement_mask([0,0],[.99,.99],mm,window=2,min_conf=.8)!=[True,True] or native.temporal_neighbor_agreement_mask([0,0],[.99,.99],{'rx_i':mm['rx_i']},window=2,min_conf=.8)!=[False,False]:raise ValueError('Original batch_neighbor gate metadata is not preserved')
 with numerical_context(d.FULL_FP32_POLICY):
  control=api.build(c,device).train();initial={k:v.detach().clone() for k,v in control.state_dict().items()};base_loss,_=fixture(control,1,device)
  for arm in d.ARMS[1:]:
   cc=d.config(next(r for r in d.rows() if r['treatment']==arm and r['model_seed']==d.SEEDS[0]),str(base),profile_path,metadata);model=api.build(cc,device).train()
   if any(not torch.equal(v,model.state_dict()[n]) for n,v in initial.items()):raise ValueError('Feature mechanism changes F initialization')
   warm,_=fixture(model,1,device)
   if not torch.equal(warm.detach(),base_loss.detach()):raise ValueError('Warmup alters the original RDrop objective')
   captured=[];old_penalty=me.penalty;old_loss=me.loss
   def capture_penalty(*args,**kwargs):
    out=old_penalty(*args,**kwargs);captured.append(.1*out[0]);return out
   def capture_joint(*args,**kwargs):
    out=old_loss(*args,**kwargs);captured.append(.05*out[0]);return out
   me.penalty=capture_penalty;me.loss=capture_joint
   try:total,metrics=fixture(model,15,device)
   finally:me.penalty=old_penalty;me.loss=old_loss
   if len(captured)!=(4 if cc['joint160_geometry'] else 2):raise ValueError('Actual feature objectives or forward reuse differ')
   aux=torch.stack(captured).sum()/2;named=dict(model.named_parameters());gW,gS=torch.autograd.grad(aux,[named[W],named[S]],allow_unused=True,retain_graph=True);main=torch.autograd.grad(total,named[W],retain_graph=True)[0]
   if gW is not None or gS is None or not torch.isfinite(gS).all() or gS.norm()<=0 or not torch.isfinite(main).all() or main.norm()<=0:raise ValueError('Feature auxiliary/head joint gradient boundary differs')
   if model._supcon_handles or model._supcon_pending is not None or model._supcon_phase is not None or model._legacy_extra_packets!=((15-1)*222+1)*128:raise ValueError('Original L/RDrop hook lifetime changed')
   teacher=copy.deepcopy(model).eval();calls=teacher._supcon_calls
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256,device=device))
   if teacher._supcon_calls!=calls or teacher._supcon_pending is not None or teacher._supcon_handles:raise ValueError('EMA/eval activates auxiliary objective')
   model.eval();x=torch.randn(2,2,256,device=device)
   with torch.no_grad(),fresh_parameter_caches(model):z=model.forward_identity_only(x)['id_feat_joint'];exported=api.export(model)(x)
   if not torch.equal(z,exported) or sum(p.numel() for p in model.parameters())!=900722:raise ValueError('Actual f_theta export/parameter count differs')
   records.append(dict(treatment=arm,auxiliary_head_gradient=None,auxiliary_Sinc_gradient=float(gS.norm()),main_head_gradient=float(main.norm()),temporary_hooks_remaining=0,extra_model_forwards=0,feature_export_exact=True))
  # Actual own checkpoint, no data/query fitting; not a trained performance claim.
  out=Path(output);out.mkdir(parents=True,exist_ok=True);checkpoint=out/'own_initial_F_checkpoint.pt';torch.save(dict(model=initial,config=c,scratch_only=True,ancestors=[]),checkpoint);control.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model'],strict=True);control.eval()
  with torch.no_grad(),fresh_parameter_caches(control):z=control.forward_identity_only(torch.zeros(2,2,256,device=device))
  if z['tx_logits'].shape!=(2,6) or not torch.isfinite(z['tx_logits']).all():raise ValueError('Own actual checkpoint no-query smoke failed')
 result=dict(status='PASS',device=device,protocol_negative_cases=negative,source_loader_shapes=dict(L=[128,2,256],U=[256,2,256],V=[256,2,256]),U_TX_hidden=True,original_temporal_neighbor_gate_preserved=True,source_handles='READY EXACT_MATCH reused',warmup_matches_original=True,actual_feature_flows=records,network_parameters=900722,default_head=[6,160],own_real_checkpoint_no_query_smoke=True,optimizer_updates=0,target_reads=0,truth_reads=0,source_IQ='Only source loader/gate checks; no parameter updates or source performance claim')
 (Path(output)/'checks.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result),flush=True)
 return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();run(a.profile,a.metadata,a.output,a.device)
