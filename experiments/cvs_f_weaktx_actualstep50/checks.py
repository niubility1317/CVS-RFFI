"""Actual F native-step oracle; synthetic histories force a visible risk conflict."""
import copy,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from .checks_math import run as math_checks
from experiments.cvs_f_supcon50.checks import W,S
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_rdrop_sinc_units.optimizer import build as native_optimizer,clip_physical

def fixture(m,step,device='cpu'):
 from cvsrffi.tensors import unpack_batch
 torch.manual_seed(7061);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;domain=torch.arange(128,device=device)%15;rx=torch.tensor((1,1,1,3,3,3,4,4,4,6,6,6,8,8,8),device=device)[domain];prior=step-1
 m._feature_l_steps=prior;m._legacy_calls=prior;m._legacy_extra_packets=prior*128;m._supcon_calls=prior
 if hasattr(m,'_step_calls'):m._step_calls=prior;m._step_post_calls=prior
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);_,_,extra=unpack_batch((x,y,domain.cpu(),{'rx_i':rx.cpu()}))
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra=extra,domains=domain,optimizer_step=step,identity_loss=ce,labeled_count=128)
 return ce+correction,metrics

def optimizer(m,step):
 opt=native_optimizer(m,m.parameters(),'CENTER_UNITS',lr=.0002,weight_decay=.0001,eps=1e-8);opt._sinc_unit_steps=step-1;opt._sinc_physical_clip_calls=step-1;return opt

def history(opt,parameters,g,step):
 offset=0
 for name,p in parameters:
  end=offset+p.numel();a=g[offset:end].reshape_as(p).to(device=p.device,dtype=p.dtype);offset=end
  opt.state[p].update(step=torch.tensor(float(step-1)),exp_avg=-1000*a,exp_avg_sq=a.square()+.0001)

def run(profile,metadata,output,device='cpu'):
 torch.set_num_threads(2);math=math_checks()
 from . import design as d,model as api,mechanism as me
 from experiments.cvs_f_supcon50 import model as reference
 from .baseline import original_config
 base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata);negative=math['protocol_negative_cases'];flows=[];step=3112
 for key,value in [('epochs',20),('test_views',['practical_high']),('weak_indices',[1]),('step_every_steps',2),('separate_head_training',True),('step_default_W_excluded',False),('step_risks',['weak1_CE']),('step_norm_match_tolerance',.1)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Altered recipe/head/risk config accepted')
 refgeo=d.reference_geometry(d.SEEDS[0]);baseV=dict(recall=[.98]*6,accuracy=.98,worst_RX=.97);good=dict(baseV,geometry=copy.deepcopy(refgeo),step_scheduled=2498,step_eligible_fraction=1.,step_nonzero_fraction=1.,step_fallbacks=0,step_maximum_norm_match_error=0.)
 for index in (1,3):good['geometry']['per_TX'][index]['out_RX_margin']['q10']+=.003;good['geometry']['per_TX'][index]['RX_spread_to_inter_TX_gap']*=.99
 assert d.expansion({a:good for a in d.ARMS},baseV)['expand_arms']==list(d.ARMS)
 for field,value in [('one_weak',None),('step_eligible_fraction',.98),('step_nonzero_fraction',.09),('step_fallbacks',1),('step_maximum_norm_match_error',.002)]:
  bad=copy.deepcopy(good)
  if field=='one_weak':bad['geometry']['per_TX'][1]['out_RX_margin']['q10']-=.003
  else:bad[field]=value
  assert not d.expansion({a:bad for a in d.ARMS},baseV)['expand_arms']
 with numerical_context(d.FULL_FP32_POLICY):
  for off in (1,1111):
   ref=reference.build(original_config(c['model_seed']),device).train();m=api.build(c,device).train();a,_=fixture(ref,off,device);rng=torch.get_rng_state();b,_=fixture(m,off,device);assert torch.equal(a.detach(),b.detach()) and torch.equal(rng,torch.get_rng_state());assert not m._step_records[-1]['scheduled'];del a,b,ref,m
  baseline=reference.build(original_config(c['model_seed']),device).train();base_loss,_=fixture(baseline,step,device);rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [];base_loss.backward();base_parameters=me.select(baseline);base_grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in baseline.named_parameters()};native_before=None
  for row in d.rows(d.SEEDS[:1]):
   c=d.config(row,str(base),profile,metadata);m=api.build(c,device).train();opt=optimizer(m,step);me.bind_optimizer(m,opt);captured=[];forward_calls=[];original_route=me.route
   def observed(*args,**kwargs):
    result=original_route(*args,**kwargs);captured.append((args,result));return result
   handle=m.get_submodule('id_backbone.encoder.core.id_backbone.time_down').register_forward_pre_hook(lambda *args:forward_calls.append(1));loss,metrics=fixture(m,step,device);handle.remove();assert len(forward_calls)==2 and torch.equal(base_loss.detach(),loss.detach()) and torch.equal(rng,torch.get_rng_state())
   if cuda_rng:assert all(torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all()))
   pending=m._step_pending;g=pending['gradients'][0];history(opt,pending['parameters'],g,step);loss.backward()
   for name,p in m.named_parameters():
    old=base_grads[name]
    if old is None:assert p.grad is None
    else:torch.testing.assert_close(p.grad,old,rtol=0,atol=2e-7 if device.startswith('cuda') and not name.startswith('id_backbone.encoder.core.id_backbone.cls_head.') else 0)
   raw_reference=reference.build(original_config(c['model_seed']),device).train();raw_reference.load_state_dict(m.state_dict(),strict=True);raw_named=dict(raw_reference.named_parameters())
   for name,p in m.named_parameters():raw_named[name].grad=None if p.grad is None else p.grad.detach().clone()
   raw_features=me.select(raw_reference);bo=optimizer(raw_reference,step);history(bo,raw_features,g,step);clip_physical(bo,raw_reference.parameters(),5.);bo.step();native_before=torch.cat([p.detach().reshape(-1) for _,p in raw_features]).cpu();base_W=raw_named[W].detach().clone();base_state=bo.state_dict()
   clip_physical(opt,m.parameters(),5.);witness=[];opt.register_step_post_hook(lambda *args:witness.append(dict(m.named_parameters())[S].detach().clone()));me.route=observed
   step_rng=torch.get_rng_state().clone();step_cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []
   try:opt.step()
   finally:me.route=original_route
   assert torch.equal(step_rng,torch.get_rng_state())
   if step_cuda_rng:assert all(torch.equal(x,y) for x,y in zip(step_cuda_rng,torch.cuda.get_rng_state_all()))
   assert len(captured)==1 and len(witness)==1 and torch.equal(witness[0],dict(m.named_parameters())[S]) and m._step_pending is None and m._step_post_calls==step
   args,(pc,ct,info)=captured[0];torch.testing.assert_close(args[1],native_before,rtol=0,atol=0);chosen=pc if row['treatment']=='UPDATE_CONE' else ct;actual=torch.cat([p.detach().reshape(-1) for _,p in me.select(m)]).cpu();torch.testing.assert_close(actual,chosen,rtol=0,atol=0)
   assert info['raw_normalized_dots'][0]>0 and info['actual_correction_nonzero'] and info['stored_norm_match_error']<=.001 and not info['solver_fallback'] and not info['storage_match_fallback'] and not info['projection_storage_fallback']
   scale=torch.zeros_like(args[0],dtype=torch.float64)
   for lo,hi in args[3]:scale[lo:hi]=(args[1][lo:hi].double()-args[0][lo:hi].double()).square().mean().sqrt()
   hh=args[2]*scale;hh=hh/hh.norm(dim=1).clamp_min(1e-300)[:,None];stored_x=(chosen.double()-args[0].double())/scale.clamp_min(1e-300);expected_risk=hh@stored_x
   torch.testing.assert_close(torch.tensor(m._step_records[-1]['applied_normalized_dots'],dtype=torch.float64),expected_risk,rtol=1e-12,atol=1e-8)
   assert m._step_records[-1]['constraint_enforced']==(row['treatment']=='UPDATE_CONE')
   if row['treatment']=='STEP_NORM':assert expected_risk[0]>0
   assert info['full_strength_max_residual_relative']<=.001 and torch.equal(dict(m.named_parameters())[W],base_W) and opt._sinc_physical_clip_calls==step and opt._sinc_unit_steps==step
   for name,p in m.named_parameters():
    if name.startswith('id_backbone.encoder.core.id_backbone.cls_head.'):assert torch.equal(p,raw_named[name])
   for key,sv in base_state['state'].items():
    for n,value in sv.items():torch.testing.assert_close(opt.state_dict()['state'][key][n],value,rtol=0,atol=0)
   teacher=copy.deepcopy(m).eval()
   with torch.no_grad():
    for q,p in zip(teacher.parameters(),m.parameters()):q.copy_(p)
   calls=teacher._step_calls
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256,device=device))
   assert teacher._step_calls==calls and teacher._step_pending is None
   m.eval();x=torch.randn(2,2,256,device=device)
   with torch.no_grad(),fresh_parameter_caches(m):actual_feature=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
   assert torch.equal(actual_feature,exported) and sum(p.numel() for p in m.parameters())==900722
   flows.append(dict(treatment=row['treatment'],actual_identity_forwards=2,original_scalar_RNG_exact=True,CPU_all_preclip_gradients_exact=not device.startswith('cuda'),CUDA_nonhead_backward_absolute_tolerance=2e-7 if device.startswith('cuda') else 0,poststep_RNG_exact=True,unmodified_native_step_oracle=True,all_native_AdamW_moments_exact=True,default_W_joint_update_exact=True,poststep_Sinc_witness_sees_final_state=True,feature_export_exact=True,selected_feature_parameters=231233,record=m._step_records[-1]))
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 ref=Path(profile).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source';rc=d.read(ref/'local_source_complete.json')['config'];source_proof(rc);checkpoint=torch.load(ref/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,rc);smoke_model=api.build(c,'cpu').eval();smoke_model.load_state_dict(checkpoint['model'],strict=True)
 with torch.no_grad(),fresh_parameter_caches(smoke_model):smoke=smoke_model.forward_identity_only(torch.zeros(2,2,256))
 assert smoke['tx_logits'].shape==(2,6) and torch.isfinite(smoke['tx_logits']).all()
 return dict(status='PASS',device=device,math=math,protocol_negative_cases=negative,actual_F_flows=flows,real_checkpoint_no_query_smoke=True,loaded_checkpoint_discarded_not_inherited_by_training=True,parameters=900722,IQ_reads=0,target_reads=0,head_fit_updates=0,discarded_synthetic_native_optimizer_steps=4,real_data_training_updates=0,EMA_eval_untouched=True)
if __name__=='__main__':print(json.dumps(run(*sys.argv[1:]),allow_nan=False))
