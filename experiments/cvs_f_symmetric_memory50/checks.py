"""Real F current-anchor/history flow, whole-head exclusion and native physical optimizer."""
import copy,json,sys
from pathlib import Path
import torch
from .checks_math import run as math_checks
from torch.nn import functional as F
from experiments.cvs_f_supcon50.checks import W,S
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context
def fixture(m,step,device,token=0):
 from cvsrffi.tensors import unpack_batch
 torch.manual_seed(7061+token);x=torch.randn(128,2,256,device=device);y=torch.tensor(([1]*16+[3]*16+[0]*8+[2]*8+[4]*8+[5]*8)*2,device=device);domain=torch.tensor([0]*64+[3]*64,device=device);rx=torch.tensor((1,1,1,3,3,3,4,4,4,6,6,6,8,8,8),device=device)[domain];prior=step-1
 m._feature_l_steps=prior;m._legacy_calls=prior;m._legacy_extra_packets=prior*128;m._supcon_calls=prior
 if hasattr(m,'_gap_calls'):m._gap_calls=prior
 meta=dict(rx_i=rx.cpu(),day_i=(domain%3+1).cpu(),physical_id=tuple('synthetic'+str(token)+'_'+str(i) for i in range(128)),split_source=['ssdg_labeled_tx_visible']*128,tx_label_visible=torch.ones(128,dtype=torch.bool))
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  assert torch.equal(m._legacy_actual_L_input,x)
  original_forward=m.forward_identity_only;second_inputs=[]
  def second_forward(xx,*args,**kw):second_inputs.append(xx.detach().clone());return original_forward(xx,*args,**kw)
  m.forward_identity_only=second_forward
  try:
   ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);_,_,extra=unpack_batch((x,y,domain.cpu(),meta));correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra=extra,domains=domain,optimizer_step=step,identity_loss=ce,labeled_count=128)
  finally:m.forward_identity_only=original_forward
  assert len(second_inputs)==1 and torch.equal(second_inputs[0],x) and m._legacy_actual_L_input is None
 return ce+correction,metrics

def run(profile,metadata,output,device='cpu',real_checkpoint_smoke=True):
 torch.set_num_threads(2);math=math_checks()
 from . import design as d,model as api,mechanism as me
 from .baseline import original_config
 from experiments.cvs_f_supcon50 import model as reference
 base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata);neg=math['protocol_negative_cases']
 for key,value in [('epochs',20),('test_views',['practical_high']),('weak_indices',[1]),('contrastive_temperature',.1),('contrastive_layers',['joint160']),('current_keys_detached',True),('separate_head_training',True),('contrastive_class_reweight',True)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:neg+=1
  else:raise AssertionError('Changed fixed protocol accepted')
 baseV=dict(recall=[.98]*6,accuracy=.98,worst_RX=.97);good=dict(recall=[.98,.981,.98,.981,.98,.98],accuracy=.9803,worst_RX=.9701,geometry=dict(per_TX=[dict(zero_feature_count=0)]*6))
 assert d.expansion({a:good for a in d.ARMS},baseV)['expand_arms']==list(d.ARMS)
 bad=copy.deepcopy(good);bad['recall'][3]=.98;assert not d.expansion({a:bad for a in d.ARMS},baseV)['expand_arms']
 flows=[];grads=[]
 with numerical_context(d.FULL_FP32_POLICY):
  ref=reference.build(original_config(c['model_seed']),device).train();m=api.build(c,device).train();a,_=fixture(ref,1,device);rng=torch.get_rng_state().clone();b,_=fixture(m,1,device);assert torch.equal(a.detach(),b.detach()) and torch.equal(rng,torch.get_rng_state());del a,b,m,ref
  ref=reference.build(original_config(c['model_seed']),device).train();prime,_=fixture(ref,1111,device);del prime;ref_total,_=fixture(ref,1112,device,1);headprefix='id_backbone.encoder.core.id_backbone.cls_head.';ref_head=[(n,p) for n,p in ref.named_parameters() if n.startswith(headprefix) and p.requires_grad];ref_head_grad=dict(zip((n for n,p in ref_head),[g.detach().clone() for g in torch.autograd.grad(ref_total,[p for n,p in ref_head])]))
  del ref_total,ref
  for row in d.rows(d.SEEDS[:1]):
   c=d.config(row,str(base),profile,metadata);m=api.build(c,device).train();prime,_=fixture(m,1111,device);del prime;captured=[];inputs=[];original=me.penalty
   def observe(*args,**kw):
    value=original(*args,**kw);captured.append(value[0]);return value
   h=m.get_submodule('id_backbone.encoder.core.id_backbone.time_down').register_forward_pre_hook(lambda mod,args:inputs.append(args[0].detach().clone()));me.penalty=observe
   try:total,metrics=fixture(m,1112,device,1)
   finally:me.penalty=original;h.remove()
   assert len(inputs)==2 and len(captured)==2 and len(m._supcon_records[-1]['details'])==4 and m._supcon_pending is None and not m._supcon_handles,dict(forwards=len(inputs),shapes=[list(v.shape) for v in inputs],max_IQ_difference=float((inputs[0]-inputs[1]).abs().max()) if len(inputs)==2 else None,penalties=len(captured),details=len(m._supcon_records[-1]['details']),pending=m._supcon_pending is not None,handles=len(m._supcon_handles))
   assert all(v['historical_keys']==(128 if row['treatment']=='SYM4' else 0) for v in m._supcon_records[-1]['details'])
   named=dict(m.named_parameters());head=[(n,p) for n,p in named.items() if n.startswith(headprefix) and p.requires_grad];aux=torch.stack(captured).mean();hg=torch.autograd.grad(aux,[p for n,p in head],allow_unused=True,retain_graph=True);main=torch.autograd.grad(total,[p for n,p in head],retain_graph=True);assert len(head)==6 and all(g is None for g in hg) and all(torch.equal(g.detach(),ref_head_grad[n]) for (n,p),g in zip(head,main))
   shallow='id_backbone.encoder.core.id_backbone.t1.pw.weight';middle=next(n for n in named if '.t2.' in n and named[n].requires_grad);gs,gsh,gm=torch.autograd.grad(aux,[named[S],named[shallow],named[middle]],retain_graph=True);assert all(torch.isfinite(g).all() and g.norm()>0 for g in (gs,gsh,gm));grads.append(gsh.detach().cpu().clone());total.backward()
   from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
   opt=build(m,m.parameters(),'CENTER_UNITS',lr=.0002,weight_decay=.0001,eps=1e-8);wb=named[W].detach().clone();sb=named[S].detach().clone();clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(wb,named[W]) and not torch.equal(sb,named[S]) and opt._sinc_unit_steps==1 and opt._sinc_physical_clip_calls==1
   teacher=copy.deepcopy(m).eval();calls=teacher._supcon_calls
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256,device=device))
   assert teacher._supcon_calls==calls and all(b.updates==(2 if row['treatment']=='SYM4' else 0) for phase in teacher._crossbatch_banks.values() for b in phase.values())
   m.eval();x=torch.randn(2,2,256,device=device)
   with torch.no_grad(),fresh_parameter_caches(m):actual=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
   assert torch.equal(actual,exported) and sum(p.numel() for p in m.parameters())==900722
   flows.append(dict(treatment=row['treatment'],actual_identity_forwards=2,same_actual_IQ=True,aux_all_6_default_head_gradients_None=True,main_all_6_default_head_gradients_exact_original=True,aux_Sinc_gradient_norm=float(gs.norm()),aux_shallow_gradient_norm=float(gsh.norm()),aux_middle_gradient_norm=float(gm.norm()),physical_Sinc_updated=True,default_head_jointly_updated=True,own_EMA_safe=True,export_exact=True,discarded_synthetic_native_optimizer_steps=1))
 assert len(grads)==1
 smoke=False
 if device=='cpu' and real_checkpoint_smoke:
  from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
  root=Path(profile).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source';rc=d.read(root/'local_source_complete.json')['config'];source_proof(rc);cp=torch.load(root/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(cp,rc);sm=api.build(c,'cpu').eval();sm.load_state_dict(cp['model'],strict=True)
  with torch.no_grad(),fresh_parameter_caches(sm):out=sm.forward_identity_only(torch.zeros(2,2,256))
  assert out['tx_logits'].shape==(2,6) and torch.isfinite(out['tx_logits']).all();smoke=True
 return dict(status='PASS',device=device,math=math,protocol_negative_cases=neg,source_expansion_positive_negative=True,actual_F_flows=flows,real_checkpoint_no_query_smoke=smoke,loaded_checkpoint_discarded_not_inherited=True,parameters=900722,extra_parameters=0,IQ_reads=0,target_reads=0,head_fit_updates=0,real_data_training_updates=0,discarded_synthetic_native_optimizer_steps=1)
if __name__=='__main__':print(json.dumps(run(*sys.argv[1:]),allow_nan=False))
