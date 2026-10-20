"""Actual two-forward F flow, source boundary and default-head joint checks."""
import copy,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from .checks_math import run as math_checks
from experiments.cvs_f_supcon50.checks import W,S
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context

def fixture(m,step,device):
 from cvsrffi.tensors import unpack_batch
 torch.manual_seed(7061);x=torch.randn(128,2,256,device=device);y=torch.tensor(([1]*16+[3]*16+[0]*8+[2]*8+[4]*8+[5]*8)*2,device=device);domain=torch.tensor([0]*64+[3]*64,device=device);rx=torch.tensor((1,1,1,3,3,3,4,4,4,6,6,6,8,8,8),device=device)[domain];prior=step-1
 m._feature_l_steps=prior;m._legacy_calls=prior;m._legacy_extra_packets=prior*128;m._supcon_calls=prior
 if hasattr(m,'_gap_calls'):m._gap_calls=prior
 meta=dict(rx_i=rx.cpu(),day_i=(domain%3+1).cpu(),physical_id=tuple('synthetic'+str(i) for i in range(128)),split_source=['ssdg_labeled_tx_visible']*128,tx_label_visible=torch.ones(128,dtype=torch.bool))
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);_,_,extra=unpack_batch((x,y,domain.cpu(),meta));correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra=extra,domains=domain,optimizer_step=step,identity_loss=ce,labeled_count=128)
 return ce+correction,metrics

def run(profile,metadata,output,device='cpu',real_checkpoint_smoke=True):
 torch.set_num_threads(2);math=math_checks()
 from . import design as d,model as api,mechanism as me
 from .baseline import original_config
 from experiments.cvs_f_supcon50 import model as reference
 base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata);neg=math['protocol_negative_cases']
 for key,value in [('epochs',20),('test_views',['practical_high']),('weak_indices',[1]),('gap_margin',.2),('gap_min_packets_per_class',1),('gap_grouping','RX_only'),('separate_head_training',True),('gap_pair_average_before_partition',False)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:neg+=1
  else:raise AssertionError('Altered fixed source/defaulthead/gap recipe accepted')
 refgeo=d.reference_geometry(d.SEEDS[0]);baseV=dict(recall=[.98]*6,accuracy=.98,worst_RX=.97);good=dict(baseV,geometry=copy.deepcopy(refgeo),gap_scheduled=9990,gap_eligible_fraction=1.,gap_active_fraction=1.,gap_cell_coverage=1.)
 for i in (1,3):good['geometry']['per_TX'][i]['out_RX_margin']['q10']+=.003;good['geometry']['per_TX'][i]['RX_spread_to_inter_TX_gap']*=.99
 assert d.expansion({a:good for a in d.ARMS},baseV)['expand_arms']==list(d.ARMS)
 for field,value in [('one_weak',None),('gap_eligible_fraction',.84),('gap_active_fraction',.09),('gap_cell_coverage',.99)]:
  bad=copy.deepcopy(good)
  if field=='one_weak':bad['geometry']['per_TX'][1]['out_RX_margin']['q10']-=.003
  else:bad[field]=value
  assert not d.expansion({a:bad for a in d.ARMS},baseV)['expand_arms']
 flows=[];grads=[]
 with numerical_context(d.FULL_FP32_POLICY):
  ref=reference.build(original_config(c['model_seed']),device).train();m=api.build(c,device).train();a,_=fixture(ref,1,device);rng=torch.get_rng_state().clone();b,_=fixture(m,1,device);assert torch.equal(a.detach(),b.detach()) and torch.equal(rng,torch.get_rng_state());assert not m._gap_records[-1]['scheduled'];del a,b,m,ref
  ref=reference.build(original_config(c['model_seed']),device).train();ref_total,_=fixture(ref,1111,device);ref_head=[(n,p) for n,p in ref.named_parameters() if n.startswith('id_backbone.encoder.core.id_backbone.cls_head.') and p.requires_grad];ref_head_grad=dict(zip((n for n,p in ref_head),[g.detach().clone() for g in torch.autograd.grad(ref_total,[p for n,p in ref_head])]))
  del ref_total,ref
  for row in d.rows(d.SEEDS[:1]):
   c=d.config(row,str(base),profile,metadata);m=api.build(c,device).train();captures=[];forward=[];raw_features=[];original_penalty=me.penalty
   def observe(*args,**kw):
    value=original_penalty(*args,**kw);raw=original_penalty(raw_features[0],raw_features[1],*args[2:],**kw);assert torch.equal(args[0].detach(),raw_features[0].detach()) and torch.equal(args[1].detach(),raw_features[1].detach()) and torch.equal(value[0].detach(),raw[0].detach());captures.append((value,raw));return value
   handle=m.get_submodule('id_backbone.encoder.core.id_backbone.time_down').register_forward_pre_hook(lambda *args:forward.append(1));raw_handle=m.id_backbone.register_forward_hook(lambda mod,args,out:raw_features.append(out['feat_joint']));me.penalty=observe
   try:total,metrics=fixture(m,1111,device)
   finally:me.penalty=original_penalty;handle.remove();raw_handle.remove()
   assert len(forward)==2 and len(captures)==1 and m._gap_scheduled==1 and m._gap_eligible==1 and m._gap_active==1 and len(m._supcon_records[-1]['details'])==4 and m._supcon_pending is None and not m._supcon_handles, dict(mode=row['treatment'],forwards=len(forward),captures=len(captures),scheduled=m._gap_scheduled,eligible=m._gap_eligible,active=m._gap_active,original_details=len(m._supcon_records[-1]['details']),record=m._gap_records[-1])
   named=dict(m.named_parameters());mid=next(n for n in named if '.t2.' in n and named[n].requires_grad);aux=captures[0][0][0];rawaux=captures[0][1][0];head=[(n,p) for n,p in named.items() if n.startswith('id_backbone.encoder.core.id_backbone.cls_head.') and p.requires_grad];head_aux=torch.autograd.grad(aux,[p for n,p in head],allow_unused=True,retain_graph=True);head_main=torch.autograd.grad(total,[p for n,p in head],retain_graph=True);assert len(head)==6 and all(g is None for g in head_aux) and all(torch.equal(g.detach(),ref_head_grad[n]) for (n,p),g in zip(head,head_main));gw,gs,gsh,gmid=torch.autograd.grad(aux,[named[W],named[S],named['id_backbone.encoder.core.id_backbone.t1.pw.weight'],named[mid]],allow_unused=True,retain_graph=True);mainw=torch.autograd.grad(total,named[W],retain_graph=True)[0]
   raw_feature_grad=torch.autograd.grad(rawaux,[named[S],named['id_backbone.encoder.core.id_backbone.t1.pw.weight'],named[mid]],retain_graph=True);vjp_max=[float((a.detach()-b.detach()).abs().max()) for a,b in zip((gs,gsh,gmid),raw_feature_grad)];assert all(torch.allclose(a.detach(),b.detach(),atol=1e-7,rtol=2e-5) for a,b in zip((gs,gsh,gmid),raw_feature_grad)),vjp_max
   assert gw is None and all(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in (gs,gsh,gmid,mainw));grads.append(gsh.detach().cpu().clone());total.backward()
   from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
   opt=build(m,m.parameters(),'CENTER_UNITS',lr=.0002,weight_decay=.0001,eps=1e-8);wb=named[W].detach().clone();sb=named[S].detach().clone();clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(wb,named[W]) and not torch.equal(sb,named[S]) and opt._sinc_unit_steps==1 and opt._sinc_physical_clip_calls==1
   teacher=copy.deepcopy(m).eval();calls=teacher._gap_calls
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256,device=device))
   assert teacher._gap_calls==calls
   m.eval();x=torch.randn(2,2,256,device=device)
   with torch.no_grad(),fresh_parameter_caches(m):actual=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
   assert torch.equal(actual,exported) and sum(p.numel() for p in m.parameters())==900722
   flows.append(dict(treatment=row['treatment'],actual_identity_forwards=2,extra_model_forwards=0,aux_all_6_default_head_gradients_None=True,main_all_default_head_gradients_exact_original=True,aux_feature_values_exact=True,aux_backbone_VJPs_match_raw_joint=True,aux_backbone_VJP_max_abs_errors=vjp_max,aux_default_W_gradient=None,aux_Sinc_gradient_norm=float(gs.norm()),aux_shallow_gradient_norm=float(gsh.norm()),aux_middle_gradient_norm=float(gmid.norm()),main_W_gradient_norm=float(mainw.norm()),discarded_native_optimizer_steps=1,default_W_updated_jointly=True,physical_Sinc_updated=True,export_exact=True,record=m._gap_records[-1]))
 assert (grads[0]-grads[1]).norm()>1e-7
 real_smoke=False
 if device=='cpu' and real_checkpoint_smoke:
  from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
  root=Path(profile).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source';rc=d.read(root/'local_source_complete.json')['config'];source_proof(rc);cp=torch.load(root/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(cp,rc);sm=api.build(c,'cpu').eval();sm.load_state_dict(cp['model'],strict=True)
  with torch.no_grad(),fresh_parameter_caches(sm):out=sm.forward_identity_only(torch.zeros(2,2,256))
  assert out['tx_logits'].shape==(2,6) and out['id_feat_joint'].shape==(2,160) and torch.isfinite(out['tx_logits']).all();real_smoke=True
 return dict(status='PASS',device=device,math=math,protocol_negative_cases=neg,source_expansion_positive_negative_PASS=True,actual_F_flows=flows,real_checkpoint_no_query_smoke=real_smoke,loaded_checkpoint_discarded_not_inherited=True,parameters=900722,extra_parameters=0,IQ_reads=0,target_reads=0,head_fit_updates=0,real_data_training_updates=0,discarded_synthetic_native_optimizer_steps=2)
if __name__=='__main__':print(json.dumps(run(*sys.argv[1:]),allow_nan=False))
