"""Actual original F graph: block gradient oracle/defaulthead and no-query checkpoint."""
import copy,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from .checks_math import run as math_checks
from experiments.cvs_f_supcon50.checks import W,S
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context

def fixture(m,step,device='cpu'):
 from cvsrffi.tensors import unpack_batch
 torch.manual_seed(7061);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;domain=torch.arange(128,device=device)%15;rx=torch.tensor((1,1,1,3,3,3,4,4,4,6,6,6,8,8,8),device=device)[domain];prior=step-1
 m._feature_l_steps=prior;m._legacy_calls=prior;m._legacy_extra_packets=prior*128;m._supcon_calls=prior
 if hasattr(m,'_pc_calls'):m._pc_calls=prior
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);_,_,extra=unpack_batch((x,y,domain.cpu(),{'rx_i':rx.cpu()}))
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra=extra,domains=domain,optimizer_step=step,identity_loss=ce,labeled_count=128)
 return ce+correction,metrics

def run(profile,metadata,output):
 torch.set_num_threads(2);math=math_checks()
 from . import design as d,model as api,mechanism as me
 from experiments.cvs_f_supcon50 import model as reference
 from .baseline import original_config
 base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata);negative=math['protocol_negative_cases'];flows=[];directions=[]
 for key,value in [('epochs',20),('test_views',['practical_high']),('weak_indices',[1]),('pc_every_steps',2),('separate_head_training',True),('pc_default_W_excluded',False),('pc_max_strength',.5)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Altered source/defaulthead/gradient recipe accepted')
 refgeo=d.reference_geometry(d.SEEDS[0]);baseV=dict(recall=[.98]*6,accuracy=.98,worst_RX=.97);good=dict(baseV,geometry=copy.deepcopy(refgeo),pc_scheduled=2498,pc_eligible_fraction=1.,pc_nonzero_fraction=1.)
 for index in (1,3):good['geometry']['per_TX'][index]['out_RX_margin']['q10']+=.003;good['geometry']['per_TX'][index]['RX_spread_to_inter_TX_gap']*=.99
 assert d.expansion({a:good for a in d.ARMS},baseV)['expand_arms']==list(d.ARMS)
 for field,value in [('one_weak',None),('pc_eligible_fraction',.98),('pc_nonzero_fraction',.09)]:
  bad=copy.deepcopy(good)
  if field=='one_weak':bad['geometry']['per_TX'][1]['out_RX_margin']['q10']-=.003
  else:bad[field]=value
  assert not d.expansion({a:bad for a in d.ARMS},baseV)['expand_arms']
 with numerical_context(d.FULL_FP32_POLICY):
  original=reference.build(original_config(c['model_seed']),'cpu').train();m=api.build(c,'cpu').train()
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.state_dict().items())
  for step in (1,1111):
   a,_=fixture(original,step);ra=torch.get_rng_state();b,_=fixture(m,step);assert torch.equal(a.detach(),b.detach()) and torch.equal(ra,torch.get_rng_state());assert m._pc_second_captures==0
  original=reference.build(original_config(c['model_seed']),'cpu').train();a,_=fixture(original,1112);ra=torch.get_rng_state();wa=torch.autograd.grad(a,dict(original.named_parameters())[W])[0]
  for row in d.rows(d.SEEDS[:1]):
   c=d.config(row,str(base),profile,metadata);m=api.build(c,'cpu').train();captures=[];proxies=[];forwards=[];saved={n:v.clone() for n,v in m.state_dict().items()};old_corrected=me.corrected;old_surrogate=me.surrogate
   def record(parameters,gradients,eta,mode):
    value=old_corrected(parameters,gradients,eta,mode);captures.append((parameters,gradients,value));return value
   def record_proxy(parameters,deltas):
    proxy=old_surrogate(parameters,deltas);proxies.append(proxy);return proxy
   handle=m.get_submodule('id_backbone.encoder.core.id_backbone.time_down').register_forward_pre_hook(lambda *args:forwards.append(1));me.corrected=record;me.surrogate=record_proxy
   try:total,metrics=fixture(m,1112)
   finally:me.corrected=old_corrected;me.surrogate=old_surrogate;handle.remove()
   assert len(captures)==len(proxies)==1 and len(forwards)==2 and m._pc_second_captures==1 and m._pc_scheduled==1 and m._pc_vjp_calls==3 and m._pc_pending is None
   assert len(m._supcon_records[-1]['details'])==4 and m._legacy_extra_packets==1112*128 and not m._supcon_handles and m._supcon_pending is None
   assert torch.equal(a.detach(),total.detach()) and torch.equal(ra,torch.get_rng_state())
   named=dict(m.named_parameters());parameters,grads,(deltas,info)=captures[0];proxy=proxies[0];auxw=torch.autograd.grad(proxy,named[W],allow_unused=True,retain_graph=True)[0];assert auxw is None
   mainw=torch.autograd.grad(total,named[W],retain_graph=True)[0];torch.testing.assert_close(mainw,wa,rtol=0,atol=0)
   assert info['nonzero_blocks']>0 and sum(p.numel() for _,p in parameters)==231233 and all(not n.startswith('id_backbone.encoder.core.id_backbone.cls_head.') for n,p in parameters)
   total.backward();unused=0;changed=[]
   for (name,p),gl,delta in zip(parameters,grads[0],deltas):
    if gl is None:assert p.grad is None and delta is None;unused+=1;continue
    expected=gl if delta is None else gl+delta;torch.testing.assert_close(p.grad,expected,rtol=2e-4,atol=3e-6)
    if delta is not None:changed.append(name)
   assert changed and any('.t2.' in n for n in changed)
   assert all(any('sequence_pool.views.'+str(i)+'.' in n for n,p in parameters) for i in range(3))
   assert all(delta is None for (name,p),delta in zip(parameters,deltas) if any(b['block']=='core/sinc' and not b['conflict'] for b in info['blocks']) and '.sinc.' in name)
   assert named[W].grad is not None and torch.isfinite(named[S].grad).all() and all(torch.equal(v,m.state_dict()[n]) for n,v in saved.items())
   directions.append(torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1) for (_,p),g in zip(parameters,deltas)]))
   teacher=copy.deepcopy(m).eval();calls=teacher._pc_calls
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256))
   assert teacher._pc_calls==calls and teacher._pc_pending is None
   m.eval();x=torch.randn(2,2,256)
   with torch.no_grad(),fresh_parameter_caches(m):actual=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
   assert torch.equal(actual,exported) and sum(p.numel() for p in m.parameters())==900722
   flows.append(dict(treatment=c['treatment'],actual_identity_forwards=2,extra_model_forwards=0,aux_W_gradient=None,main_W_gradient_exact=True,feature_gradient_oracle=True,unused_leaves_remain_None=unused,selected_feature_parameters=231233,changed_parameters=changed,combined_backward=True,record=m._pc_records[-1]))
 assert float((directions[0]-directions[1]).norm())>1e-7
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 ref=Path(profile).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source';rc=d.read(ref/'local_source_complete.json')['config'];source_proof(rc);checkpoint=torch.load(ref/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,rc);smoke_model=api.build(c,'cpu').eval();smoke_model.load_state_dict(checkpoint['model'],strict=True)
 with torch.no_grad(),fresh_parameter_caches(smoke_model):smoke=smoke_model.forward_identity_only(torch.zeros(2,2,256))
 assert smoke['tx_logits'].shape==(2,6) and torch.isfinite(smoke['tx_logits']).all()
 return dict(status='PASS',math=math,protocol_negative_cases=negative,actual_feature_gradient_difference=float((directions[0]-directions[1]).norm()),scheduled_calls=2498,source_geometry_promotion_test=True,individual_weak_guard=True,coverage_guard=True,actual_F_flows=flows,real_checkpoint_no_query_smoke=True,loaded_checkpoint_not_inherited_by_training=True,default_head_joint=True,feature_export_exact=True,EMA_eval_untouched=True,warmup_and_inactive_step_exact=True,scalar_and_RNG_exact=True,parameters=900722,IQ_reads=0,target_reads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run(*sys.argv[1:4]),allow_nan=False))
