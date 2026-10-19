"""Actual F paired-feature flow, legal source recipe and no-query smoke."""
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
 if hasattr(m,'_tail_calls'):m._tail_calls=prior
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);_,_,extra=unpack_batch((x,y,domain.cpu(),{'rx_i':rx.cpu()}))
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra=extra,domains=domain,optimizer_step=step,identity_loss=ce,labeled_count=128)
 return ce+correction,metrics

def run(profile,metadata,output):
 torch.set_num_threads(2);math=math_checks()
 try:
  from . import design as d,model as api,mechanism as me
 except ImportError as e:raise AssertionError('Weak tail loss not connected to actual deployed F training') from e
 from experiments.cvs_f_supcon50 import model as reference
 from .baseline import original_config
 base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata);negative=math['protocol_negative_cases'];flows=[];gradients=[]
 for key,value in [('epochs',20),('test_views',['practical_high']),('weak_indices',[1]),('tail_fraction',.25),('separate_head_training',True),('tail_levels',['joint160']),('tail_RX_balanced',False),('tail_pair_average_before_selection',False)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Altered source/defaulthead/tail recipe accepted')
 refgeo=d.reference_geometry(d.SEEDS[0]);baseV=dict(recall=[.98]*6,accuracy=.98,worst_RX=.97);good=dict(baseV,geometry=copy.deepcopy(refgeo),tail_scheduled=9990,tail_eligible_fraction=1.,tail_cell_coverage=1.)
 for index in (1,3):good['geometry']['per_TX'][index]['out_RX_margin']['q10']+=.003;good['geometry']['per_TX'][index]['RX_spread_to_inter_TX_gap']*=.99
 if d.expansion({a:good for a in d.ARMS},baseV)['expand_arms']!=list(d.ARMS):raise AssertionError('Fixed source rule cannot promote good features')
 for field,value in [('one_weak',None),('tail_eligible_fraction',.98),('tail_cell_coverage',.94)]:
  bad=copy.deepcopy(good)
  if field=='one_weak':bad['geometry']['per_TX'][1]['out_RX_margin']['q10']-=.003
  else:bad[field]=value
  if d.expansion({a:bad for a in d.ARMS},baseV)['expand_arms']:raise AssertionError('One weak TX or coverage failure hidden by average')
 with numerical_context(d.FULL_FP32_POLICY):
  original=reference.build(original_config(c['model_seed']),'cpu').train();m=api.build(c,'cpu').train()
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in original.state_dict().items()):raise AssertionError('Tail objective changes initial F parameters')
  reference_calls=[];handle=original.get_submodule('id_backbone.encoder.core.id_backbone.time_down').register_forward_pre_hook(lambda *args:reference_calls.append(1))
  try:a,_=fixture(original,1)
  finally:handle.remove()
  if len(reference_calls)!=2:raise AssertionError('Reference physical identity forward count differs')
  b,_=fixture(m,1)
  if not torch.equal(a.detach(),b.detach()):raise AssertionError('Warmup differs from original objective')
  for row in d.rows(d.SEEDS[:1]):
   c=d.config(row,str(base),profile,metadata);m=api.build(c,'cpu').train();captured=[];forwards=[];saved={n:v.clone() for n,v in m.state_dict().items()};fn=me.penalty
   def record(*args,**kwargs):
    value=fn(*args,**kwargs);captured.append(value);return value
   handle=m.get_submodule('id_backbone.encoder.core.id_backbone.time_down').register_forward_pre_hook(lambda *args:forwards.append(1));me.penalty=record
   try:total,metrics=fixture(m,1111)
   finally:me.penalty=fn;handle.remove()
   if len(captured)!=1 or len(forwards)!=2 or m._tail_applied!=1 or len(m._supcon_records[-1]['details'])!=4 or m._legacy_extra_packets!=1111*128 or m._supcon_handles or m._supcon_pending is not None:raise AssertionError('Original paired L/RDrop/SupCon lifetime changed')
   named=dict(m.named_parameters());middle=next(n for n in named if '.t2.' in n and named[n].requires_grad);aux=captured[0][0]
   w,s,shallow,mid=torch.autograd.grad(aux,[named[W],named[S],named['id_backbone.encoder.core.id_backbone.t1.pw.weight'],named[middle]],allow_unused=True,retain_graph=True);mainw=torch.autograd.grad(total,named[W],retain_graph=True)[0]
   if w is not None or any(v is None or not torch.isfinite(v).all() or v.norm()<=0 for v in (s,shallow,mid,mainw)) or any(p.grad is not None for p in m.parameters()):raise AssertionError('Actual feature gradient or default head boundary differs')
   gradients.append(shallow.detach().clone());total.backward()
   if named[W].grad is None or named[S].grad is None or not torch.isfinite(named[W].grad).all() or not torch.isfinite(named[S].grad).all() or any(not torch.equal(v,m.state_dict()[n]) for n,v in saved.items()):raise AssertionError('Combined main/tail backward failed or changed parameters')
   teacher=copy.deepcopy(m).eval();calls=teacher._tail_calls
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256))
   if teacher._tail_calls!=calls:raise AssertionError('EMA/eval invokes tail objective')
   m.eval();x=torch.randn(2,2,256)
   with torch.no_grad(),fresh_parameter_caches(m):actual=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
   if not torch.equal(actual,exported) or sum(p.numel() for p in m.parameters())!=900722:raise AssertionError('Deployed feature or parameter budget changed')
   flows.append(dict(treatment=c['treatment'],actual_identity_forwards=2,extra_model_forwards=0,aux_W_gradient=None,aux_Sinc_gradient_norm=float(s.norm()),aux_shallow_gradient_norm=float(shallow.norm()),aux_middle_gradient_norm=float(mid.norm()),middle_parameter=middle,main_W_gradient_norm=float(mainw.norm()),combined_backward=True,actual_tail_record=m._tail_records[-1]))
 if float((gradients[0]-gradients[1]).norm())<=1e-7:raise AssertionError('Tail selection did not change actual feature gradient')
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 ref=Path(profile).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source';rc=d.read(ref/'local_source_complete.json')['config'];source_proof(rc);checkpoint=torch.load(ref/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,rc);smoke_model=api.build(c,'cpu').eval();smoke_model.load_state_dict(checkpoint['model'],strict=True)
 with torch.no_grad(),fresh_parameter_caches(smoke_model):smoke=smoke_model.forward_identity_only(torch.zeros(2,2,256))
 if smoke['tx_logits'].shape!=(2,6) or not torch.isfinite(smoke['tx_logits']).all():raise AssertionError('Actual matching final50 no-query smoke failed')
 return dict(status='PASS',math=math,protocol_negative_cases=negative,actual_feature_gradient_difference=float((gradients[0]-gradients[1]).norm()),scheduled_calls=9990,source_geometry_promotion_test=True,individual_weak_guard=True,coverage_guard=True,actual_F_flows=flows,real_checkpoint_no_query_smoke=True,loaded_checkpoint_not_inherited_by_training=True,default_head_joint=True,feature_export_exact=True,EMA_eval_untouched=True,warmup_exact=True,parameters=900722,IQ_reads=0,target_reads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run(*sys.argv[1:4]),allow_nan=False))
