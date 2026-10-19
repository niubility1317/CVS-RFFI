"""Focused second-order, actual F/main/head and protocol-negative checks."""
import copy,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as me
from .objective import active,anchor_for,masks,balanced_ce
from experiments.cvs_f_supcon50.checks import W,S
from experiments.cvs_f_supcon50 import model as reference
from .baseline import original_config
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches

def fixture(m,step):
 from cvsrffi.tensors import unpack_batch
 torch.manual_seed(7061);x=torch.randn(128,2,256);y=torch.arange(128)%6;domain=torch.arange(128)%15;rx=torch.tensor((1,1,1,3,3,3,4,4,4,6,6,6,8,8,8))[domain];prior=step-1
 m._feature_l_steps=prior;m._legacy_calls=prior;m._legacy_extra_packets=prior*128;m._supcon_calls=prior
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);_,_,extra=unpack_batch((x,y,domain,{'rx_i':rx}))
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra=extra,domains=domain,optimizer_step=step,identity_loss=ce,labeled_count=128)
 return ce+correction,metrics

def run(profile,metadata,output):
 torch.set_num_threads(2);base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata);negative=0;flows=[];gradients=[]
 for key,value in [('epochs',20),('test_views',['practical_high']),('weak_indices',[1]),('meta_inner_alpha',.05),('separate_head_training',True),('meta_every',1)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Altered source/defaulthead/fixed recipe accepted')
 for labels,rx in [(torch.arange(128)%6-1,torch.ones(128,dtype=torch.long)),(torch.arange(128)%6,torch.full((128,),2))]:
  try:masks(labels,rx,1)
  except ValueError:negative+=1
  else:raise AssertionError('Hidden/foreign TX or RX accepted')
 scalar=torch.tensor(.7,dtype=torch.float64,requires_grad=True);inner=.5*(scalar-.2).square();g=torch.autograd.grad(inner,scalar,create_graph=True)[0];virtual=scalar-.25*g;objective=.5*(inner+.5*(virtual+.4).square());actual=torch.autograd.grad(objective,scalar)[0];expected=.5*((.7-.2)+((.7-.25*(.7-.2))+.4)*.75)
 if abs(float(actual)-expected)>1e-12:raise AssertionError('Second-order scalar oracle differs')
 calendar=[anchor_for(s) for s in range(1,11101) if active(s)]
 if len(calendar)!=624 or [calendar.count(r) for r in (1,3,4,6,8)]!=[125,125,125,125,124]:raise AssertionError('Meta RX calendar differs')
 refgeo=d.reference_geometry(d.SEEDS[0]);baseV=dict(recall=[.98]*6,accuracy=.98,worst_RX=.97);good=dict(baseV,geometry=copy.deepcopy(refgeo),metarx_scheduled=624,metarx_eligible_fraction=1.)
 for index in (1,3):good['geometry']['per_TX'][index]['out_RX_margin']['q10']+=.003;good['geometry']['per_TX'][index]['RX_spread_to_inter_TX_gap']*=.99
 if d.expansion({a:good for a in d.ARMS},baseV)['expand_arms']!=list(d.ARMS):raise AssertionError('Independent source geometry promotion failed')
 bad=copy.deepcopy(good);bad['geometry']['per_TX'][1]['out_RX_margin']['q10']-=.003
 if d.expansion({a:bad for a in d.ARMS},baseV)['expand_arms']:raise AssertionError('One weak TX may not hide the other')
 bad=copy.deepcopy(good);bad['metarx_eligible_fraction']=.98
 if d.expansion({a:bad for a in d.ARMS},baseV)['expand_arms']:raise AssertionError('Ineligible episodes promoted')
 with numerical_context(d.FULL_FP32_POLICY):
  control=reference.build(original_config(c['model_seed']),'cpu').train();m=api.build(c,'cpu').train()
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in control.state_dict().items()):raise AssertionError('Meta changes initial deployed network')
  a,_=fixture(control,1);b,_=fixture(m,1)
  if not torch.equal(a.detach(),b.detach()):raise AssertionError('Warmup differs from original objective')
  for row in d.rows(d.SEEDS[:1]):
   c=d.config(row,str(base),profile,metadata);m=api.build(c,'cpu').train();captured=[];saved={n:v.clone() for n,v in m.state_dict().items()};original=me.objective
   def record(*args,**kwargs):
    value=original(*args,**kwargs);captured.append(value[0]);return value
   me.objective=record
   try:total,metrics=fixture(m,1120)
   finally:me.objective=original
   if len(captured)!=1 or m._metarx_applied!=1 or m._metarx_packets<=128 or len(m._supcon_records[-1]['details'])!=4 or m._legacy_extra_packets!=1120*128 or m._supcon_handles or m._supcon_pending is not None:raise AssertionError('Actual L/RDrop/SupCon lifetime differs')
   named=dict(m.named_parameters());aux=captured[0];w,s,p=torch.autograd.grad(aux,[named[W],named[S],named['id_backbone.encoder.core.id_backbone.t1.pw.weight']],allow_unused=True,retain_graph=True);mainw=torch.autograd.grad(total,named[W],retain_graph=True)[0]
   if w is not None or s is None or p is None or not torch.isfinite(s).all() or s.norm()<=0 or mainw.norm()<=0 or any(q.grad is not None for q in m.parameters()):raise AssertionError('Feature/head auxiliary boundary differs')
   gradients.append(p.detach().clone());total.backward()
   if named[W].grad is None or not torch.isfinite(named[W].grad).all() or named[S].grad is None or not torch.isfinite(named[S].grad).all() or any(not torch.equal(v,m.state_dict()[n]) for n,v in saved.items()):raise AssertionError('Combined second-order main backward changed parameters or failed')
   teacher=copy.deepcopy(m).eval();calls=len(teacher._metarx_records)
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256))
   if len(teacher._metarx_records)!=calls:raise AssertionError('EMA/eval invokes meta update')
   m.eval();x=torch.randn(2,2,256)
   with torch.no_grad(),fresh_parameter_caches(m):actual=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
   if not torch.equal(actual,exported) or sum(p.numel() for p in m.parameters())!=900722:raise AssertionError('Deployed F feature altered')
   flows.append(dict(treatment=c['treatment'],actual_extra_L_packets=m._metarx_packets,aux_W_gradient=None,aux_Sinc_gradient_norm=float(s.norm()),main_W_gradient_norm=float(mainw.norm()),combined_backward=True,actual_records=m._metarx_records[-1]))
 if float((gradients[0]-gradients[1]).norm())<=1e-7:raise AssertionError('Differentiable inner update did not alter actual feature derivative')
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 ref=Path(profile).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source';rc=d.read(ref/'local_source_complete.json')['config'];source_proof(rc);checkpoint=torch.load(ref/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,rc);smoke_model=api.build(c,'cpu').eval();smoke_model.load_state_dict(checkpoint['model'],strict=True)
 with torch.no_grad(),fresh_parameter_caches(smoke_model):smoke=smoke_model.forward_identity_only(torch.zeros(2,2,256))
 if smoke['tx_logits'].shape!=(2,6) or not torch.isfinite(smoke['tx_logits']).all():raise AssertionError('Actual matching final50 checkpoint no-query smoke failed')
 return dict(status='PASS',protocol_negative_cases=negative,scalar_second_order_oracle=True,actual_feature_gradient_difference=float((gradients[0]-gradients[1]).norm()),scheduled_calls=624,source_geometry_promotion_test=True,individual_weak_guard=True,eligible_episode_guard=True,actual_F_flows=flows,real_checkpoint_no_query_smoke=True,loaded_checkpoint_not_inherited_by_training=True,default_head_joint=True,feature_export_exact=True,EMA_eval_untouched=True,warmup_exact=True,parameters=900722,IQ_reads=0,target_reads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run(*sys.argv[1:4]),allow_nan=False))
