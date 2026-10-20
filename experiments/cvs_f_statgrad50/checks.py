"""Focused original-recipe negatives and real-checkpoint, no-query paired flow."""
import copy,tempfile,torch
from pathlib import Path
from torch.nn import functional as F
from . import design as d,model as api,mechanism as me
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context
W='id_backbone.encoder.core.id_backbone.cls_head.weight'
S='id_backbone.encoder.core.id_backbone.sinc.low_coordinate'
def configuration_checks():
 assert len(d.rows())==6 and all([r['model_seed'] for r in d.rows() if r['treatment']==a]==list(d.SEEDS) for a in d.ARMS)
 assert d.fixed_authorization()['fixed_rows']==d.rows() and not d.fixed_authorization()['source_gate_applied']
 for row in d.rows():d.validate(d.config(row,str(Path(tempfile.gettempdir())/d.RUN),'PROFILE','METADATA'))
 return True
def fixture(m,epoch,device,seed=7061):
 from cvsrffi.tensors import unpack_batch
 torch.manual_seed(seed);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;domain=torch.arange(128,device=device)%15;rx=torch.tensor(me.RX_BY_DOMAIN,device=device)[domain];prior=(epoch-1)*222;m._feature_l_steps=prior;m._legacy_calls=prior;m._legacy_extra_packets=prior*128;m._supcon_calls=prior
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);_,_,extra=unpack_batch((x,y,domain.cpu(),dict(rx_i=rx.cpu(),split_source=['ssdg_labeled_tx_visible']*128,tx_label_visible=torch.ones(128,dtype=torch.bool))))
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra=extra,domains=domain,optimizer_step=prior+1,identity_loss=ce,labeled_count=128)
 return ce+correction,metrics
def run(profile,metadata,output):
 configuration_checks();torch.set_num_threads(2);neg=0;c=d.config(d.rows()[0],str(Path(output)/d.RUN),profile,metadata)
 for key,value in [('epochs',20),('test_views',['practical_high']),('model_seed',2026092704),('separate_head_training',True),('statistic_gradient_policy','MEM4'),('contrastive_temperature',.1),('contrastive_max_weight',.2),('baseline_ckpt','oldweight'),('weak_indices',[1]),('checkpoint_selection','best_target'),('extra_model_forwards',1),('class_weights',[1.,2.,1.,2.,1.,1.])]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:neg+=1
  else:raise AssertionError('Changed fixed recipe accepted '+key)
 from .score import preflight
 try:preflight(c,Path(output))
 except FileNotFoundError:neg+=1
 else:raise AssertionError('Missing predictions allowed truth preflight')
 from .analysis import verdict
 try:verdict([],[])
 except ValueError:neg+=1
 else:raise AssertionError('Partial seeds accepted')
 from .checks_math import run as math_checks
 math_cpu=math_checks('cpu');math_cuda=math_checks('cuda:0')
 from .baseline import original_config
 from experiments.cvs_f_supcon50 import model as native
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 root=Path(profile).parent.parent/'runs'/c['reference_local_run']/'BASE-s2026092701/source';rc=d.read(root/'local_source_complete.json')['config'];source_proof(rc);checkpoint=torch.load(root/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,rc)
 flows=[]
 with numerical_context(d.FULL_FP32_POLICY):
  for mode in d.ARMS:
   cc=d.config(next(r for r in d.rows() if r['treatment']==mode),c['runtime_base'],profile,metadata);m=api.build(cc,'cpu');ref=native.build(original_config(2026092701),'cpu')
   assert list(m.state_dict())==list(ref.state_dict())
   m.load_state_dict(checkpoint['model'],strict=True);ref.load_state_dict(checkpoint['model'],strict=True);m.eval();ref.eval();x=torch.zeros(2,2,256)
   with torch.no_grad(),fresh_parameter_caches(m),fresh_parameter_caches(ref):
    a=m.forward_identity_only(x);b=ref.forward_identity_only(x);exported=api.export(m)(x)
   assert torch.equal(a['tx_logits'],b['tx_logits']) and torch.equal(a['id_feat_joint'],b['id_feat_joint']) and torch.equal(a['id_feat_joint'],exported)
   del ref;m.train();saved={n:v.clone() for n,v in m.state_dict().items()};captured=[];prior=me.penalty
   def capture(*args,**kwargs):
    value=prior(*args,**kwargs);captured.append(value[0]);return value
   me.penalty=capture
   try:total,metrics=fixture(m,6,'cpu')
   finally:me.penalty=prior
   assert len(captured)==2 and m._supcon_pending is None and not m._supcon_handles and m._supcon_applied==1 and m._legacy_extra_packets==(5*222+1)*128
   aux=torch.stack(captured).mean();named=dict(m.named_parameters());aw,asinc=torch.autograd.grad(aux,[named[W],named[S]],retain_graph=True,allow_unused=True);mainW=torch.autograd.grad(total,named[W],retain_graph=True)[0]
   assert aw is None and asinc is not None and torch.isfinite(asinc).all() and asinc.norm()>0 and torch.isfinite(mainW).all() and mainW.norm()>0
   assert not any(p.grad is not None for p in m.parameters()) and all(torch.equal(v,m.state_dict()[n]) for n,v in saved.items())
   details=m._supcon_records[-1]['details'];assert len(details)==4 and all(v['backward']['backward_calls']==1 for v in details)
   teacher=copy.deepcopy(m).eval();before=teacher._supcon_calls
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(x)
   assert teacher._supcon_calls==before and teacher._supcon_pending is None and not teacher._supcon_handles
   flows.append(dict(policy=mode,forward_logits_joint160_export_exact=True,two_original_L_forwards=True,aux_direct_head_gradient=None,aux_Sinc_gradient_norm=float(asinc.norm()),joint_head_gradient_norm=float(mainW.norm()),no_model_state_update=True,EMA_eval_untouched=True,temporary_hooks_remaining=0,details=details))
 return dict(status='PASS',protocol_negative_cases=neg,math_cpu=math_cpu,math_cuda=math_cuda,actual_checkpoint_no_query_smoke=True,checkpoint_seed=2026092701,checkpoint_loaded_once_for_both_policies=True,loaded_checkpoint_discarded_not_inherited=True,actual_F_flows=flows,IQ_reads=0,target_reads=0,U_truth_reads=0,head_fit_updates=0,optimizer_updates=0)
