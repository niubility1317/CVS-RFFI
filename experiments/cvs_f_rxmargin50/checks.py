"""Focused angle oracle and actual F/RDrop feature-gradient boundaries."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
from . import design as d,model as api,mechanism as me
from .loss import angular_margin
from experiments.cvs_f_supcon50.checks import fixture,W,S
from experiments.cvs_f_supcon50 import model as reference
from .baseline import original_config
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches

def run(profile,metadata,output):
 torch.set_num_threads(2);base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata);negative=0
 for key,value in [('epochs',20),('test_views',['practical_high']),('RX_margin',.1),('class_weights',[1.5]*6),('separate_head_training',True)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Altered fixed recipe accepted')
 y=torch.tensor([0,0,1,1]);rx=torch.tensor([1,3,1,3]);z=torch.zeros(4,160,dtype=torch.float64);z[:,0]=torch.tensor([1,0,1,0]);z[:,1]=torch.tensor([0,1,0,1]);value,info=angular_margin(z,y,rx)
 if abs(float(value)-1.05)>1e-12 or info['active_fraction']!=1:raise AssertionError('Known angle oracle differs')
 good=torch.zeros_like(z);good[:,0]=torch.tensor([1,1,-1,-1]);good.requires_grad_();zero,_=angular_margin(good,y,rx);g=torch.autograd.grad(zero,good)[0]
 if zero!=0 or g.count_nonzero():raise AssertionError('Satisfied identity margin still forced')
 for labels,receivers in [(y-3,rx),(y,rx*0+2)]:
  try:angular_margin(z,labels,receivers)
  except ValueError:negative+=1
  else:raise AssertionError('Hidden labels/foreignRX accepted')
 singleton=torch.randn(4,160,requires_grad=True);zz,ii=angular_margin(singleton,y,torch.ones_like(rx));gg=torch.autograd.grad(zz,singleton)[0]
 if zz!=0 or gg.count_nonzero() or ii['valid_anchors']:raise AssertionError('Missing crossRX positives must be finite zero')
 # An independent NumPy calculation checks count normalization/class balance.
 torch.manual_seed(67);rz=torch.randn(12,160,dtype=torch.float64);ry=torch.arange(12)%3;rr=torch.tensor([1,1,1,3,3,3,4,4,4,6,6,6]);actual,_=angular_margin(rz,ry,rr);a=rz.numpy().copy();a/=np.linalg.norm(a,axis=1,keepdims=True);sim=a@a.T;vals=[]
 for i in range(12):
  p=(ry.numpy()==int(ry[i]))&(rr.numpy()!=int(rr[i]));n=(ry.numpy()!=int(ry[i]))&(rr.numpy()==int(rr[i]))
  if p.any() and n.any():
   x=-sim[i,p]/.05;t=x.max();sp=-.05*(t+np.log(np.exp(x-t).mean()));vals.append((int(ry[i]),max(.05+sim[i,n].max()-sp,0)))
 if len(vals)!=12 or len(set(x[0] for x in vals))!=3:raise AssertionError('Angle oracle fixture lacks valid crossRX positives or sameRX negatives')
 expected=np.mean([np.mean([v for label,v in vals if label==c]) for c in sorted(set(x[0] for x in vals))])
 if not np.isfinite(expected):raise AssertionError('Independent oracle must be finite')
 if abs(float(actual)-expected)>1e-12:raise AssertionError('Independent class balanced angle formula differs')
 perm=torch.tensor([2,0,1]);shift,_=angular_margin(rz,perm[ry],rr)
 if not torch.equal(actual,shift) and abs(float(actual-shift))>1e-14:raise AssertionError('TX permutation changes objective')
 with numerical_context(d.FULL_FP32_POLICY):
  control=reference.build(original_config(c['model_seed']),'cpu').train();m=api.build(c,'cpu').train()
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in control.state_dict().items()):raise AssertionError('Margin changes baseline initialization')
  warm,_=fixture(control,1,'cpu');newwarm,_=fixture(m,1,'cpu')
  if not torch.equal(warm.detach(),newwarm.detach()):raise AssertionError('Warmup changes native objective')
  captured=[];old=me.angular_margin
  def record(*args,**kw):
   value=old(*args,**kw);captured.append(value[0]);return value
  me.angular_margin=record
  try:total,metrics=fixture(m,15,'cpu')
  finally:me.angular_margin=old
  if len(captured)!=2 or m._supcon_pending is not None or m._supcon_handles or len(m._supcon_records[-1]['details'])!=6 or m._legacy_extra_packets!=((15-1)*222+1)*128:raise AssertionError('Must use exactly existing two real L/RDrop forwards')
  aux=torch.stack(captured).mean();named=dict(m.named_parameters());gw,gs=torch.autograd.grad(aux,[named[W],named[S]],allow_unused=True,retain_graph=True);mainw=torch.autograd.grad(total,named[W],retain_graph=True)[0]
  if gw is not None or gs is None or not torch.isfinite(gs).all() or gs.norm()<=0 or not torch.isfinite(mainw).all() or mainw.norm()<=0:raise AssertionError('Actual feature/head gradient boundary differs')
  teacher=copy.deepcopy(m).eval();calls=teacher._supcon_calls
  with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256))
  if teacher._supcon_calls!=calls or teacher._supcon_pending is not None or teacher._supcon_handles:raise AssertionError('EMA/eval activates margin')
  m.eval();x=torch.randn(2,2,256)
  with torch.no_grad(),fresh_parameter_caches(m):actual=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
  if not torch.equal(actual,exported) or sum(p.numel() for p in m.parameters())!=900722:raise AssertionError('Deployed feature or parameter count changed')
 # Actual completed matching BASE checkpoint, zero-input only. This model
 # is discarded after smoke; every real candidate initializes from scratch.
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 # Resolve from the registered local experiment workspace, not a target path.
 ref=Path(c['runtime_profile']).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source'
 rc=d.read(ref/'local_source_complete.json')['config'];source_proof(rc);checkpoint=torch.load(ref/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,rc)
 actual_model=api.build(c,'cpu');actual_model.load_state_dict(checkpoint['model'],strict=True);actual_model.eval().requires_grad_(False)
 with torch.no_grad(),fresh_parameter_caches(actual_model):smoke=actual_model.forward_identity_only(torch.zeros(2,2,256))
 if smoke['tx_logits'].shape!=(2,6) or smoke['id_feat_joint'].shape!=(2,160) or not torch.isfinite(smoke['tx_logits']).all() or not torch.isfinite(smoke['id_feat_joint']).all():raise AssertionError('Matching final50 checkpoint no-query smoke failed')
 return dict(status='PASS',real_final50_checkpoint_no_query_smoke=True,loaded_checkpoint_not_inherited_by_training=True,angle_oracle=True,independent_numpy_formula=True,satisfied_margin_zero_gradient=True,class_permutation_equivariant=True,protocol_negative_cases=negative,actual_F_RDrop_forward_reuse=True,original_SupCon_retained=True,joint160_captures=2,total_feature_objectives_per_step=6,aux_direct_head_gradient=None,aux_Sinc_gradient_norm=float(gs.norm()),main_head_gradient_norm=float(mainw.norm()),warmup_exact=True,EMA_eval_untouched=True,feature_export_exact=True,parameters=900722,head_fit_updates=0,optimizer_updates=0,IQ_reads=0,target_reads=0)
if __name__=='__main__':print(json.dumps(run(*sys.argv[1:4])))
