"""Scoped native view and logging hooks;teacher qualification and native losses stay original."""
import time
import torch
from .u_routing import qualification_positions
class PendingLoader:
 def __init__(self,loader,observer):self.loader=loader;self.observer=observer
 def __getattr__(self,k):return getattr(self.loader,k)
 def __len__(self):return len(self.loader)
 def __iter__(self):
  for batch in self.loader:
   x,y,domains,meta=batch
   if bool((y>=0).any()):raise ValueError('U labels must remain hidden -1')
   if self.observer.pending_U is not None:raise ValueError('Previous U batch has not consumed its native student loss')
   self.observer.pending_U=dict(meta=meta,count=len(y),hidden_y=True)
   yield batch
class NativeUScope:
 def __init__(self,observer,native):self.o=observer;self.n=native;self.teacher_handle=None;self.bound_model=None
 def install(self):
  n=self.n;self.old={k:getattr(n,k) for k in ('_threshold_mask','_temporal_mask_tensor','_strong_augment')}
  n._threshold_mask=self.threshold;n._temporal_mask_tensor=self.temporal;n._strong_augment=self.strong
  if getattr(self.o,'model',None) is not None:self.model_created(self.o.model)
 def model_created(self,model):
  if self.bound_model is model:return
  if self.bound_model is not None or getattr(model,'_source_unlabeled_feature_hook',None) is not None or getattr(model,'_source_unlabeled_selection_hook',None) is not None:raise ValueError('Original recipe must have no competing model U hooks')
  self.bound_model=model;self.old_feature_present=hasattr(model,'_source_unlabeled_feature_hook');self.old_feature=getattr(model,'_source_unlabeled_feature_hook',None)
  # A function closure is copied unchanged into EMA;only the student's native callsite invokes it.
  def observed_feature(**kwargs):return self.feature(**kwargs)
  model._source_unlabeled_feature_hook=observed_feature
 def teacher_created(self,teacher):
  def pre(m,args):
   p=self.o.pending_U
   if p is not None:
    if m.training or torch.is_grad_enabled() or len(args[0])!=p['count'] or 'teacher_raw' in p:raise ValueError('Exactly one eval/no-grad teacher raw U pass required')
    p['teacher_raw']=args[0];p['teacher_raw_no_grad']=True
  self.teacher_handle=teacher.register_forward_pre_hook(pre)
 def threshold(self,conf,domains,args):
  result=self.old['_threshold_mask'](conf,domains,args);p=self.o.pending_U
  if p is not None:
   if 'teacher_raw' not in p or 'conf_mask' in p or args.pseudo_strong_agreement or args.pseudo_domain_gate:raise ValueError('Common teacher-only qualification must precede student strong')
   self.native_args=args
   p.update(conf=conf.detach().clone(),conf_mask=result.detach().clone(),domains=domains.detach().clone(),threshold_policy=dict(mode=args.pseudo_threshold_mode,quantile=float(args.pseudo_quantile),tau_min=float(args.tau_min),tau_max=float(args.tau_max)))
  return result
 def temporal(self,pseudo,conf,extra,args,device):
  result=self.old['_temporal_mask_tensor'](pseudo,conf,extra,args,device);p=self.o.pending_U
  if p is not None:
   if 'conf_mask' not in p or args.pseudo_temporal_mode!='batch_neighbor' or not args.pseudo_temporal_gate or not torch.equal(conf,p['conf']):raise ValueError('Original batch_neighbor qualification required before student')
   p.update(pseudo=pseudo.detach().clone(),temporal_mask=result.detach().clone(),temporal_policy=dict(mode='batch_neighbor',window=int(args.pseudo_temporal_window),minimum_confidence=float(args.pseudo_temporal_min_conf)))
  return result
 def strong(self,x,std):
  start=time.perf_counter();strong=self.old['_strong_augment'](x,std);strong_seconds=time.perf_counter()-start;p=self.o.pending_U
  if p is None:return strong
  if float(std)!=.015 or 'temporal_mask' not in p or x is not p['teacher_raw']:raise ValueError('Native raw teacher U and full Gaussian draw must precede routing')
  local_step=len(self.o.u_records)+1;epoch=(local_step-1)//222+1
  rng_check=bool(getattr(self.o,'smoke_rng_check',False))
  if rng_check:
   import random,numpy as np
   before_cpu=torch.random.get_rng_state().clone();before_np=np.random.get_state();before_py=random.getstate()
  # Domain and student-agreement gates are disabled by the common recipe.
  # Reuse only the two original returned masks; do not recompute qualification.
  qualification=p['conf_mask']&p['temporal_mask'];p['routing_qualification']=qualification.detach().clone()
  student,record=self.o.u_router.route(x,strong,p['meta'],qualification,epoch=epoch+20,step=local_step+4440)
  if rng_check:
   after_np=np.random.get_state()
   if not torch.equal(before_cpu,torch.random.get_rng_state()) or before_np[0]!=after_np[0] or not np.array_equal(before_np[1],after_np[1]) or before_np[2:]!=after_np[2:] or before_py!=random.getstate():raise ValueError('Private route changed global RNG after the existing strong draw')
   record['smoke_CPU_Torch_numpy_Python_RNG_unchanged_by_route']=True
  record['original_strong_cpu_wall_seconds']=strong_seconds;p['record']=record;p['student_input']=student
  return student
 def feature(self,*,model,teacher,student_output,teacher_output,extra,domains,optimizer_step,identity_loss,pseudo_loss,entropy_loss):
  p=self.o.pending_U;args=self.native_args
  if p is None or 'record' not in p or model is not self.o.model or args.pseudo_strong_agreement or getattr(model,'_source_unlabeled_selection_hook',None) is not None or optimizer_step!=p['record']['step']:raise ValueError('Actual native model U feature-hook interface differs')
  logits=student_output['tx_logits'];pseudo=p['pseudo'];mask=p['conf_mask']&p['temporal_mask'];loss_u=pseudo_loss;loss_ent=entropy_loss;epoch=p['record']['epoch'];batch_idx=(optimizer_step-1)%222+1
  if logits.shape!=(p['count'],6) or teacher_output['tx_logits'].shape!=logits.shape or not torch.equal(domains,p['domains']):raise ValueError('Actual native U output/domain interface differs')
  start=time.perf_counter();r=p['record'];positions=r['selected_positions'];eligible=mask.detach().cpu().tolist();qualified=qualification_positions(mask,p['count'])
  if not torch.equal(mask,p['routing_qualification']) or r['qualification_positions']!=qualified or r['qualification_count']!=len(qualified) or r['selected_slots']!=len(qualified) or r['As_packets']!=len(qualified) or (r['mode']=='QUALIFIED_PRACTICAL' and positions!=qualified):raise ValueError('Routing and native teacher qualification must use the same unchanged mask')
  r.update(native_epoch=int(epoch),native_batch_idx=int(batch_idx),teacher_conf=p['conf'].cpu().tolist(),teacher_pseudo=pseudo.detach().cpu().tolist(),domains=p['domains'].cpu().tolist(),conf_mask=p['conf_mask'].cpu().tolist(),temporal_mask=p['temporal_mask'].cpu().tolist(),eligibility=eligible,threshold_policy=p['threshold_policy'],temporal_policy=p['temporal_policy'],domain_gate=False,strong_agreement=False,teacher_raw_same_input=True,teacher_eval_no_grad=p['teacher_raw_no_grad'],teacher_qualification_completed_before_strong=True,teacher_NN=p['count'],student_NN=p['count'],student_logits=logits.detach().cpu().tolist(),native_loss_u=float(loss_u.detach()),native_loss_entropy=float(loss_ent.detach()),lambda_u=float(args.lambda_u),lambda_entropy=float(args.lambda_ent),CE_denominator=int(mask.sum().detach()),entropy_denominator=p['count'],eligible_selected_slots=sum(bool(eligible[i]) for i in positions),eligible_Gaussian_slots=sum(bool(eligible[i]) for i in r['Gaussian_positions']),eligible_As_slots=sum(bool(eligible[i]) for i in positions),CE_label_smoothing=0.,native_global_L_label_smoothing=float(args.label_smoothing),extra_CE_operator_calls=0,entropy_all_U=True,entropy_includes_unqualified=True,entropy_includes_As=bool(r['As_packets']),U_truth_used=False,extra_teacher=False,native_losses_preserved=True,native_feature_adjustment=None,eligibility_source='actual frozen native base_mask_and_ones with no selectionhook',native_optimizer_step=int(optimizer_step),native_hook='model._source_unlabeled_feature_hook')
  r['logging_cpu_wall_seconds']=time.perf_counter()-start;r['CPU_timing']='perf_counter host wall;no explicit CUDA sync;detach readouts may synchronize;route includes selection/simulation/copy,do not add inclusive route again'
  self.o.u_records.append(r);self.o.pending_U=None
  return None,{}
 def restore(self):
  for k,v in self.old.items():setattr(self.n,k,v)
  if self.teacher_handle is not None:self.teacher_handle.remove()
  if self.bound_model is not None:
   if self.old_feature_present:self.bound_model._source_unlabeled_feature_hook=self.old_feature
   else:delattr(self.bound_model,'_source_unlabeled_feature_hook')
