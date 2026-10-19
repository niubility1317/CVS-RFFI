"""Original scalar/gradient/clip/AdamW moments, followed by feature-only displacement."""
from types import MethodType
import torch
from .core import select,blocks,contributions,route,active,strength
W='id_backbone.encoder.core.id_backbone.cls_head.weight'
def identity_forward(self,*args,**kwargs):
 out=self._step_original_identity(*args,**kwargs);capture=self._step_capture
 if capture is not None:
  if not self.training or not torch.is_grad_enabled() or self._supcon_phase!='second' or capture['second'] is not None or out['tx_logits'].shape!=(128,6):raise ValueError('Only actual original second L logits')
  capture['second']=out['tx_logits'];self._step_second_captures+=1
 return out
def training_hook(self,**kw):
 step=kw['optimizer_step'];scheduled=active(step)
 if not self.training or not torch.is_grad_enabled() or step!=self._step_calls+1 or self._step_capture is not None or self._step_pending is not None:raise ValueError('Ordered single native optimizer ownership required')
 capture=dict(second=None) if scheduled else None;self._step_capture=capture;eligible=False;counts=[];pending=dict(step=step);vjp=0
 try:
  correction,metrics=self._step_original_hook(**kw)
  if scheduled:
   if capture['second'] is None or kw['labeled_count']!=128:raise ValueError('Missing actual paired L128 logits')
   losses,counts=contributions(kw['labeled_output']['tx_logits'],capture['second'],kw['labels']);eligible=all(n>0 for n in counts)
   if eligible:
    parameters=select(self);ps=[p for _,p in parameters];grads=[torch.autograd.grad(loss,ps,retain_graph=True,create_graph=False,allow_unused=True) for loss in losses];g=torch.stack([torch.cat([(torch.zeros_like(p) if z is None else z.detach()).reshape(-1) for (_,p),z in zip(parameters,values)]) for values in grads]).detach().double().cpu();before=torch.cat([p.detach().reshape(-1) for _,p in parameters]).cpu().clone();groups=blocks(parameters)
    pending.update(parameters=parameters,gradients=g,before=before,block_ranges=[(v['start'],v['end']) for v in groups],head_before=dict(self.named_parameters())[W].detach().cpu().clone());vjp=3;self._step_eligible+=1;self._step_vjp_calls+=3
  record=dict(step=step,epoch=(step-1)//222+1,scheduled=scheduled,eligible=eligible,class_counts=counts,strength=strength(step) if scheduled else 0.,feature_VJP_calls=vjp,finalized=False,extra_model_forwards=0,extra_optimizer_steps=0,head_fit_updates=0);pending['record']=record;self._step_pending=pending;self._step_records.append(record);self._step_calls=step;self._step_scheduled+=int(scheduled)
  metrics.update({'F_actual_step/calls':step,'F_actual_step/scheduled':int(scheduled),'F_actual_step/scheduled_calls':self._step_scheduled,'F_actual_step/eligible_calls':self._step_eligible,'F_actual_step/feature_VJP_calls':self._step_vjp_calls,'F_actual_step/previous_post_steps':self._step_post_calls,'F_actual_step/previous_changed_steps':self._step_changed,'F_actual_step/strength':record['strength'],'F_actual_step/extra_model_forwards':0,'F_actual_step/head_direct_aux_gradient':0})
  return correction,metrics
 finally:self._step_capture=None
training_hook._accepts_labeled_inputs=False
def bind_optimizer(m,opt):
 if hasattr(opt,'_feature_step_owner'):raise ValueError('One actual feature-step owner per optimizer')
 opt._feature_step_owner=True
 def after_native(opt,args,kwargs):
  pending=m._step_pending
  if pending is None or pending['step']!=m._step_post_calls+1 or pending['step']!=opt._sinc_unit_steps or pending['step']!=m._step_calls:raise ValueError('Complete native AdamW before feature projection required')
  record=pending['record'];record.update(actual_correction_nonzero=False,solver_fallback=False,projection_storage_fallback=False,storage_match_fallback=False,stored_norm_match_error=0.,constraint_enforced=False)
  if record['eligible']:
   parameters=pending['parameters'];native=torch.cat([p.detach().reshape(-1) for _,p in parameters]).cpu();head=dict(m.named_parameters())[W];raw_head=head.detach().clone()
   pc,control,info=route(pending['before'],native,pending['gradients'],pending['block_ranges'],record['strength']);chosen=pc if m._step_mode=='UPDATE_CONE' else control
   actual_changes=bool((chosen!=native).any());target=chosen.to(parameters[0][1].device);offset=0
   with torch.no_grad():
    for name,p in parameters:
     end=offset+p.numel()
     if not torch.equal(chosen[offset:end],native[offset:end]):p.copy_(target[offset:end].reshape_as(p))
     offset=end
   if not torch.equal(raw_head,head):raise ValueError('Actual feature correction touched default W')
   stored=torch.cat([p.detach().reshape(-1) for _,p in parameters]).cpu()
   if not torch.equal(stored,chosen):raise ValueError('Stored feature step differs from checked FP32 displacement')
   record.update(info);record['candidate_stored_normalized_dots']=record.pop('stored_normalized_dots');record['candidate_full_strength_max_residual_relative']=record.pop('full_strength_max_residual_relative');record['candidate_constraint_enforced']=record.pop('constraint_enforced');record['candidate_normalized_correction_fraction']=record.pop('normalized_correction_fraction');is_cone=m._step_mode=='UPDATE_CONE';record['applied_normalized_dots']=record['candidate_stored_normalized_dots'] if is_cone else record['control_stored_normalized_dots'];record['applied_full_strength_max_residual_relative']=max(record['applied_normalized_dots'])/max(record['raw_step_normalized_norm'],1e-30);record['constraint_enforced']=is_cone and record['candidate_constraint_enforced'];record['applied_normalized_correction_fraction']=record['candidate_normalized_correction_fraction'] if is_cone else record['control_normalized_correction_fraction'];record['applied_raw_cosine']=record['raw_projected_cosine'] if is_cone else record['control_raw_cosine'];record.update(actual_correction_nonzero=actual_changes,default_W_post_correction_exact=True,native_main_W_update_norm=float((raw_head.detach().cpu().double()-pending['head_before'].double()).norm()),selected_feature_parameters=sum(p.numel() for _,p in parameters));m._step_changed+=int(actual_changes);m._step_fallbacks+=int(info['solver_fallback'] or info['projection_storage_fallback'] or info['storage_match_fallback']);m._step_max_norm_error=max(m._step_max_norm_error,info['stored_norm_match_error'])
  record['finalized']=True;m._step_post_calls+=1;m._step_pending=None
 opt.register_step_post_hook(after_native)
 return opt
def attach(m,c):
 if m._supcon_calls or hasattr(m,'_step_mode'):raise ValueError('Attach once before scratch joint training')
 m._step_mode=c['treatment'];m._step_capture=None;m._step_pending=None;m._step_calls=0;m._step_post_calls=0;m._step_scheduled=0;m._step_eligible=0;m._step_changed=0;m._step_vjp_calls=0;m._step_second_captures=0;m._step_fallbacks=0;m._step_max_norm_error=0.;m._step_records=[];m._step_original_identity=m.forward_identity_only;m.forward_identity_only=MethodType(identity_forward,m);m._step_original_hook=m._source_labeled_feature_hook;m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['actual_feature_step']=dict(mode=c['treatment'],risk_groups=['L_weak1_CE','L_weak3_CE','L_other4_CE'],actual_sum_divided_by128=True,default_head_main_joint=True,post_correction_W_untouched=True,full_deployed_features=True,actual_native_gradient_clip_AdamW_moments_unchanged=True,poststep_before_EMA=True,full_step_includes_U_RDrop_SC_physical_decay=True,scalar_objective_unchanged=True,Sinc_origin_decay_applied_once_in_native=True,extra_model_forwards=0,extra_parameters=0,extra_real_optimizer_steps=0,warm_epochs=5,ramp_epochs=10,every_steps=4,scheduled_calls=2498,extra_feature_VJP_calls_max=7494,metric='native complete step RMS per feature block; zero-step blocks stay zero',matching='FP32 stored same-state normalized full step norm; control retains original native direction; numerical fallback logs retained',guarantee='Only full-strength nonfallback feature-only first-order current L risks up to storage tolerance; no joint-head/nonlinear/heldRX/target guarantee')
 return m
