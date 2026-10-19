"""Same native RDrop/SupCon graph; feature-only block weak-pair gradient routing."""
from types import MethodType
import torch
from .core import select,contributions,corrected,surrogate,active,strength
def identity_forward(self,*args,**kwargs):
 out=self._pc_original_identity(*args,**kwargs);pending=self._pc_pending
 if pending is not None:
  if not self.training or not torch.is_grad_enabled() or self._supcon_phase!='second' or pending['second'] is not None or out['tx_logits'].shape!=(128,6):raise ValueError('Exactly the original scheduled second L/RDrop logits required')
  pending['second']=out['tx_logits'];self._pc_second_captures+=1
 return out
def training_hook(self,**kw):
 step=kw['optimizer_step'];scheduled=active(step)
 if not self.training or not torch.is_grad_enabled() or step!=self._pc_calls+1 or self._pc_pending is not None:raise ValueError('Ordered source L-only gradient coordination required')
 pending=dict(step=step,second=None) if scheduled else None;self._pc_pending=pending;info=dict(blocks=[],conflicted_blocks=0,nonzero_blocks=0,maximum_norm_match_error=0.);eligible=False;counts=[];vjp_calls=0;eta=strength(step) if scheduled else 0.;proxy=None
 try:
  correction,metrics=self._pc_original_hook(**kw);original_identity=kw['identity_loss']+correction
  if scheduled:
   if pending['second'] is None or kw['labeled_count']!=128:raise ValueError('Actual original paired L logits missing')
   first=kw['labeled_output']['tx_logits'];second=pending['second'];parts,counts=contributions(first,second,kw['labels']);eligible=all(n>0 for n in counts)
   if eligible:
    parameters=select(self);ps=[p for _,p in parameters];gradients=[torch.autograd.grad(value,ps,retain_graph=True,create_graph=False,allow_unused=True) for value in (original_identity,*parts)];vjp_calls=3;deltas,info=corrected(parameters,gradients,eta,self._pc_mode);proxy=surrogate(parameters,deltas)
    if float(proxy.detach())!=0.:raise FloatingPointError('Gradient-only surrogate changed scalar L identity objective')
    correction=correction+proxy;self._pc_eligible+=1;self._pc_changed_steps+=int(info['nonzero_blocks']>0);self._pc_vjp_calls+=3
  self._pc_calls=step;self._pc_scheduled+=int(scheduled);self._pc_records.append(dict(step=step,epoch=(step-1)//222+1,scheduled=scheduled,eligible=eligible,strength=eta,weak_counts=counts,feature_VJP_calls=vjp_calls,original_L_identity_loss=float(original_identity.detach()),surrogate_value=0.,**info))
  metrics.update({'F_blockPC/calls':step,'F_blockPC/scheduled':int(scheduled),'F_blockPC/scheduled_calls':self._pc_scheduled,'F_blockPC/eligible_calls':self._pc_eligible,'F_blockPC/changed_steps':self._pc_changed_steps,'F_blockPC/feature_VJP_calls':self._pc_vjp_calls,'F_blockPC/conflicted_blocks':info['conflicted_blocks'],'F_blockPC/nonzero_blocks':info['nonzero_blocks'],'F_blockPC/strength':eta,'F_blockPC/maximum_norm_match_error':info['maximum_norm_match_error'],'F_blockPC/extra_model_forwards':0,'F_blockPC/head_direct_aux_gradient':0})
  return correction,metrics
 finally:self._pc_pending=None
training_hook._accepts_labeled_inputs=False
def attach(m,c):
 if m._supcon_calls or hasattr(m,'_pc_mode'):raise ValueError('Install once before original scratch L training')
 m._pc_mode=c['treatment'];m._pc_pending=None;m._pc_calls=0;m._pc_scheduled=0;m._pc_eligible=0;m._pc_changed_steps=0;m._pc_vjp_calls=0;m._pc_second_captures=0;m._pc_records=[];m._pc_original_identity=m.forward_identity_only;m.forward_identity_only=MethodType(identity_forward,m);m._pc_original_hook=m._source_labeled_feature_hook;m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['weak_TX_block_gradient']=dict(mode=c['treatment'],weak_indices=[1,3],full_deployed_identity_features=True,default_W_excluded=True,default_head_main_joint=True,paired_CE_L_contribution=True,class_weighting='actual sum divided by128, no class upweight',warm_epochs=5,ramp_epochs=10,every_steps=4,first_step=1112,scheduled_calls=2498,extra_feature_VJP_calls_max=7494,extra_model_forwards=0,extra_parameters=0,extra_optimizer_steps=0,surrogate_scalar_value=0.,Sinc_common_unit_scale_invariant=True,control='same perblock corrected Lidentity gradient norm, original Lidentity direction; U/aux gradient sums then original physical clip/AdamW apply',no_AdamW_descent_guarantee=True)
 return m
