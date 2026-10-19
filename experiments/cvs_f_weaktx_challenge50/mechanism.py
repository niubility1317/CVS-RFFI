"""Preserve two RDrop/SupCon forwards; extra L-only masked feature objective."""
from collections.abc import Mapping
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
from torch.func import functional_call
from experiments.cvs_f_supcon50.mechanism import schedule,LEVELS,PREFIX
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from .objective import eligible,channel_mask
W='encoder.core.id_backbone.cls_head.weight'
def active(step):return step>1110 and step%4==0
def level_for(step):
 if not active(step):raise ValueError('Outside fixed challenge calendar')
 return LEVELS[((step//4)-278)%2]
@contextmanager
def labeled_scope(self):
 step=self._feature_l_steps+1
 if self._challenge_pending is not None:raise ValueError('Unconsumed L challenge graph')
 with self._challenge_original_scope():
  if active(step) and self.training and torch.is_grad_enabled():
   self._challenge_pending=dict(step=step,level=level_for(step))
  yield

def extra_loss(self,x,y,mask,level):
 if not self.training or not torch.is_grad_enabled() or x.shape!=(len(y),2,256) or mask.shape[0]!=len(y):raise ValueError('Extra challenge uses only captured actual L IQ')
 modes=[(v,v.training) for v in self.id_backbone.modules()]
 parameters=dict(self.id_backbone.named_parameters());buffers={n:v.detach().clone() for n,v in self.id_backbone.named_buffers()}
 if W not in parameters:raise ValueError('Existing default classifier missing')
 parameters[W]=parameters[W].detach();calls=[0]
 def replace(module,args,out):
  if out.shape[:2]!=mask.shape[:2] or out.ndim!=3:raise ValueError('Actual chosen layer shape differs')
  calls[0]+=1;return out*mask
 handle=self.get_submodule(PREFIX+level).register_forward_hook(replace)
 try:
  self.id_backbone.eval()
  # All buffers cloned and caches cleared. Additional eval forward consumes
  # no main/dropout RNG; the two stochastic native RDrop passes remain exact.
  with fresh_parameter_caches(self.id_backbone):
   output=functional_call(self.id_backbone,(parameters,buffers),(x,),dict(return_aux=True),strict=True)
  if calls[0]!=1 or not isinstance(output,Mapping) or output['logits'].shape!=(len(y),6):raise ValueError('One exact identity feature challenge forward required')
  return F.cross_entropy(output['logits'],y,label_smoothing=.01)
 finally:
  handle.remove()
  for module,mode in modes:module.training=mode

def training_hook(self,**kw):
 step=kw['optimizer_step'];pending=self._challenge_pending;aux=None;count=0;level='none';gradient=None
 if active(step):
  if pending is None or pending['step']!=step or self._supcon_pending is None:raise ValueError('Challenge requires actual first L forward')
  y=kw['labels'];selected=eligible(y);count=int(selected.sum());level=pending['level'];feature=self._supcon_pending['first'][level]
  x=self._legacy_actual_L_input
  if x is None or x.shape!=(128,2,256) or feature.shape[0]!=128:raise ValueError('Actual augmented L input/features missing')
  if count:
   if self._challenge_mode=='SALIENCY_WEAK':
    logit=kw['labeled_output']['tx_logits'];target=logit[selected].gather(1,y[selected,None]).sum()
    gradient=torch.autograd.grad(target,feature,retain_graph=True,create_graph=False)[0].detach()[selected]
   mask=channel_mask(feature[selected],gradient,self._challenge_mode,self._challenge_seed+step)
   # Original graph ownership ends below; the input is a detached clone of
   # the same actual augmentation, never raw IQ or U/validation/target.
   x=x[selected];yy=y[selected]
 try:
  correction,metrics=self._challenge_original_hook(**kw)
  if active(step) and count:
   aux=extra_loss(self,x,yy,mask,level);weight=schedule((step-1)//222+1)
   correction=correction+weight*aux;self._challenge_applied+=1;self._challenge_packets+=count
  else:weight=0.
  record=dict(step=step,active=active(step),level=level,packets=count,loss=float(aux.detach()) if aux is not None else 0.,weight=weight,default_head_direct_aux_gradient=False,source_L_only=True,extra_optimizer_updates=0)
  self._challenge_records.append(record)
  metrics.update({'F_challenge/scheduled':int(active(step)),'F_challenge/applied':self._challenge_applied,'F_challenge/extra_L_packets':self._challenge_packets,'F_challenge/packets':count,'F_challenge/loss':record['loss'],'F_challenge/weight':weight,'F_challenge/head_direct_aux_gradient':0,'F_challenge/layer':0 if level=='time_down' else 1 if level=='t2.pool' else -1})
  return correction,metrics
 finally:self._challenge_pending=None
training_hook._accepts_labeled_inputs=False

def attach(m,c):
 if m._supcon_calls or hasattr(m,'_challenge_mode'):raise ValueError('Install challenge once before first L update')
 m._challenge_mode=c['treatment'];m._challenge_seed=c['model_seed']*100000;m._challenge_pending=None;m._challenge_applied=0;m._challenge_packets=0;m._challenge_records=[]
 m._challenge_original_scope=m._phase1_labeled_feature_context;m._challenge_original_hook=m._source_labeled_feature_hook
 m._phase1_labeled_feature_context=MethodType(labeled_scope,m);m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['weak_TX_feature_challenge']=dict(mode=c['treatment'],weak_indices=[1,3],layers=list(LEVELS),drop_fraction=.125,every=4,max_weight=.1,warmup=5,ramp=10,extra_parameters=0,default_head_aux_detached=True,source_L_only=True,ordinary_augmented_IQ_reused=True,original_RDrop_and_SupCon_unchanged=True,extra_forward_mode='eval, cloned buffers and detached existing W')
 return m
