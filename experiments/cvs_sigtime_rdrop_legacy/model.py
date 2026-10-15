"""Retain old SIG_TIME construction and normal augmentation; reuse actual L input."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_feature_sequence50 import model as old
classifier=old.classifier;source_domain_count=old.source_domain_count
@contextmanager
def labeled_context(self):
 with self._legacy_original_labeled_scope():
  if not self.training:
   yield;return
  if self._legacy_mode=='ORIGINAL':
   self._legacy_calls+=1
   yield;return
  if self._legacy_actual_L_input is not None:raise RuntimeError('Previous L input not consumed')
  count=[0]
  def capture(module,args):
   if module is not self or not args or args[0].shape!=(128,2,256) or count[0]:raise ValueError('One actual native augmented L128 forward required')
   self._legacy_actual_L_input=args[0].detach().clone();count[0]+=1
  handle=self.register_forward_pre_hook(capture)
  try:yield
  finally:handle.remove()
  if count[0]!=1 or self._legacy_actual_L_input is None:raise ValueError('Actual native augmented L input was not captured')
def terms(first,second,y,alpha,smoothing=.01):
 if alpha not in (0.,1.) or smoothing!=.01 or first.shape!=second.shape or first.shape!=(len(y),6) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not torch.isfinite(first).all() or not torch.isfinite(second).all():raise ValueError('Old smoothed source identity objective required')
 ce1=F.cross_entropy(first,y,label_smoothing=smoothing);ce2=F.cross_entropy(second,y,label_smoothing=smoothing)
 p,q=first.log_softmax(1),second.log_softmax(1);kl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
 return .5*(ce1+ce2)+alpha*kl,.5*(ce1+ce2),kl,ce1,ce2
def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count):
 if self is not model or self._legacy_mode=='ORIGINAL' or not self.training or not torch.is_grad_enabled() or optimizer_step!=self._feature_l_steps or optimizer_step!=self._legacy_calls+1 or labeled_count!=128 or labels.shape!=(128,):raise ValueError('Only ordered real paired L student steps allowed')
 first=labeled_output['tx_logits'];ce1=F.cross_entropy(first,labels,label_smoothing=.01)
 if not torch.allclose(ce1.detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native historical label smoothing/normalization differs')
 self._legacy_calls+=1
 x=self._legacy_actual_L_input;self._legacy_actual_L_input=None
 if x is None or x.shape!=(128,2,256):raise ValueError('Only captured post-augmentation L input may be reused')
 second=self.forward_identity_only(x,domain_labels=domains)['tx_logits']
 total,paired,kl,ce1,ce2=terms(first,second,labels,self._legacy_alpha)
 self._legacy_extra_packets+=128;self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 metrics={'legacy_rdrop/calls':self._legacy_calls,'legacy_rdrop/extra_L_packets':self._legacy_extra_packets,'legacy_rdrop/CE1':ce1.detach(),'legacy_rdrop/CE2':ce2.detach(),'legacy_rdrop/paired_CE':paired.detach(),'legacy_rdrop/symmetric_KL':kl.detach(),'legacy_rdrop/alpha':self._legacy_alpha,'legacy_rdrop/positive_KL_calls':self._legacy_positive_KL_calls,'legacy_rdrop/actual_augmented_input_reused':1}
 return .5*(ce2-identity_loss)+self._legacy_alpha*kl,metrics
hook._accepts_labeled_inputs=False
def attach(model,c):
 model._legacy_mode=c['mode'];model._legacy_alpha=c['training_mechanism']['alpha'];model._legacy_calls=0;model._legacy_extra_packets=0;model._legacy_positive_KL_calls=0;model._legacy_actual_L_input=None
 model._legacy_original_labeled_scope=model._phase1_labeled_feature_context;model._phase1_labeled_feature_context=MethodType(labeled_context,model)
 if c['mode']!='ORIGINAL':model._source_labeled_feature_hook=MethodType(hook,model)
 elif getattr(model,'_source_labeled_feature_hook',None) is not None:raise ValueError('Original arm must retain untouched native CE objective')
 model._feature_learning_report['original_SIGTIME_RDrop']=c['training_mechanism'];model._feature_learning_report['sinc_coordinates']='RAW_HZ'
 if type(model.get_submodule('id_backbone.encoder.core.id_backbone.sinc')).__name__=='CoordinateSinc':raise ValueError('Original SIG_TIME must retain original Sinc coordinates')
 return model
def install(model,c):d.validate(c);return attach(old.install(model,d.legacy_config(c['model_seed'])),c)
def build(c,device):d.validate(c);return attach(old.build(d.legacy_config(c['model_seed']),device),c)
