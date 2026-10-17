"""Same fixed F/defaulthead/ordinaryAs;only preclip supervised-view organization changes."""
from types import MethodType
from contextlib import contextmanager
from collections import Counter
from collections.abc import Mapping
import torch
from . import design as d
from .core import capture,rng_get
from experiments.cvs_clock_factorial30 import model as original
from experiments.cvs_fullsource_concat20 import supcon
from experiments.cvs_sigtime_rdrop_legacy.model import terms
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export
def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
 if model is not self or not self.training or not torch.is_grad_enabled() or labeled_count!=128 or labels.shape!=(128,) or labeled_inputs.shape!=(128,2,256) or optimizer_step!=self._feature_l_steps or optimizer_step!=self._full_calls+1:raise ValueError('One ordered real L128 update required')
 if not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping):raise ValueError('Native L metadata required')
 rx=torch.as_tensor(extra[1]['rx_i'],device=labels.device,dtype=torch.long)
 if domains is None or domains.shape!=labels.shape or not bool(((domains>=0)&(domains<15)).all()) or not torch.equal(rx,torch.tensor(supcon.RX_BY_DOMAIN,device=domains.device)[domains]):raise ValueError('All original source RX/day mapping required')
 step=optimizer_step;epoch=(step-1)//222+1;weight=supcon.schedule(epoch);pending=self._supcon_pending
 ordinary=self._legacy_actual_L_input;self._legacy_actual_L_input=None
 if ordinary is None or ordinary.shape!=(128,2,256):raise ValueError('Generated ordinary IQ must be reused')
 first=labeled_output['tx_logits']
 if not torch.allclose(torch.nn.functional.cross_entropy(first,labels,label_smoothing=.01).detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native CE normalization differs')
 try:
  if weight>0:
   if pending is None or tuple(pending['first'])!=supcon.LEVELS:raise ValueError('Actual first pair feature maps required')
   self._supcon_phase='second'
  second=self.forward_identity_only(ordinary,domain_labels=domains)['tx_logits'];self._supcon_phase=None
  common,ce,kl,_,_=terms(first,second,labels,1.);details=[]
  if weight>0:
   if tuple(pending['second'])!=supcon.LEVELS:raise ValueError('Second pair feature maps required')
   auxs=[]
   for phase in ('first','second'):
    value,records=supcon.penalty(pending[phase],labels,rx,'CROSS_RX');auxs.append(value);details.extend(dict(phase=phase,**v) for v in records)
   aux=torch.stack(auxs).mean();self._supcon_applied+=1
  else:aux=common*0.
 finally:supcon.clear(self)
 arm=self._full_config['arm'];observed=[];extra_ce=extra_kl=extra_loss=None
 if arm=='NATIVE':total=common+weight*aux;nf=2;view_mse=0.
 else:
  raw=labeled_inputs.detach();view=ordinary
  if arm=='CONCAT':
   view,observed=self._full_channel.apply(raw,extra[1],epoch=epoch,step=step)
   if len(observed)!=len(raw):raise ValueError('All satellite L observations must be supervised')
   for y,r,o in zip(labels.detach().cpu().tolist(),rx.detach().cpu().tolist(),observed):self._full_coverage[(int(y),int(r),o['scene'],o['state_end'],o['receiver_lock_status'])]+=1
  view_mse=float((view-raw).square().mean())
  z=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)]
  extra_loss,extra_ce,extra_kl,_,_=terms(*z,labels,1.)
  total=.5*common+.5*extra_loss+weight*aux;nf=4
 if len(self._sam_draws)!=4 or any(not torch.equal(self._sam_draws[i][0],ordinary if i<2 else view) for i in range(4)):raise ValueError('Actual ordinary/secondary IQ replay differs')
 capture(self,total,labels,rx,domains,weight,step)
 self._full_calls+=1;self._full_L_packets+=nf*len(labels);self._supcon_calls=step;self._legacy_calls+=1;self._legacy_extra_packets+=len(labels);self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 record=dict(step=step,epoch=epoch,total_L=float(total.detach()),common_RD=float(common.detach()),common_CE=float(ce.detach()),common_KL=float(kl.detach()),extra_RD=None if extra_loss is None else float(extra_loss.detach()),extra_CE=None if extra_ce is None else float(extra_ce.detach()),extra_KL=None if extra_kl is None else float(extra_kl.detach()),SupCon=float(aux.detach()),SupCon_weight=weight,SupCon_details=details,L_forwards=nf,As_packets=len(observed),ordinary_raw_MSE=float((ordinary-labeled_inputs).square().mean()),second_raw_MSE=view_mse,same_IQ_each_pair=True,ordinary_redraws=0,scene_counts=dict(Counter(o['scene'] for o in observed)),state_counts=dict(Counter(o['state_end'] for o in observed)),all_satellite_L_supervised=True)
 self._full_records.append(record)
 return total-identity_loss,{'concat/'+k:record[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
hook._accepts_labeled_inputs=True

def features(e,x):
 owner=e._sam_owner
 if owner.training and owner._sam_recording:
  if len(owner._sam_draws)>=4:raise ValueError('Only actual4 L forwards may be captured')
  owner._sam_draws.append((x.detach().clone(),rng_get(x)))
 return e._sam_original_features(x)
@contextmanager
def labeled_context(self):
 if self.training and torch.is_grad_enabled():
  if self._sam_pending is not None or self._sam_recording or self._sam_draws:raise ValueError('Previous L replay not consumed')
  self._sam_recording=True
 with self._sam_original_scope():yield
def attach(m,c):
 m._sam_mode=c['update_mode'];m._sam_radius=c['update_policy']['radius'];m._sam_pending=None;m._sam_recording=False;m._sam_draws=[];m._sam_steps=0;m._sam_vjp_calls=0;m._sam_replay_packets=0;m._sam_records=[]
 m._source_labeled_feature_hook=MethodType(hook,m);m._sam_original_scope=m._phase1_labeled_feature_context;m._phase1_labeled_feature_context=MethodType(labeled_context,m)
 e=m.id_backbone.encoder;object.__setattr__(e,'_sam_owner',m);e._sam_original_features=e.features;e.features=MethodType(features,e)
 m._feature_learning_report['L_SAM_policy']=c['update_policy'];return m
def build(c,device='cpu'):
 d.validate(c);return attach(original.build(d.nominal_config(c['arm']),device),c)
def install(m,c):
 d.validate(c);return attach(original.install(m,d.nominal_config(c['arm'])),c)
