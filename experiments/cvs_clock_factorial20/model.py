"""Original F and losses; only the second generated IQ view changes."""
from collections import Counter
from collections.abc import Mapping
from types import MethodType
import torch
from . import design as d
from .clock import Clock
from experiments.cvs_fullsource_concat20 import model as old,supcon
from experiments.cvs_fullsource_concat20.channel import Practical
from experiments.cvs_sigtime_rdrop_legacy.model import terms
classifier,source_domain_count,export=old.classifier,old.source_domain_count,old.export
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
 raw=labeled_inputs.detach();view=ordinary.clone();observed=[];ag=0
 gmask=(torch.arange(128,device=labels.device)+step).remainder(2)==0;smask=~gmask
 def subset(mask):return {k:v[mask.to(v.device)] if torch.is_tensor(v) and v.ndim>0 and len(v)==128 else v for k,v in extra[1].items()}
 if self._full_config['new_Ag']:
  view[gmask]=self._clock.apply(ordinary[gmask],subset(gmask),epoch,step);ag=64
 if self._full_config['As']:
  satellite,observed=self._full_channel.apply(raw[smask],subset(smask),epoch=epoch,step=step)
  if len(observed)!=64:raise ValueError('Every generated labeled satellite packet must be supervised')
  view[smask]=satellite
  for y,r,o in zip(labels[smask].detach().cpu().tolist(),rx[smask].detach().cpu().tolist(),observed):self._full_coverage[(int(y),int(r),o['scene'],o['state_end'],o['receiver_lock_status'])]+=1
 view_mse=float((view-raw).square().mean());z=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)]
 extra_loss,extra_ce,extra_kl,_,_=terms(*z,labels,1.)
 total=.5*common+.5*extra_loss+weight*aux;nf=4
 self._full_calls+=1;self._full_L_packets+=nf*len(labels);self._supcon_calls=step;self._legacy_calls+=1;self._legacy_extra_packets+=len(labels);self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 record=dict(step=step,epoch=epoch,total_L=float(total.detach()),common_RD=float(common.detach()),common_CE=float(ce.detach()),common_KL=float(kl.detach()),extra_RD=None if extra_loss is None else float(extra_loss.detach()),extra_CE=None if extra_ce is None else float(extra_ce.detach()),extra_KL=None if extra_kl is None else float(extra_kl.detach()),SupCon=float(aux.detach()),SupCon_weight=weight,SupCon_details=details,L_forwards=nf,Ag_packets=ag,As_packets=len(observed),ordinary_raw_MSE=float((ordinary-labeled_inputs).square().mean()),second_raw_MSE=view_mse,same_IQ_each_pair=True,ordinary_redraws=0,scene_counts=dict(Counter(o['scene'] for o in observed)),state_counts=dict(Counter(o['state_end'] for o in observed)),all_satellite_L_supervised=True)
 self._full_records.append(record)
 return total-identity_loss,{'concat/'+k:record[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._full_config=c;m._full_channel=Practical('train') if c['As'] else None;m._clock=Clock() if c['new_Ag'] else None
 m._feature_learning_report.update(clock_factorial=c['factorial'],clock=c['clock'],new_deployment_parameters=0)
 return m

def build(c,device='cpu'):return attach(old.build(d.reference_config(),device),c)
def install(m,c):return attach(old.install(m,d.reference_config()),c)
