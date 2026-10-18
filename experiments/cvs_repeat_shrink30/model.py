"""Fixed F/default cosine/SupCon/U;matched6 labeledforwards,one raw anchor."""
from collections.abc import Mapping
from types import MethodType
import torch
from . import design as d
from .augmentation import Contraction
from experiments.cvs_clock_factorial30 import model as previous
from experiments.cvs_fullsource_concat20 import supcon
from experiments.cvs_sigtime_rdrop_legacy.model import terms
classifier,source_domain_count,export=previous.classifier,previous.source_domain_count,previous.export
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
 raw=labeled_inputs.detach();view,ag_record=self._repeat_ag.apply(raw,epoch+20,step+4440)
 a=[self.forward_identity_only(raw,domain_labels=domains)['tx_logits'] for _ in range(2)]
 anchor,anchor_ce,anchor_kl,_,_=terms(*a,labels,1.)
 z=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)]
 slot,slot_ce,slot_kl,_,_=terms(*z,labels,1.)
 extra_loss=.5*anchor+.5*slot;extra_ce=.5*anchor_ce+.5*slot_ce;extra_kl=.5*anchor_kl+.5*slot_kl
 total=.5*common+.25*anchor+.25*slot+weight*aux;nf=6
 self._full_calls+=1;self._full_L_packets+=nf*len(labels);self._supcon_calls=step;self._legacy_calls+=1;self._legacy_extra_packets+=len(labels);self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 record=dict(step=step,epoch=epoch,total_L=float(total.detach()),common_RD=float(common.detach()),common_CE=float(ce.detach()),common_KL=float(kl.detach()),extra_RD=float(extra_loss.detach()),extra_CE=float(extra_ce.detach()),extra_KL=float(extra_kl.detach()),anchor_RD=float(anchor.detach()),anchor_CE=float(anchor_ce.detach()),anchor_KL=float(anchor_kl.detach()),slot_RD=float(slot.detach()),slot_CE=float(slot_ce.detach()),slot_KL=float(slot_kl.detach()),SupCon=float(aux.detach()),SupCon_weight=weight,SupCon_details=details,L_forwards=nf,Ag_packets=128 if self._full_config['new_Ag'] else 0,As_packets=0,ordinary_raw_MSE=float((ordinary-raw).square().mean()),second_raw_MSE=float((view-raw).square().mean()),same_IQ_each_pair=True,ordinary_redraws=0,scene_counts={},state_counts={},all_satellite_L_supervised=True,supervised_view_weights=[.5,.25,.25],total_L_scale=1.,operator=ag_record)
 self._full_records.append(record)
 return total-identity_loss,{'concat/'+k:record[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._full_config=c;m._full_channel=None;m._repeat_ag=Contraction(c['new_Ag']);m._clock=m._repeat_ag if c['new_Ag'] else None
 m._feature_learning_report.update(IQ_operator=c['IQ_operator'],supervised_views=c['supervised_views'],E30_amendment=dict(parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,optimizer='resetcos10',EMA='sameNoAsparent',no_new_parameters=True));return m

def install(m,c):return attach(previous.install(m,d.model_config(c)),c)
def build(c,device='cpu'):return attach(previous.build(d.model_config(c),device),c)
