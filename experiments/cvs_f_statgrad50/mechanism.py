"""Only descriptor statistic backward changes; exact native paired L forwards."""
from types import MethodType
from collections.abc import Mapping
import torch
from experiments.cvs_f_supcon50.mechanism import LEVELS,RX_BY_DOMAIN,schedule,clear
from .loss import penalty
def training_hook(self,**kwargs):
 step=kwargs['optimizer_step'];epoch=(step-1)//222+1;weight=schedule(epoch)
 if not self.training or not torch.is_grad_enabled() or step!=self._supcon_calls+1:raise ValueError('Only native ordered L source objective calls permitted')
 active=weight>0;pending=self._supcon_pending
 if active and (pending is None or pending['step']!=step or tuple(pending['first'])!=LEVELS):raise ValueError('Paired SupCon must use the actual native L forward')
 if not active and (pending is not None or self._supcon_handles):raise ValueError('Warmup must leave F path unchanged')
 try:
  if active:self._supcon_phase='second'
  correction,metrics=self._supcon_original_hook(**kwargs)
  self._supcon_phase=None;details=[]
  if active:
   if tuple(pending['second'])!=LEVELS:raise ValueError('Original RDrop second forward missing features')
   y=kwargs['labels'];domains=kwargs['domains']
   if domains is None or domains.shape!=y.shape or domains.dtype!=torch.long or not bool(((domains>=0)&(domains<15)).all()):raise ValueError('Actual source RX/day domain labels required')
   extra=kwargs['extra']
   if not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping) or 'rx_i' not in extra[1]:raise ValueError('Native (domain, metadata) L receiver extra required')
   meta=extra[1]
   if set(meta.get('split_source',[]))!={'ssdg_labeled_tx_visible'} or not bool(torch.as_tensor(meta.get('tx_label_visible',False)).all()) or kwargs['labeled_count']!=128:raise ValueError('Only actual visible L128, no U/V/target')
   rx=torch.as_tensor(meta['rx_i'],device=y.device,dtype=torch.long)
   if rx.shape!=y.shape or not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=domains.device,dtype=torch.long)[domains]):raise ValueError('Physical L RX and original domain map disagree')
   values=[]
   for phase in ('first','second'):
    value,dd=penalty(pending[phase],y,rx,self._statistic_policy);values.append(value);details.extend(dict(stage=phase,**v) for v in dd)
   aux=torch.stack(values).mean();correction=correction+weight*aux;self._supcon_applied+=1;raw=float(aux.detach())
  else:raw=0.
  self._supcon_calls=step;self._supcon_records.append(dict(step=step,epoch=epoch,weight=weight,loss=raw,mode=self._supcon_mode,statistic_policy=self._statistic_policy,details=details,head_fit_updates=0,U_truth_read=False,extra_model_forwards=0))
  metrics.update({'F_supcon/calls':step,'F_supcon/applied_calls':self._supcon_applied,'F_supcon/loss':raw,'F_supcon/weight':weight,'F_supcon/extra_model_forwards':0,'F_supcon/default_head_direct_aux_gradient':0});return correction,metrics
 finally:clear(self)
training_hook._accepts_labeled_inputs=False

def attach(m,c):
 if m._supcon_calls or m._supcon_pending is not None or m._supcon_handles or hasattr(m,'_statistic_policy') or c['statistic_gradient_policy'] not in ('STAT_GRAD','NORM_ONLY'):raise ValueError('Attach one fixed statistic policy before scratch training')
 m._statistic_policy=c['statistic_gradient_policy'];m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['statistic_gradient']=dict(policy=m._statistic_policy,layers=list(LEVELS),temperature=.2,max_weight=.1,warmup=5,ramp=10,positive='sameTX differentRX sourceL',negative='differentTX',class_reweighting=False,extra_model_forwards=0,extra_parameters=0,default_head_direct_aux_gradient=False,forward_exact_native=True,limitation='NORM_ONLY matches descriptor gradient norm at the same features/upstream, not whole-network parameter norm or independent trajectories. No target benefit assumed.')
 return m
