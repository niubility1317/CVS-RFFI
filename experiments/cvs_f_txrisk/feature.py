"""Source-calibrated per-TX feature loss; original CE/RDrop/defaulthead untouched."""
from types import MethodType
from collections.abc import Mapping
import torch
from torch.nn import functional as F
from experiments.cvs_f_supcon50.mechanism import LEVELS,RX_BY_DOMAIN,schedule,masks,clear
def loss(z,y,rx,mode,class_weights):
 w=torch.as_tensor(class_weights,device=z.device,dtype=z.dtype).detach()
 if w.shape!=(6,) or not bool(torch.isfinite(w).all()) or not bool(((w>=.5)&(w<=1.5)).all()) or abs(float(w.sum())-6)>1e-5:raise ValueError('Bounded mean-one source-only six-class risk weights required')
 if z.ndim!=2 or len(z)!=len(y) or not bool(torch.isfinite(z).all()):raise ValueError('Finite current L descriptors required')
 positive,denominator,valid=masks(y,rx,mode);n=positive.sum(1)
 if not bool(valid.any()):return z.sum()*0.,dict(valid_anchors=0,total_anchors=len(y),positive_min=0,positive_max=0)
 z=F.normalize(z,dim=1,eps=1e-4);logits=(z@z.T/.2)[valid];pos=positive[valid];den=denominator[valid];count=n[valid];anchor=(torch.logsumexp(logits.masked_fill(~den,-torch.inf),1)-(logits*pos).sum(1)/count-count.to(z.dtype).log());aw=w[y[valid]];value=anchor.mean() if bool((w==1).all()) else (anchor*aw).sum()/aw.sum()
 if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite source contrastive loss')
 return value,dict(valid_anchors=int(valid.sum()),total_anchors=len(y),positive_min=int(count.min()),positive_max=int(count.max()))
def penalty(maps,y,rx,mode,class_weights):
 if tuple(maps)!=LEVELS:raise ValueError('Both actual shallow/middle F maps required')
 values=[];detail=[]
 for level in LEVELS:
  h=maps[level]
  if h.ndim!=3 or len(h)!=len(y) or not bool(torch.isfinite(h).all()):raise ValueError('Finite current L map required')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4);value,stats=loss(z,y,rx,mode,class_weights);values.append(value);detail.append(dict(level=level,shape=list(h.shape),loss=float(value.detach()),**stats))
 return torch.stack(values).mean(),detail
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
   rx=torch.as_tensor(extra[1]['rx_i'],device=y.device,dtype=torch.long)
   if rx.shape!=y.shape or not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=domains.device,dtype=torch.long)[domains]):raise ValueError('Physical L RX and original domain map disagree')
   values=[]
   for phase in ('first','second'):
    value,dd=penalty(pending[phase],y,rx,self._supcon_mode,self._tx_class_weights);values.append(value);details.extend(dict(stage=phase,**v) for v in dd)
   aux=torch.stack(values).mean();correction=correction+weight*aux;self._supcon_applied+=1;raw=float(aux.detach())
  else:raw=0.
  self._supcon_calls=step;self._supcon_records.append(dict(step=step,epoch=epoch,weight=weight,loss=raw,mode=self._supcon_mode,details=details,head_fit_updates=0,U_truth_read=False,extra_model_forwards=0))
  metrics.update({'F_supcon/calls':step,'F_supcon/applied_calls':self._supcon_applied,'F_supcon/loss':raw,'F_supcon/weight':weight,'F_supcon/extra_model_forwards':0,'F_supcon/default_head_direct_aux_gradient':0});return correction,metrics
 finally:clear(self)
training_hook._accepts_labeled_inputs=False

def install(m,class_weights):
 weights=tuple(float(v) for v in class_weights)
 if len(weights)!=6 or any(not .5<=v<=1.5 for v in weights) or abs(sum(weights)-6)>1e-10:raise ValueError('Calibrated source class weights required')
 if m._supcon_calls or m._supcon_pending is not None or m._supcon_handles:raise ValueError('Install once before the first real L optimizer step')
 if hasattr(m,'_tx_class_weights'):raise ValueError('Do not stack class-risk weighting')
 m._tx_class_weights=weights;m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['source_TXrisk']=dict(weights=weights,applied_to='CROSS_RX shallow/middle feature loss only',CE_unchanged=True,head_direct_aux_gradient=False,extra_inference_parameters=0)
 return m
