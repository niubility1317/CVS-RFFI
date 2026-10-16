"""L-only shallow/middle SupCon on the two existing RDrop forwards."""
from contextlib import contextmanager
from collections.abc import Mapping
from types import MethodType
import torch
from torch.nn import functional as F
LEVELS=('time_down','t2.pool')
PREFIX='id_backbone.encoder.core.id_backbone.'
RX_BY_DOMAIN=(1,1,1,3,3,3,4,4,4,6,6,6,8,8,8)
def schedule(epoch):return .1*min(max((int(epoch)-5)/10.,0.),1.)
def masks(y,rx,mode):
 if mode not in ('ALL_TX','CROSS_RX') or y.dtype!=torch.long or rx.dtype!=torch.long or y.ndim!=1 or rx.shape!=y.shape or not bool(((y>=0)&(y<6)).all()) or not all(int(r) in (1,3,4,6,8) for r in rx.detach().cpu().tolist()):raise ValueError('Current source L TX/RX only')
 same=y[:,None]==y[None,:];other=~torch.eye(len(y),dtype=torch.bool,device=y.device)
 positive=same&other if mode=='ALL_TX' else same&(rx[:,None]!=rx[None,:]);negative=(~same)&other;denominator=other if mode=='ALL_TX' else positive|negative;valid=positive.any(1)&negative.any(1)
 return positive,denominator,valid
def loss(z,y,rx,mode):
 if z.ndim!=2 or len(z)!=len(y) or not bool(torch.isfinite(z).all()):raise ValueError('Finite current L descriptors required')
 positive,denominator,valid=masks(y,rx,mode);n=positive.sum(1)
 if not bool(valid.any()):return z.sum()*0.,dict(valid_anchors=0,total_anchors=len(y),positive_min=0,positive_max=0)
 z=F.normalize(z,dim=1,eps=1e-4);logits=(z@z.T/.2)[valid];pos=positive[valid];den=denominator[valid];count=n[valid];value=(torch.logsumexp(logits.masked_fill(~den,-torch.inf),1)-(logits*pos).sum(1)/count-count.to(z.dtype).log()).mean()
 if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite source contrastive loss')
 return value,dict(valid_anchors=int(valid.sum()),total_anchors=len(y),positive_min=int(count.min()),positive_max=int(count.max()))
def penalty(maps,y,rx,mode):
 if tuple(maps)!=LEVELS:raise ValueError('Both actual shallow/middle F maps required')
 values=[];detail=[]
 for level in LEVELS:
  h=maps[level]
  if h.ndim!=3 or len(h)!=len(y) or not bool(torch.isfinite(h).all()):raise ValueError('Finite current L map required')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4);value,stats=loss(z,y,rx,mode);values.append(value);detail.append(dict(level=level,shape=list(h.shape),loss=float(value.detach()),**stats))
 return torch.stack(values).mean(),detail
def clear(m):
 for h in m._supcon_handles:h.remove()
 m._supcon_handles=[];m._supcon_phase=None;m._supcon_pending=None
@contextmanager
def labeled_scope(self):
 if self._supcon_pending is not None or self._supcon_handles:raise ValueError('Previous source-only feature graph was not consumed')
 step=self._feature_l_steps+1;epoch=(step-1)//222+1;active=self.training and torch.is_grad_enabled() and schedule(epoch)>0
 if active:
  self._supcon_pending=dict(step=step,epoch=epoch,first={},second={});self._supcon_phase='first'
  for level in LEVELS:
   def capture(module,args,out,level=level):
    phase=self._supcon_phase
    if not self.training or phase not in ('first','second') or self._supcon_pending is None:raise ValueError('SupCon hook escaped real L-only paired forwards')
    values=self._supcon_pending[phase]
    if level in values:raise ValueError('Layer called more than once in one identity forward')
    values[level]=out
   self._supcon_handles.append(self.get_submodule(PREFIX+level).register_forward_hook(capture))
 try:
  with self._supcon_original_scope():yield
 except BaseException:
  clear(self);raise
 finally:self._supcon_phase=None
 if active and tuple(self._supcon_pending['first'])!=LEVELS:
  clear(self);raise ValueError('First native L identity forward missing feature maps')
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
    value,dd=penalty(pending[phase],y,rx,self._supcon_mode);values.append(value);details.extend(dict(stage=phase,**v) for v in dd)
   aux=torch.stack(values).mean();correction=correction+weight*aux;self._supcon_applied+=1;raw=float(aux.detach())
  else:raw=0.
  self._supcon_calls=step;self._supcon_records.append(dict(step=step,epoch=epoch,weight=weight,loss=raw,mode=self._supcon_mode,details=details,head_fit_updates=0,U_truth_read=False,extra_model_forwards=0))
  metrics.update({'F_supcon/calls':step,'F_supcon/applied_calls':self._supcon_applied,'F_supcon/loss':raw,'F_supcon/weight':weight,'F_supcon/extra_model_forwards':0,'F_supcon/default_head_direct_aux_gradient':0});return correction,metrics
 finally:clear(self)
training_hook._accepts_labeled_inputs=False
def attach(m,c):
 if c['baseline_reuse']:return m
 if hasattr(m,'_supcon_calls') or c['arm'] not in ('ALL_TX','CROSS_RX'):raise ValueError('One fixed paired-SupCon policy per F model')
 m._supcon_mode=c['arm'];m._supcon_calls=0;m._supcon_applied=0;m._supcon_handles=[];m._supcon_pending=None;m._supcon_phase=None;m._supcon_records=[];m._supcon_original_scope=m._phase1_labeled_feature_context;m._supcon_original_hook=m._source_labeled_feature_hook;m._phase1_labeled_feature_context=MethodType(labeled_scope,m);m._source_labeled_feature_hook=MethodType(training_hook,m)
 return m
