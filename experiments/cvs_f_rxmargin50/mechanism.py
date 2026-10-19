"""Add margin only to actual two L/RDrop deployed features; keep F SupCon."""
from contextlib import contextmanager
from collections.abc import Mapping
from types import MethodType
import torch
from experiments.cvs_f_supcon50.mechanism import schedule,RX_BY_DOMAIN
from .loss import angular_margin

@contextmanager
def labeled_scope(self):
 with self._margin_original_scope():
  if self._supcon_pending is not None:
   def capture(module,args,out):
    phase=self._supcon_phase;pending=self._supcon_pending
    if phase not in ('first','second') or pending is None or not isinstance(out,Mapping) or out.get('feat_joint') is None:raise ValueError('Actual native L/RDrop joint160 missing')
    key=phase+'_joint';z=out['feat_joint']
    if z.shape!=(128,160) or key in pending:raise ValueError('Exactly one deployed feature per existing L/RDrop forward required')
    pending[key]=z
   self._supcon_handles.append(self.id_backbone.register_forward_hook(capture))
  yield

def training_hook(self,**kw):
 step=kw['optimizer_step'];epoch=(step-1)//222+1;weight=schedule(epoch);pending=self._supcon_pending
 # Original hook executes the unchanged RDrop and shallow/middle SupCon,
 # then removes all temporary hooks. Local pending retains the two graphs.
 correction,metrics=self._margin_original_hook(**kw);raw=0.
 if weight>0:
  y=kw['labels'];domain=kw['domains'];extra=kw['extra']
  if domain is None or domain.shape!=y.shape or domain.dtype!=torch.long or not ((domain>=0)&(domain<15)).all() or not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping) or 'rx_i' not in extra[1]:raise ValueError('Actual source L TX/RX/domain required')
  rx=torch.as_tensor(extra[1]['rx_i'],device=y.device,dtype=torch.long)
  if not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=domain.device)[domain]):raise ValueError('Physical RX differs from native domain')
  values=[]
  for phase in ('first','second'):
   if pending is None or phase+'_joint' not in pending:raise ValueError('Original L/RDrop feature missing')
   value,detail=angular_margin(pending[phase+'_joint'],y,rx);values.append(value);self._supcon_records[-1]['details'].append(dict(stage=phase,level='joint160_RX_MARGIN',shape=[128,160],loss=float(value.detach()),**detail))
  aux=torch.stack(values).mean();correction=correction+weight*aux;raw=float(aux.detach())
 self._supcon_records[-1].update(joint160_loss=raw,joint160_weight=weight,mechanism='RX_MATCHED_ANGULAR_MARGIN')
 metrics.update({'F_weak/joint160_loss':raw,'F_weak/joint160_weight':weight,'F_RXmargin/loss':raw,'F_RXmargin/weight':weight,'F_RXmargin/head_direct_aux_gradient':0})
 return correction,metrics
training_hook._accepts_labeled_inputs=False

def attach(m,c):
 if m._supcon_calls or m._supcon_pending is not None or m._supcon_handles or hasattr(m,'_margin_original_scope'):raise ValueError('Install margin once before first L update')
 m._margin_original_scope=m._phase1_labeled_feature_context;m._margin_original_hook=m._source_labeled_feature_hook
 m._phase1_labeled_feature_context=MethodType(labeled_scope,m);m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['RX_matched_angular_margin']=dict(feature='actual joint160',margin=.05,temperature=.05,max_weight=.1,same_IQ_two_existing_RDrop_forwards=True,positive='sameTX differentRX',negative='differentTX sameRX',class_balanced=True,extra_parameters=0,extra_model_forwards=0,direct_head_aux_gradient=False,CE_and_original_SupCon_unchanged=True)
 return m
