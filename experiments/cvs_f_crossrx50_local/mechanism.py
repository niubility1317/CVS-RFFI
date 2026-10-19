"""Feature-only treatments on the exact original native L/RDrop graph."""
from contextlib import contextmanager
from types import MethodType
from collections.abc import Mapping
import torch
from experiments.cvs_f_supcon50.mechanism import LEVELS,RX_BY_DOMAIN,schedule,clear
from experiments.cvs_f_txrisk.feature import loss,penalty

@contextmanager
def labeled_scope(self):
 with self._weak_original_scope():
  if self._weak_geometry and self._supcon_pending is not None:
   def capture(module,args,out):
    phase=self._supcon_phase;pending=self._supcon_pending
    if phase not in ('first','second') or pending is None or not isinstance(out,Mapping) or out.get('feat_joint') is None:raise ValueError('Actual native joint160 missing inside original L forward')
    z=out['feat_joint'];key=phase+'_joint'
    if z.shape!=(128,160) or key in pending:raise ValueError('Exactly one joint160 per native L/RDrop phase required')
    pending[key]=z
   self._supcon_handles.append(self.id_backbone.register_forward_hook(capture))
  yield

def training_hook(self,**kw):
 step=kw['optimizer_step'];epoch=(step-1)//222+1;weight=schedule(epoch);pending=self._supcon_pending;active=weight>0
 if not self.training or not torch.is_grad_enabled() or step!=self._supcon_calls+1:raise ValueError('Only ordered actual L steps allowed')
 if active and (pending is None or pending['step']!=step or tuple(pending['first'])!=LEVELS):raise ValueError('Original first L maps missing')
 if not active and (pending is not None or self._supcon_handles):raise ValueError('Warmup must be the original F flow')
 try:
  if active:self._supcon_phase='second'
  correction,metrics=self._supcon_original_hook(**kw)
  self._supcon_phase=None;details=[];raw=joint_raw=0.
  if active:
   if tuple(pending['second'])!=LEVELS:raise ValueError('Original second RDrop maps missing')
   y=kw['labels'];domain=kw['domains'];extra=kw['extra']
   if domain is None or domain.shape!=y.shape or domain.dtype!=torch.long or not ((domain>=0)&(domain<15)).all() or not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping) or 'rx_i' not in extra[1]:raise ValueError('Actual L TX/domain/RX required')
   rx=torch.as_tensor(extra[1]['rx_i'],device=y.device,dtype=torch.long)
   if not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=domain.device)[domain]):raise ValueError('Physical RX/domain map differs')
   values=[];joints=[]
   for phase in ('first','second'):
    value,detail=penalty(pending[phase],y,rx,'CROSS_RX',self._tx_class_weights);values.append(value);details.extend(dict(stage=phase,**v) for v in detail)
    if self._weak_geometry:
     if phase+'_joint' not in pending:raise ValueError('Deployed native joint160 was not captured')
     joint,info=loss(pending[phase+'_joint'],y,rx,'CROSS_RX',self._tx_class_weights);joints.append(joint);details.append(dict(stage=phase,level='joint160',shape=[128,160],loss=float(joint.detach()),**info))
   aux=torch.stack(values).mean();correction=correction+weight*aux;raw=float(aux.detach())
   if joints:
    joint=torch.stack(joints).mean();correction=correction+.5*weight*joint;joint_raw=float(joint.detach())
   self._supcon_applied+=1
  self._supcon_calls=step;self._supcon_records.append(dict(step=step,epoch=epoch,weight=weight,loss=raw,joint160_loss=joint_raw,joint160_weight=.5*weight if self._weak_geometry else 0.,mode='CROSS_RX',details=details,head_fit_updates=0,U_truth_read=False,extra_model_forwards=0))
  metrics.update({'F_supcon/calls':step,'F_supcon/applied_calls':self._supcon_applied,'F_supcon/loss':raw,'F_supcon/weight':weight,'F_weak/joint160_loss':joint_raw,'F_weak/joint160_weight':.5*weight if self._weak_geometry else 0.,'F_weak/default_head_direct_aux_gradient':0})
  return correction,metrics
 finally:clear(self)
training_hook._accepts_labeled_inputs=False

def install(model,c):
 if c['treatment']=='BASE':return model
 if model._supcon_calls or model._supcon_handles or model._supcon_pending is not None or hasattr(model,'_weak_geometry'):raise ValueError('Install feature treatment once before actual training')
 weights=tuple(float(v) for v in c['class_weights'])
 if len(weights)!=6 or abs(sum(weights)-6)>1e-10 or any(not .5<=v<=1.5 for v in weights):raise ValueError('Bounded source-only mean1 weights required')
 model._weak_geometry=c['joint160_geometry'];model._tx_class_weights=weights;model._weak_original_scope=model._phase1_labeled_feature_context
 model._phase1_labeled_feature_context=MethodType(labeled_scope,model);model._source_labeled_feature_hook=MethodType(training_hook,model)
 model._feature_learning_report['weak_TX_feature_treatment']=dict(treatment=c['treatment'],class_weights=list(weights),joint160_geometry=model._weak_geometry,joint160_max_weight=.05,CE_unchanged=True,U_unchanged=True,extra_model_forwards=0,extra_parameters=0,head_direct_aux_gradient=False)
 return model
