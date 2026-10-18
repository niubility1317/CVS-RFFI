"""Unchanged original six L forwards and old U; capture one loss component."""
from pathlib import Path
from types import MethodType
import torch
from . import design as d,core
from .protocol import check_checkpoint,check_source_metadata
from experiments.cvs_noise_shape_factorial43 import model as original
from experiments.cvs_crossview_supcon41 import model as p41
classifier,source_domain_count=original.classifier,original.source_domain_count
terms,capture_forward,supcon=p41.terms,p41.capture_forward,p41.supcon

def hook(self,**kw):
 if self._sa_as_pending is not None:raise ValueError('Previous native update failed to consume As loss')
 step=kw['optimizer_step'];labels=kw['labels'];smask=(torch.arange(128,device=labels.device)+step).remainder(2)!=0;calls=0;captured=[];native_terms=p41.terms
 def scoped_terms(z0,z1,y,alpha):
  nonlocal calls
  calls+=1;result=native_terms(z0,z1,y,alpha)
  if calls==2:
   if y is not labels or z0.shape!=(128,6) or z1.shape!=(128,6) or alpha!=1.:raise ValueError('Original mixed128 terms call differs')
   # Reuse the actual mixed logits; this is scalar arithmetic, never a new NN view.
   value=.25*native_terms(z0[smask],z1[smask],y[smask],alpha)[0] if self._shape_As else result[0]*0.
   captured.append(value)
  return result
 p41.terms=scoped_terms
 try:delta,old_info=original.hook(self,**kw)
 finally:p41.terms=native_terms
 if calls!=3 or len(captured)!=1 or captured[0].ndim or not torch.isfinite(captured[0].detach()):raise ValueError('Exactly original ordinary/mixed/raw RD terms required')
 self._sa_as_pending=(step,captured[0]);r=self._full_records[-1];shape=self._shape_records[-1]
 record=dict(step=step,As_RD=float(captured[0].detach()),As=self._shape_As,native_RD_calls=calls,masked_RD_recomputations=1 if self._shape_As else 0,extra_NN=0,L_forwards=6,U_changed=False)
 self._sa_component_records.append(record)
 info={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
 info.update({'shape/RD':shape['shape_RD'],'split/As_RD':record['As_RD']})
 return delta,info
hook._accepts_labeled_inputs=True

def optimizer_probe(self,*,model,optimizer,epoch,batch_idx,phase,optimizer_step,total_loss,unlabeled_loss,labeled_count,unlabeled_count):
 if model is not self or labeled_count!=128 or self._sa_as_pending is None or self._sa_as_pending[0]!=optimizer_step or not torch.is_grad_enabled() or not self.training:raise ValueError('One current full native L+U loss required')
 component=self._sa_as_pending[1];core.capture(self,total_loss,component,optimizer,optimizer_step);self._sa_as_pending=None
 self._sa_native_records.append(dict(step=optimizer_step,epoch=int(epoch),batch_idx=int(batch_idx),phase=str(phase),total_loss=float(total_loss.detach()),unlabeled_loss=float(unlabeled_loss.detach()),As_RD=float(component.detach()),labeled_count=int(labeled_count),unlabeled_count=int(unlabeled_count),includes_original_U=True,extra_NN=0))
 return None

def original_config(c):
 d.validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 row=next(r for r in d.rows() if r['row_id']==c['row_id']);ref=d.reference_row(row);cfg=b.config(next(r for r in b.rows() if r['row_id']==ref['row_id']),30)
 if cfg!=d.reference_config(row,30):raise ValueError('Actual pinned original recipe differs')
 return cfg
def attach(m,c):
 d.validate(c)
 if getattr(m,'_source_unlabeled_selection_hook',None) is not None or getattr(m,'_source_unlabeled_feature_hook',None) is not None or getattr(m,'_source_optimizer_update_probe_hook',None) is not None:raise ValueError('Original old U/no other optimizer hook required')
 core.initialize(m,c['optimizer_mode']);m._sa_as_pending=None;m._sa_component_records=[];m._sa_native_records=[]
 m._source_labeled_feature_hook=MethodType(hook,m);m._source_optimizer_update_probe_hook=MethodType(optimizer_probe,m)
 m._feature_learning_report['splitAdam']=dict(c['split_adam']);return m
def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root']);check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'])
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
