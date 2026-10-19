"""Reuse exact six student forwards, transferring the same full As IQ only."""
from types import MethodType
from pathlib import Path
import torch
from . import design as d
from .loss import correction
from .protocol import check_checkpoint,check_source_metadata
from experiments.cvs_noise_shape_factorial43 import model as original
classifier,source_domain_count=original.classifier,original.source_domain_count

def hook(self,**kw):
 old=self.forward_identity_only;had='forward_identity_only' in vars(self);prior=vars(self).get('forward_identity_only');captured=[]
 def forward(x,*args,**kwargs):
  z=old(x,*args,**kwargs)
  captured.append((x.detach().clone(),z['tx_logits']))
  return z
 self.forward_identity_only=forward
 try:delta,info=original.hook(self,**kw)
 finally:
  if had:self.forward_identity_only=prior
  else:delattr(self,'forward_identity_only')
 if len(captured)!=5 or any(x.shape!=(128,2,256) or z.shape!=(128,6) for x,z in captured):raise ValueError('Exact ordinary/mixed/mixed/raw/raw native forwards required')
 mask=(torch.arange(128,device=kw['labels'].device)+kw['optimizer_step']).remainder(2)!=0
 x0,z0=captured[1];x1,z1=captured[2]
 if not torch.equal(x0,x1):raise ValueError('Same mixed IQ must feed both original student passes')
 if self.__dict__['_class_teacher'] is None:
  from .teacher import load
  self.__dict__['_class_teacher']=load(x0.device)
  self.__dict__['_class_teacher_initial']={k:v.detach().cpu().clone() for k,v in self.__dict__['_class_teacher'].state_dict().items()}
 teacher=self.__dict__['_class_teacher']
 if teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):raise ValueError('Class teacher must stay frozen')
 with torch.no_grad():target=teacher(x0[mask])['tx_logits']
 loss,rec=correction(z0[mask],z1[mask],kw['labels'][mask],target,self._distill_mode)
 rec.update(step=kw['optimizer_step'],mode=self._distill_mode,same_actual_IQ=True,all_As_CE=True,teacher_shadow=self._distill_mode=='CE')
 self._distill_records.append(rec);self._distill_NN+=64
 record=self._full_records[-1];record['total_L']+=rec['loss_delta'];record['extra_KL']-=.5*rec['As_KL_removed'];record['extra_RD']-=.5*rec['As_KL_removed']
 info['concat/total_L']=record['total_L'];info.update({'sameIQ/'+k:v for k,v in rec.items() if isinstance(v,(float,int))})
 return delta+loss,info
hook._accepts_labeled_inputs=True

def original_config(c):d.validate(c);return d.reference_config(next(r for r in d.rows() if r['row_id']==c['row_id']),30)
def attach(m,c):
 d.validate(c);m._distill_mode=c['distill_mode'];m.__dict__['_class_teacher']=None;m._distill_NN=0;m._distill_records=[];m._source_labeled_feature_hook=MethodType(hook,m);m._feature_learning_report['sameIQ_teacher']=c['sameIQ_teacher'];return m
def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 from .source import parent_payload
 d.validate(c);parent_payload(c);out=Path(c['output_root']);check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids']);cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth');m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
