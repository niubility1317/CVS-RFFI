"""Preserve original six forward slots; replace only As CE and remove only As KL."""
from types import MethodType
from pathlib import Path
import torch
import torch.nn.functional as F
from . import design as d
from experiments.cvs_as_temperature84.loss import mixed
from experiments.cvs_noise_shape_factorial43 import model as original
from experiments.cvs_crossview_supcon41 import model as base
classifier,source_domain_count=original.classifier,original.source_domain_count
def hook(self,**kw):
 original_terms=base.terms;calls=0;step=kw['optimizer_step'];mask=(torch.arange(128,device=kw['labels'].device)+step).remainder(2)!=0
 def terms(a,b,y,alpha):
  nonlocal calls
  calls+=1
  if calls in (1,3):return original_terms(a,b,y,alpha)
  if calls!=2 or alpha!=1.:raise ValueError('Exact ordinary then mixed loss order required')
  value=mixed(a,b,y,mask,self._temperature,original_terms)
  with torch.no_grad():
   native=.5*(F.cross_entropy(a[mask],y[mask],label_smoothing=.01)+F.cross_entropy(b[mask],y[mask],label_smoothing=.01))
   probabilities=.5*((a[mask]/self._temperature).softmax(1)+(b[mask]/self._temperature).softmax(1))
   target=F.one_hot(y[mask],6).to(a)*.99+.01/6
   gradient=(probabilities-target)*30
   self._temperature_records.append(dict(step=step,T=self._temperature,As_CE_compensated=float(2*value[1]-(original_terms(a[~mask],b[~mask],y[~mask],1.)[1])),As_native_NLL=float(native),As_mean_logit_gradient_norm=float(gradient.norm(dim=1).mean()),As_packets=64,As_NN=128,As_KL=0.,ordinary_KL_retained=True,all_As_labels_supervised=True))
  return value
 base.terms=terms
 try:return original.hook(self,**kw)
 finally:
  base.terms=original_terms
  if calls!=3:raise ValueError('Temperature did not cover original mixed and raw pairs')
hook._accepts_labeled_inputs=True
def attach(m,c):
 from .channel import Practical
 d.validate(c);m=original.attach(m,d.reference_config(c['segment']));m._full_channel=Practical(arm=c['arm']);m._shape_offset=c['step_offset'];m._temperature=c['temperature']['T'];m._temperature_records=[];m._source_labeled_feature_hook=MethodType(hook,m);m._feature_learning_report['temperature84']=c['temperature'];return m
def install(m,c):return attach(original.base.old.install(m,original.base.old_config()),c)
def build(c,device='cpu'):return attach(original.base.old.build(original.base.old_config(),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 if c['segment']==30:parent_payload(c)
 p=Path(c['output_root']);done=d.read(p/'completion.json');frozen=d.read(p/'source_frozen.json');init=d.read(p/'initialization.json')
 if done['config']!=c or done['epoch']!=c['epochs'] or done['optimizer_steps']!=c['updates'] or done['target_access'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['checkpoint_sources']!=c['checkpoint_sources'] or d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Own exactsource final provenance differs')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=c['epochs'] or cp['a1_ema_successful_updates']!=c['updates'] or cp['run_id']!=c['run_id'] or cp['candidate_id']!=c['row_id'] or cp['args']['baseline_ckpt']!=('' if c['segment']==20 else c['parent_source']+'/final_ssdg.pth') or cp['args']['teacher_ckpt']:raise ValueError('Own source final identity differs')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)


