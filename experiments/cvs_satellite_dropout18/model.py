"""Preserve complete P41 ORD training;one matched existing CE/RDrop IQ slot."""
from types import MethodType
from pathlib import Path
import torch
from . import design as d,noise,dropout
from experiments.cvs_crossview_supcon41 import model as base
classifier,source_domain_count=base.classifier,base.source_domain_count
class NoAs:
 packets=0
 def __init__(self):self.returned=0;self.calls=0
 def apply(self,x,meta,epoch,step):
  self.returned+=len(x);self.calls+=1
  return x.clone(),[dict(scene='NONE',state_end='NOT_APPLICABLE',receiver_lock_status='NOT_APPLICABLE',channel_generated=False) for _ in x]
 def audit(self):return dict(enabled=False,actual_channel_calls=0,reason='Fixed As-off factorial',formal_channel_unchanged=True)
 def summary(self):return dict(enabled=False,actual_channel_calls=0,packets=0,identity_control_packets=self.returned)
def hook(self,**kw):
 delta,info=base.hook(self,**kw);step=kw['optimizer_step']+self._shape_offset;labels=kw['labels'];raw=kw['labeled_inputs'].detach();view,meta=noise.apply(raw,step,self._shape_mode)
 z0=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits'];z1=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits'];rd,ce,kl,_,_=base.terms(z0,z1,labels,1.)
 self._shape_NN+=256;self._shape_packets+=meta['applied'];self._shape_records.append(dict(step=kw['optimizer_step'],cumulative_step=step,shape_RD=float(rd.detach()),shape_CE=float(ce.detach()),shape_KL=float(kl.detach()),weight=.25,NN=256,same_IQ_each_pair=True,**meta))
 record=self._full_records[-1];record['base_L_before_shape']=record['total_L'];record['total_L']+=.25*float(rd.detach());record['L_forwards']=6;record['Ag_packets']=meta['applied'];record['extra_control_packets']=128;record['As_packets']=64 if self._shape_As else 0;record['Ag_mode']=self._shape_mode;record['same_IQ_each_pair']=True
 info['shape/RD']=float(rd.detach());return delta+.25*rd,info
hook._accepts_labeled_inputs=True
def attach(m,c):
 d.validate(c);bc=base.d.config(base.d.rows()[0],c['segment']);m=base.attach(m,bc)
 if not c['As']:m._full_channel=NoAs()
 m._shape_offset=c['step_offset'];m._shape_mode=c['Ag_mode'];m._shape_As=c['As'];m._shape_NN=0;m._shape_packets=0;m._shape_records=[];dropout.install(m,c['satellite_mode']);m._sat_steps=[];m._source_labeled_feature_hook=MethodType(satellite_hook,m);m._feature_learning_report['noise_shape43']=dict(mode=c['Ag_mode'],As=c['As'],noise=c['noise'],total_L_loss_scale=1.25,new_alignment_losses=0,extra_U_NN=0);return m
def install(m,c):return attach(base.old.install(m,base.old_config()),c)
def build(c,device='cpu'):return attach(base.old.build(base.old_config(),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 if c['segment']!=30:raise ValueError('Only own cumulative FINAL30')
 from .source import parent_payload
 parent_payload(c)
 p=Path(c['output_root']);done=d.read(p/'completion.json');frozen=d.read(p/'source_frozen.json');init=d.read(p/'initialization.json')
 if done['config']!=c or done['epoch']!=10 or done['optimizer_steps']!=2220 or done['target_access'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['checkpoint_sources']!=c['checkpoint_sources'] or d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Own exactsource final provenance differs')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=10 or cp['a1_ema_successful_updates']!=2220 or cp['run_id']!=d.RUN or cp['candidate_id']!=c['row_id'] or cp['args']['baseline_ckpt']!=c['parent_source']+'/final_ssdg.pth' or cp['args']['teacher_ckpt']:raise ValueError('Own source final identity differs')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)

# OldAsKLremoval is only an attribution control;new question is satellite featuredropout.
def satellite_hook(self,**kw):
 engine=self._sat_engine;step=kw['optimizer_step'];labels=kw['labels'];mask=(torch.arange(128,device=labels.device)+step).remainder(2)!=0
 original_forward=self.forward_identity_only;original_terms=base.terms;term_calls=0;diagnostic={};first_sat=None
 def observed(m,x,*args,**kwargs):
  nonlocal first_sat
  engine.frame+=1;output=original_forward(x,*args,**kwargs)
  if engine.frame==2:first_sat=output['tx_logits'][mask].detach().clone()
  if engine.frame==3:
   diagnostic['satellite_pair_logit_maxdiff']=float((output['tx_logits'][mask].detach()-first_sat).abs().max())
  return output
 def scoped_terms(z0,z1,y,alpha):
  nonlocal term_calls
  term_calls+=1;value=original_terms(z0,z1,y,alpha)
  if term_calls!=2:return value
  rd,ce,kl,c0,c1=value;ordinary_kl=original_terms(z0[~mask],z1[~mask],y[~mask],alpha)[2];sat_kl=original_terms(z0[mask],z1[mask],y[mask],alpha)[2];used=ce+.5*ordinary_kl
  diagnostic.update(original_mixed_KL=float(kl.detach()),ordinary_half_KL=float(ordinary_kl.detach()),satellite_half_KL=float(sat_kl.detach()),used_mixed_KL=float((.5*ordinary_kl).detach()),removed_satellite_KL=float((.5*sat_kl).detach()))
  return used,ce,.5*ordinary_kl,c0,c1
 try:
  with engine.scope(mask):
   self.forward_identity_only=MethodType(observed,self);base.terms=scoped_terms
   value,info=hook(self,**kw)
   if engine.frame!=5 or term_calls!=3 or engine.calls==0:raise ValueError('Exact original fiveadditionalLforwards/threeRDropterms/dropout scope differs')
   if engine.mode=='AS_DET' and diagnostic['satellite_pair_logit_maxdiff']>1e-5:raise ValueError('Uncontrolled satellite stochastic operation')
   record=dict(step=step,mode=engine.mode,dropout_calls=engine.calls,bypassed_elements=engine.bypassed_elements,As_supervised=64,extra_NN=0,parameter_updates_extra=0,**diagnostic);self._sat_steps.append(record);self._full_records[-1]['satellite_subnetwork']=record
   return value,info
 finally:self.forward_identity_only=original_forward;base.terms=original_terms
satellite_hook._accepts_labeled_inputs=True
