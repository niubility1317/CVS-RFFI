"""Preserve complete P41 ORD training;one matched existing CE/RDrop IQ slot."""
from types import MethodType
from pathlib import Path
import torch
from . import design as d,noise
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
 m._shape_offset=c['step_offset'];m._shape_mode=c['Ag_mode'];m._shape_As=c['As'];m._shape_NN=0;m._shape_packets=0;m._shape_records=[];m._source_labeled_feature_hook=MethodType(hook,m);m._feature_learning_report['noise_shape43']=dict(mode=c['Ag_mode'],As=c['As'],noise=c['noise'],total_L_loss_scale=1.25,new_alignment_losses=0,extra_U_NN=0);return m
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
