"""Original sixL plus two64 label-only CE forwards; no cross-view/teacher target."""
from types import MethodType
from pathlib import Path
from collections import Counter
import time,torch
from torch.nn import functional as F
from . import design as d
from .channel import hq_view
from .protocol import check_checkpoint,check_source_metadata
from experiments.cvs_noise_shape_factorial43 import model as original
classifier,source_domain_count=original.classifier,original.source_domain_count
def hook(self,**kw):
 if kw['model'] is not self or not self.training or not torch.is_grad_enabled() or kw['labeled_count']!=128 or kw['labels'].dtype!=torch.long or kw['labels'].shape!=(128,) or kw['labeled_inputs'].shape!=(128,2,256):raise ValueError('Only visible L128 native hook')
 step=kw['optimizer_step'];channel=self._full_channel;apply=channel.apply;had='apply' in vars(channel);previous=vars(channel).get('apply');captures=[]
 def captured(x,meta,*args,**kwargs):
  value=apply(x,meta,*args,**kwargs)
  if captures or x.shape!=(64,2,256) or len(value[1])!=64:raise ValueError('One original selected64 As or raw slot required')
  captures.append((x.detach().clone(),meta,value[1]));return value
 channel.apply=captured
 try:delta,info=original.hook(self,**kw)
 finally:
  if had:channel.apply=previous
  else:delattr(channel,'apply')
 if len(captures)!=1:raise ValueError('Original selected64 slot absent')
 raw,meta,first=captures[0];start=time.perf_counter();hq=None;observed=[]
 if self._shape_As:hq,observed=hq_view(channel.obj,raw,meta,first)
 physical_seconds=time.perf_counter()-start;view=hq if self._extra_mode=='HQ' else raw
 if view is None:raise ValueError('HQ candidate requires original completeAs')
 mask=(torch.arange(128,device=raw.device)+step).remainder(2)!=0;y=kw['labels'][mask];domains=kw['domains'][mask]
 z0=self.forward_identity_only(view,domain_labels=domains)['tx_logits'];z1=self.forward_identity_only(view,domain_labels=domains)['tx_logits'];ce0=F.cross_entropy(z0,y,label_smoothing=.01);ce1=F.cross_entropy(z1,y,label_smoothing=.01);ce=.5*(ce0+ce1);loss=.25*ce
 old=self._full_records[-1]['total_L'];self._full_records[-1].update(total_L=old+float(loss.detach()),original_total_L=old,L_forwards=8,extra_CE_packets=64,extra_CE_NN=128,extra_CE_weight=.25,extra_CE_kind=self._extra_mode)
 rec=dict(step=step,cumulative_step=4440+step,mode=self._extra_mode,As=self._shape_As,CE0=float(ce0.detach()),CE1=float(ce1.detach()),CE=float(ce.detach()),loss_delta=float(loss.detach()),weight=.25,NN=128,CE_operator_calls=2,physical_snapshots_extra=64 if self._shape_As else 0,HQ_consumed=64 if self._extra_mode=='HQ' else 0,HQ_shadow=64 if self._shape_As and self._extra_mode=='RAW' else 0,original_total_L=old,total_L=old+float(loss.detach()),total_L_nominal_scale=1.5,same_IQ_two_CE=True,extra_KL=0,extra_teacher_NN=0,physical_CPU_wall_seconds=physical_seconds,quality_delta_db_sum=sum(o['quality_delta_db'] for o in observed),quality_records=len(observed),HQ_scene_counts=dict(Counter(o['scene'] for o in observed)),HQ_state_counts=dict(Counter(o['state_end'] for o in observed)),HQ_lock_counts=dict(Counter(o['receiver_lock_status'] for o in observed)),full_state_counts=dict(Counter(o['state_end'] for o in first)),full_lock_counts=dict(Counter(o['receiver_lock_status'] for o in first)),all_extra_labels_supervised=True,no_cross_view_consistency=True)
 self._extraCE_records.append(rec);self._extra_NN+=128
 if getattr(self,'_extra_smoke_debug',False):self._extra_debug=dict(raw=raw,view=view,HQ=hq,HQ_records=observed,first_records=first,mask=mask,CE=ce,loss=loss,logits=(z0,z1))
 info.update({'concat/total_L':rec['total_L'],'concat/L_forwards':8,'extraCE/CE':rec['CE'],'extraCE/loss':rec['loss_delta'],'extraCE/NN':128})
 return delta+loss,info
hook._accepts_labeled_inputs=True
def original_config(c):
 d.validate(c);return d.reference_config(next(r for r in d.rows() if r['row_id']==c['row_id']),30)
def attach(m,c):
 d.validate(c);m._extra_mode=c['extra_view'];m._extra_NN=0;m._extraCE_records=[];m._source_labeled_feature_hook=MethodType(hook,m);m._feature_learning_report['signal_bridge']=dict(c['signal_bridge']);return m
def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 from .source import parent_payload
 d.validate(c);parent_payload(c);out=Path(c['output_root']);check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids']);cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth');m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
