"""Capture one original augmentation call, preserving native draws and outputs."""
import torch
from . import design as d

def paired(aug,x,y):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or y.shape!=(len(x),) or bool(((y<0)|(y>5)).any()):raise ValueError('Finite visible L IQ256/labels required')
 if aug.p_dac!=0 or aug.enable_class_signature or aug.p_pa<=0:raise ValueError('Actual ordinary PA enabled,DAC/class signature disabled required')
 saved={name:getattr(aug,name) for name in ('_apply_general_channel','simulate_pa',*d.SUBCHAINS)};trace=dict(prefix=None,selected=None,strengths=None,subchains=[])
 def prefix(*args,**kwargs):
  value=saved['_apply_general_channel'](*args,**kwargs)
  if trace['prefix'] is not None:raise ValueError('Exactly one native general-channel call required')
  trace['prefix']=value.detach().clone();return value
 def pa(*args,**kwargs):
  if trace['selected'] is not None:raise ValueError('At most one normal PA subset call required')
  trace['selected']=args[0].detach().clone();value,strength=saved['simulate_pa'](*args,**kwargs);trace['strengths']=strength.detach().clone();return value,strength
 def subchain(name):
  def call(*args,**kwargs):trace['subchains'].append(name);return saved[name](*args,**kwargs)
  return call
 aug._apply_general_channel=prefix;aug.simulate_pa=pa
 for name in d.SUBCHAINS:setattr(aug,name,subchain(name))
 try:full,strength=aug(x,labels=y,no_pa=False,return_pa_strength=True)
 finally:
  for name,value in saved.items():setattr(aug,name,value)
 bypass=trace['prefix']
 if bypass is None or not bool(torch.isfinite(full).all()) or not bool(torch.isfinite(bypass).all()):raise ValueError('Finite captured prefix/full required')
 triggered=strength>0;changed=(full!=bypass).flatten(1).any(1)
 if trace['selected'] is None:
  if bool(triggered.any()) or bool(changed.any()):raise ValueError('No-call PA mask must be empty')
 else:
  if len(trace['selected'])!=int(triggered.sum()) or not torch.equal(trace['selected'],bypass[triggered]) or not torch.equal(trace['strengths'],strength[triggered]):raise ValueError('Captured PA input and actual strength mask differ')
 if bool((changed&~triggered).any()) or not torch.equal(full[~triggered],bypass[~triggered]):raise ValueError('Untriggered packets must be exactly identical')
 delta=(full-bypass).square();power=x.square().mean((1,2)).clamp_min(1e-20)
 info=dict(triggered=triggered.tolist(),changed=changed.tolist(),strength=strength.tolist(),PA_MSE=delta.mean((1,2)).tolist(),PA_reference_window_MSE=delta[:,:,80:160].mean((1,2)).tolist(),prefix_raw_MSE=(bypass-x).square().mean((1,2)).tolist(),full_raw_MSE=(full-x).square().mean((1,2)).tolist(),full_raw_power_ratio=(full.square().mean((1,2))/power).tolist(),bypass_raw_power_ratio=(bypass.square().mean((1,2))/power).tolist(),subchains=trace['subchains'],same_native_draw=True,untriggered_exact=True)
 return full,bypass,info

def summary(labels,full,bypass,changed,rx,day):
 if not (len(labels)==len(full)==len(bypass)==len(changed)==len(rx)==len(day)):raise ValueError('Complete paired coverage required')
 def cell(take):
  repairs=sum(full[i]!=labels[i] and bypass[i]==labels[i] for i in take);broken=sum(full[i]==labels[i] and bypass[i]!=labels[i] for i in take)
  return dict(count=len(take),repairs=repairs,broken=broken,recovery_pp=100*(repairs-broken)/len(take) if take else None)
 take=[i for i,v in enumerate(changed) if v]
 return dict(changed=cell(take),all=cell(range(len(labels))),by_RX={str(r):cell([i for i in take if rx[i]==r]) for r in d.FIT},by_day={str(dd):cell([i for i in take if day[i]==dd]) for dd in (1,2,3)},by_TX={str(tx):cell([i for i in take if labels[i]==tx]) for tx in range(6)})

def qualify(s):
 c=s['changed'];rx=s['by_RX'];days=s['by_day']
 conditions=dict(sufficient_changed=c['count']>=1000,changed_identity_recovery=c['recovery_pp'] is not None and c['recovery_pp']>=2.,RX_coverage=all(rx[str(r)]['count']>0 for r in d.FIT),RX_positive=sum(v['recovery_pp'] is not None and v['recovery_pp']>0 for v in rx.values())>=3,RX_protection=all(v['recovery_pp'] is not None and v['recovery_pp']>=-1. for v in rx.values()),day_protection=all(v['recovery_pp'] is not None and v['recovery_pp']>=0 for v in days.values()))
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions)
