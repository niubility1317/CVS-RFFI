"""Canonical full native draws both arms;only PA return bypass differs."""
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

class Wrapper:
 def __init__(self,native,arm,records):
  if arm not in d.ARMS:raise ValueError('Fixed PA arm required')
  object.__setattr__(self,'_native',native);object.__setattr__(self,'_arm',arm);object.__setattr__(self,'_records',records)
 def __getattr__(self,key):return getattr(self._native,key)
 def __setattr__(self,key,value):setattr(self._native,key,value)
 def __call__(self,x,*args,**kwargs):
  if args or set(kwargs)!={'labels','no_pa'} or kwargs['no_pa'] is not False or kwargs['labels'] is None or len(x)!=128:raise ValueError('Only actual original normal visible L128 call permitted;U paths untouched')
  full,prefix,info=paired(self._native,x,kwargs['labels']);ch=info['changed'];tr=info['triggered']
  record=dict(step=len(self._records)+1,count=len(x),effective_p_pa=self._native.p_pa,effective_max_time_shift=self._native.max_time_shift,triggered=sum(tr),changed=sum(ch),triggered_bits=''.join('1' if v else '0' for v in tr),changed_bits=''.join('1' if v else '0' for v in ch),subchains=info['subchains'],PA_MSE_sum=sum(info['PA_MSE']),PA_reference_window_MSE_sum=sum(info['PA_reference_window_MSE']),full_raw_MSE_sum=sum(info['full_raw_MSE']),prefix_raw_MSE_sum=sum(info['prefix_raw_MSE']),strength_max=max(info['strength']),full_raw_power_ratio_min=min(info['full_raw_power_ratio']),full_raw_power_ratio_max=max(info['full_raw_power_ratio']),prefix_raw_power_ratio_min=min(info['bypass_raw_power_ratio']),prefix_raw_power_ratio_max=max(info['bypass_raw_power_ratio']),untriggered_exact=info['untriggered_exact'],same_native_draw=True,bypass_returned=self._arm=='B10')
  self._records.append(record);return prefix if self._arm=='B10' else full
