"""One native ordinary call, preserving the shift draw before optional bypass."""
import torch
from . import design as d

def observed(native,x,y,bypass):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or y.dtype!=torch.long or y.shape!=(len(x),) or bool(((y<0)|(y>5)).any()):raise ValueError('Finite visible L IQ256/int64 labels required')
 if native.p_dac!=0 or native.enable_class_signature or native.p_pa<=0 or not 3<=native.max_time_shift<=8:raise ValueError('Actual capped ordinary recipe required')
 scope=native._apply_general_channel.__func__.__globals__;original=scope['_apply_time_shift'];trace=dict(calls=0,shifts=[0]*len(x),changed=[False]*len(x),MSE_sum=0.,reference_MSE_sum=0.,inspection_draws=0)
 def get():return torch.cuda.get_rng_state(x.device) if x.device.type=='cuda' else torch.get_rng_state()
 def put(v):
  if x.device.type=='cuda':torch.cuda.set_rng_state(v,x.device)
  else:torch.set_rng_state(v)
 def shift(value,maximum):
  if trace['calls'] or maximum!=native.max_time_shift:raise ValueError('Exactly one matching native shift call required')
  before=get();gathered=original(value,maximum);after=get()
  try:
   put(before);shifts=torch.randint(-maximum,maximum+1,(len(value),),device=value.device)
   idx=(torch.arange(value.shape[2],device=value.device).view(1,-1)-shifts.view(-1,1))%value.shape[2]
   if not torch.equal(gathered,torch.gather(value,2,idx[:,None,:].expand_as(value))) or not torch.equal(after,get()):raise ValueError('Native shift draw/output mismatch')
  finally:put(after)
  changed=(gathered!=value).flatten(1).any(1);delta=(gathered-value).square()
  if bool((changed&(shifts==0)).any()):raise ValueError('Zero shift must preserve IQ')
  trace.update(calls=1,shifts=shifts.tolist(),changed=changed.tolist(),MSE_sum=float(delta.mean((1,2)).sum()),reference_MSE_sum=float(delta[:,:,80:160].mean((1,2)).sum()),inspection_draws=len(value))
  return value if bypass else gathered
 scope['_apply_time_shift']=shift
 try:out,strength=native(x.clone(),labels=y,no_pa=False,return_pa_strength=True)
 finally:scope['_apply_time_shift']=original
 if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(strength).all()):raise ValueError('Finite ordinary IQ and PA strengths required')
 power=x.square().mean((1,2)).clamp_min(1e-20);ratio=out.square().mean((1,2))/power;pa=strength.tolist();shifts=trace['shifts']
 return out,dict(count=len(x),effective_p_time_shift=native.p_time_shift,effective_max_time_shift=native.max_time_shift,effective_p_pa=native.p_pa,shift_gate_calls=trace['calls'],shifts=shifts,nonzero_bits=''.join('1' if v else '0' for v in shifts),nonzero=sum(v!=0 for v in shifts),shift_changed_bits=''.join('1' if v else '0' for v in trace['changed']),shift_changed=sum(trace['changed']),shift_MSE_sum=trace['MSE_sum'],shift_reference_MSE_sum=trace['reference_MSE_sum'],shift_inspection_scalar_draws=trace['inspection_draws'],PA_triggered_bits=''.join('1' if v>0 else '0' for v in pa),PA_strength=pa,returned_raw_MSE_sum=float((out-x).square().mean((1,2)).sum()),returned_raw_power_ratio_min=float(ratio.min()),returned_raw_power_ratio_max=float(ratio.max()),same_native_shift_draw=True,RNG_inspection_no_change=True,bypass_returned=bypass)

class Wrapper:
 def __init__(self,native,arm,records):
  if arm not in d.ARMS:raise ValueError('Fixed shift arm required')
  object.__setattr__(self,'_native',native);object.__setattr__(self,'_arm',arm);object.__setattr__(self,'_records',records)
 def __getattr__(self,key):return getattr(self._native,key)
 def __setattr__(self,key,value):setattr(self._native,key,value)
 def __call__(self,x,*args,**kwargs):
  if args or set(kwargs)!={'labels','no_pa'} or kwargs['no_pa'] is not False or kwargs['labels'] is None or len(x)!=128:raise ValueError('Only original normal visible L128 call permitted;U methods untouched')
  value,record=observed(self._native,x,kwargs['labels'],self._arm=='B10');record['step']=len(self._records)+1;self._records.append(record);return value
