"""Full native replay, bypassing only circular shift with exogenous draws fixed."""
import random
import torch
from . import design as d

def rng():return random.getstate(),torch.random.get_rng_state().clone()
def restore(state):random.setstate(state[0]);torch.random.set_rng_state(state[1])
def same(a,b):return a[0]==b[0] and torch.equal(a[1],b[1])

def paired(aug,x,y,canonical=False):
 if x.device.type!='cpu' or x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or y.dtype!=torch.long or y.shape!=(len(x),) or bool(((y<0)|(y>5)).any()):raise ValueError('Finite CPU visible L IQ256/int64 labels required')
 if aug.p_dac!=0 or aug.enable_class_signature or aug.p_pa<=0 or aug.max_time_shift<=0:raise ValueError('Actual ordinary recipe required')
 scope=aug._apply_general_channel.__func__.__globals__;original=scope['_apply_time_shift'];begin=rng();raw=x.clone();expected=None
 if canonical:
  expected,expected_strength=aug(x.clone(),labels=y,no_pa=False,return_pa_strength=True);expected_end=rng();restore(begin)
 traces=[]
 def replay(bypass):
  trace=dict(shifts=[0]*len(x),calls=0,inspection_draws=0)
  def shift(value,maximum):
   if trace['calls']:raise ValueError('One shift call per native ordinary batch required')
   before=torch.random.get_rng_state();shifted=original(value,maximum);after=torch.random.get_rng_state()
   torch.random.set_rng_state(before);shifts=torch.randint(-maximum,maximum+1,(len(value),),device=value.device)
   idx=(torch.arange(value.shape[2]).view(1,-1)-shifts.view(-1,1))%value.shape[2]
   reconstructed=torch.gather(value,2,idx[:,None,:].expand_as(value))
   if not torch.equal(reconstructed,shifted) or not torch.equal(after,torch.random.get_rng_state()):raise ValueError('Native shift draw/output changed')
   torch.random.set_rng_state(after);trace.update(shifts=shifts.tolist(),calls=1,inspection_draws=len(value))
   return value if bypass else shifted
  scope['_apply_time_shift']=shift
  try:out,strength=aug(x.clone(),labels=y,no_pa=False,return_pa_strength=True)
  finally:scope['_apply_time_shift']=original
  traces.append(trace);return out,strength,rng()
 try:
  full,full_strength,end=replay(False);restore(begin);bypass,bypass_strength,bypass_end=replay(True)
  if not same(end,bypass_end) or traces[0]!=traces[1] or not torch.equal(full_strength,bypass_strength):raise ValueError('Native random replay/PA selection changed')
  if canonical and (not torch.equal(expected,full) or not torch.equal(expected_strength,full_strength) or not same(expected_end,end)):raise ValueError('Canonical whole native output/RNG mismatch')
  if not torch.equal(x,raw) or not bool(torch.isfinite(full).all()) or not bool(torch.isfinite(bypass).all()):raise ValueError('Raw mutated or nonfinite augmented IQ')
  shifts=traces[0]['shifts'];nonzero=torch.tensor(shifts)!=0;changed=(full!=bypass).flatten(1).any(1)
  if bool((changed&~nonzero).any()) or not torch.equal(full[~nonzero],bypass[~nonzero]):raise ValueError('Unshifted packets must be identical')
  delta=(full-bypass).square();power=x.square().mean((1,2)).clamp_min(1e-20)
  info=dict(shifts=shifts,nonzero_shift=nonzero.tolist(),changed=changed.tolist(),shift_calls=traces[0]['calls'],shift_inspection_scalar_draws=2*traces[0]['inspection_draws'],PA_strength=full_strength.tolist(),pair_MSE=delta.mean((1,2)).tolist(),reference_window_MSE=delta[:,:,80:160].mean((1,2)).tolist(),full_raw_MSE=(full-x).square().mean((1,2)).tolist(),bypass_raw_MSE=(bypass-x).square().mean((1,2)).tolist(),full_raw_power_ratio=(full.square().mean((1,2))/power).tolist(),bypass_raw_power_ratio=(bypass.square().mean((1,2))/power).tolist(),same_native_draw=True,unshifted_exact=True,canonical_checked=canonical)
  return full,bypass,info
 finally:
  scope['_apply_time_shift']=original
  if 'end' in locals():restore(end)
  else:restore(begin)

def summary(labels,full,bypass,nonzero,rx,day):
 if not (len(labels)==len(full)==len(bypass)==len(nonzero)==len(rx)==len(day)):raise ValueError('Complete paired coverage required')
 def cell(take):
  repairs=sum(full[i]!=labels[i] and bypass[i]==labels[i] for i in take);broken=sum(full[i]==labels[i] and bypass[i]!=labels[i] for i in take)
  return dict(count=len(take),repairs=repairs,broken=broken,recovery_pp=100*(repairs-broken)/len(take) if take else None)
 take=[i for i,v in enumerate(nonzero) if v]
 return dict(nonzero_shift=cell(take),all=cell(range(len(labels))),by_RX={str(r):cell([i for i in take if rx[i]==r]) for r in d.FIT},by_day={str(v):cell([i for i in take if day[i]==v]) for v in (1,2,3)},by_TX={str(v):cell([i for i in take if labels[i]==v]) for v in range(6)})
def qualify(s):
 c=s['nonzero_shift'];rx=s['by_RX'];conditions=dict(nonzero_coverage=c['count']>0,identity_recovery=c['recovery_pp'] is not None and c['recovery_pp']>=2.,RX_coverage=all(v['count']>0 for v in rx.values()),RX_positive=sum(v['recovery_pp'] is not None and v['recovery_pp']>0 for v in rx.values())>=3)
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions)
