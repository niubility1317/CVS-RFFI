"""L-scoped native primitive output bypass; original draws and shadow work retained."""
import inspect,random
import numpy as np
import torch
_ACTIVE=False

def rng_state(device):
 return (random.getstate(),torch.get_rng_state().clone(),np.random.get_state(),torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None)
def same_rng(a,b):
 return a[0]==b[0] and torch.equal(a[1],b[1]) and a[2][0]==b[2][0] and np.array_equal(a[2][1],b[2][1]) and a[2][2:]==b[2][2:] and ((a[3] is None and b[3] is None) or (a[3] is not None and b[3] is not None and torch.equal(a[3],b[3])))
def restore_rng(state,device):
 random.setstate(state[0]);torch.set_rng_state(state[1]);np.random.set_state(state[2])
 if state[3] is not None:torch.cuda.set_rng_state(state[3],device)
def patch_active():return _ACTIVE

def draw(native,x,labels,enabled=True):
 global _ACTIVE
 if _ACTIVE:raise ValueError('Nested L primitive patch forbidden')
 if x.shape!=(128,2,256) or x.dtype!=torch.float32 or x.requires_grad or not bool(torch.isfinite(x).all()) or labels.shape!=(128,) or labels.dtype!=torch.int64 or labels.device!=x.device or bool(((labels<0)|(labels>5)).any()):raise ValueError('Finite visible L128 IQ256 and labels0..5 required')
 module=inspect.getmodule(type(native));MP=module._apply_multipath;AWGN=module._apply_awgn;saved=x.clone();events=[]
 def primitive(kind,original):
  def call(z,*args,**kwargs):
   before=rng_state(z.device);entry=z.clone();shadow=original(z,*args,**kwargs);after=rng_state(z.device)
   if shadow.shape!=z.shape or shadow.dtype!=z.dtype or not bool(torch.isfinite(shadow).all()) or not torch.equal(z,entry):raise ValueError('Native primitive input/shape/finite contract differs')
   returned=entry if enabled else shadow
   if not same_rng(after,rng_state(z.device)):raise ValueError('Primitive observer consumed random state')
   events.append(dict(kind=kind,packets=len(z),shadow_packets=len(z),bypassed=bool(enabled),input_unchanged=True,CPU_rng_advanced=not torch.equal(before[1],after[1]),CUDA_rng_advanced=False if before[3] is None else not torch.equal(before[3],after[3]),Python_rng_advanced=before[0]!=after[0],post_return_rng_preserved=True))
   return returned
  return call
 try:
  _ACTIVE=True;module._apply_multipath=primitive('MP',MP);module._apply_awgn=primitive('AWGN',AWGN)
  result=native(x.clone(),labels=labels,no_pa=False)
 finally:
  module._apply_multipath=MP;module._apply_awgn=AWGN;_ACTIVE=False
 if result.shape!=x.shape or result.dtype!=x.dtype or not bool(torch.isfinite(result).all()) or not torch.equal(x,saved):raise ValueError('Native ordinary input/output contract differs')
 if module._apply_multipath is not MP or module._apply_awgn is not AWGN:raise ValueError('L primitive patch not restored')
 record=dict(count=128,enabled=bool(enabled),input_unchanged=True,primitive_restored=True,full_native_aug_computation=True,post_return_rng_preserved=all(e['post_return_rng_preserved'] for e in events),primitive_CPU_rng_advanced=sum(e['CPU_rng_advanced'] for e in events),primitive_CUDA_rng_advanced=sum(e['CUDA_rng_advanced'] for e in events),primitive_Python_rng_advanced=sum(e['Python_rng_advanced'] for e in events),returned_ordinary_raw_MSE=float((result-x).square().mean()),extra_NN=0)
 for kind in ('MP','AWGN'):
  ee=[e for e in events if e['kind']==kind];record[kind+'_calls']=len(ee);record[kind+'_packets']=sum(e['packets'] for e in ee);record['shadow_'+kind+'_packets']=sum(e['shadow_packets'] for e in ee);record[kind+'_hit']=bool(ee)
 if any(record[k+'_calls'] not in (0,1) for k in ('MP','AWGN')):raise ValueError('Only one native general MP/AWGN primitive per L call expected')
 return result,record

class Wrapper:
 def __init__(self,native,records):object.__setattr__(self,'_native',native);object.__setattr__(self,'_records',records)
 def __getattr__(self,key):return getattr(self._native,key)
 def __setattr__(self,key,value):setattr(self._native,key,value)
 def __call__(self,x,*args,**kwargs):
  if args or set(kwargs)!={'labels','no_pa'} or kwargs['no_pa'] is not False or kwargs['labels'] is None:raise ValueError('Only actual visible L labels/no_paFalse call is wrapped')
  y,record=draw(self._native,x,kwargs['labels']);self._records.append(dict(step=len(self._records)+1,**record));return y

def ordinary_factory(original,records):
 def create(*args,**kwargs):
  create.calls+=1
  if create.calls!=1:raise ValueError('Exactly one original L ordinary object required')
  return Wrapper(original(*args,**kwargs),records)
 create.calls=0;return create

def strong_passthrough(original,records,step_provider):
 def call(x,*args,**kwargs):
  if _ACTIVE:raise ValueError('U strong must execute outside L primitive patch')
  if x.ndim!=3 or tuple(x.shape[1:])!=(2,256) or len(x) not in (256,124):raise ValueError('Original U256/last124 required')
  out=original(x,*args,**kwargs);end=rng_state(x.device)
  if out.shape!=x.shape or not bool(torch.isfinite(out).all()) or not same_rng(end,rng_state(x.device)):raise ValueError('Native U output/guard RNG differs')
  records.append(dict(step=step_provider(),packets=len(x),native_output_passthrough=True,L_primitive_wrappers_inactive=True,post_return_rng_preserved=True,extra_NN=0))
  return out
 return call
