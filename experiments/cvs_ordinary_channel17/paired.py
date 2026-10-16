"""Paired total-chain ordinary controls;original primitives still consume their draws."""
import copy,inspect,random
import torch
from . import design as d

def power(z):return (z.real.double().square()+z.imag.double().square()).mean(1).tolist()
def kernel_parameters(z,before,after,num_taps,max_delay,k_factor):
 g=torch.Generator(device='cpu');g.set_state(before);delays=torch.randint(0,max_delay+1,(num_taps,),generator=g);hr=torch.randn(num_taps,generator=g,dtype=z.real.dtype);hi=torch.randn(num_taps,generator=g,dtype=z.real.dtype)
 if not torch.equal(g.get_state(),after):raise ValueError('Native multipath primitive draws differ')
 h=torch.complex(hr,hi)
 if k_factor>0:h[0]+=float(k_factor)**.5
 h=h/(torch.sqrt(h.real.square().add(h.imag.square()).sum())+1e-12);hh=torch.zeros(int(delays.max())+1,dtype=torch.complex64)
 for i in range(num_taps):hh[int(delays[i])]+=h[i].to(torch.complex64)
 return dict(num_taps=num_taps,max_delay=max_delay,k_factor=k_factor,delays=delays.tolist(),tap_real=h.real.tolist(),tap_imag=h.imag.tolist(),kernel_real=hh.real.tolist(),kernel_imag=hh.imag.tolist())

def paired_views(aug,x,labels):
 if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype!=torch.float32 or x.device.type!='cpu' or x.requires_grad or not bool(torch.isfinite(x).all()) or not bool(x.double().square().sum((1,2)).gt(0).all()):raise ValueError('Finite nonzero CPU sourcefloat32 IQ[B,2,256],no autograd required')
 if not isinstance(labels,torch.Tensor) or labels.shape!=(len(x),) or labels.dtype!=torch.int64 or labels.device.type!='cpu' or not bool(((labels>=0)&(labels<6)).all()):raise ValueError('Visible matching sixclass L labels required')
 if aug.enable_class_signature or aug.p_dac!=0:raise ValueError('Original classsignatureoff/DACoff sourceordinary required')
 module=inspect.getmodule(type(aug));native_MP,native_AWGN=module._apply_multipath,module._apply_awgn;saved=x.clone();attributes=copy.deepcopy(aug.__dict__);begin_py=random.getstate();begin=torch.random.get_rng_state();outputs={};events={};reference_states=None;reference_strength=None;replay=0;full_end=None
 try:
  for arm in d.ARMS:
   random.setstate(begin_py);torch.random.set_rng_state(begin);tape=[];states=[]
   def MP(z,num_taps,max_delay,k_factor=0.):
    nonlocal replay
    before=torch.random.get_rng_state();out=native_MP(z,num_taps,max_delay,k_factor);after=torch.random.get_rng_state();params=kernel_parameters(z,before,after,num_taps,max_delay,k_factor);replay+=3*num_taps;removed=arm in ('no_MP','no_both');returned=z if removed else out;tape.append(dict(kind='MP',parameters=params,removed=removed,input_power=power(z),native_output_power=power(out),returned_power=power(returned)));states.append((before,after));return returned
   def AWGN(z,snr_db):
    before=torch.random.get_rng_state();out=native_AWGN(z,snr_db);after=torch.random.get_rng_state();removed=arm in ('no_AWGN','no_both');returned=z if removed else out;tape.append(dict(kind='AWGN',parameters=dict(snr_db=snr_db.tolist()),removed=removed,input_power=power(z),native_output_power=power(out),returned_power=power(returned),native_added_noise_power=power(out-z)));states.append((before,after));return returned
   module._apply_multipath,module._apply_awgn=MP,AWGN
   try:o,sp=aug(x.clone(),labels=labels.clone(),no_pa=False,return_pa_strength=True)
   finally:module._apply_multipath,module._apply_awgn=native_MP,native_AWGN
   if o.shape!=x.shape or o.dtype!=x.dtype or not bool(torch.isfinite(o).all()) or not bool(torch.isfinite(sp).all()) or not torch.equal(x,saved) or aug.__dict__!=attributes:raise ValueError('Native observation/input/config contract differs')
   end=(random.getstate(),torch.random.get_rng_state())
   if arm=='full':full_end=end;reference_states=states;reference_strength=sp.clone()
   else:
    if end[0]!=full_end[0] or not torch.equal(end[1],full_end[1]) or not torch.equal(sp,reference_strength) or len(states)!=len(reference_states) or [(e['kind'],e['parameters']) for e in tape]!=[(e['kind'],e['parameters']) for e in events['full']] or any(not torch.equal(a,c) or not torch.equal(b,e) for (a,b),(c,e) in zip(states,reference_states)):raise ValueError('Native primitive parameters/PAstrength/RNG not matched')
    if not any(z['removed'] for z in tape) and not torch.equal(o,outputs['full']):raise ValueError('No active removal must preserve originalIQ exactly')
   outputs[arm]=o;events[arm]=tape
  random.setstate(full_end[0]);torch.random.set_rng_state(full_end[1]);record=dict(events=events,all_native_parameters_RNG_PA_matched=True,wrappers_restored=True,input_unchanged=True,ordinary_generations=4*len(x),kernel_replay_random_scalars=replay)
  return outputs,reference_strength,record
 except Exception:
  random.setstate(begin_py);torch.random.set_rng_state(begin);raise
 finally:module._apply_multipath,module._apply_awgn=native_MP,native_AWGN
