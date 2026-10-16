"""Native residual replay,finite-waveform quality and fixed receiver control."""
from contextlib import contextmanager
from dataclasses import asdict
import copy,math
import numpy as np

def power(x):return float(np.mean(np.abs(x)**2))
def qdb(p,n):
 if not math.isfinite(p+n) or p<=0 or n<=0:raise ValueError('Positive finite power/noise required')
 return 10*math.log10(p/n)
@contextmanager
def held_receiver(residual,status,scale):
 old=residual.receiver_quality;residual.receiver_quality=lambda cfg,q:(status,scale)
 try:yield
 finally:residual.receiver_quality=old

def measure(ch,residual,linear_filter,cfg,x,seed,receiver):
 if cfg.processing_route!='residual' or cfg.mode!='post_sync' or cfg.equalization_enabled or cfg.reference_snr_db!=30. or cfg.boundary!='reflect' or np.asarray(x).shape!=(256,) or not np.isfinite(x).all():raise ValueError('Original residual/EQoff/reflect/sourceIQ256 only')
 x=np.asarray(x,dtype=np.complex128);calls=0
 def call(q=None):
  nonlocal calls
  instance=residual.ResidualChannel(cfg,seed,receiver);y,m,tr=instance.process(x,quality_snr_db=q,return_traces=True);calls+=1
  if not np.array_equal(tr['equalizer_kernel'],np.ones(1,complex)) or m['channel_equalization_applied'] or m['full_channel_waveform_synthesized'] or not np.isfinite(y).all():raise ValueError('Original G1 finite residual required')
  return y,m,tr,copy.deepcopy(instance.latent.rngs['noise'].bit_generator.state)
 y,m,tr,nstate=call();h=tr['propagation_kernel'];hx,_=linear_filter(x,h,cfg.boundary);start=len(h)-1
 if start<0 or start>=256:raise ValueError('Finite causal boundary-free region required;never drop packet')
 pin=power(x);eh=float(np.sum(np.abs(h)**2));nv=m['added_noise_variance_reference'];pw=power(hx);pv=power(hx[start:]);q0=qdb(pin*eh,nv);qw=qdb(pw,nv);qv=qdb(pv,nv)
 if not math.isclose(q0,m['quality_snr_db'],abs_tol=1e-12):raise ValueError('Native white-input proxy formula changed')
 yc,mc,tc,nc=call(m['quality_snr_db']);yw,mw,tw,nw=call(qw);yv,mv,tv,nvstate=call(qv)
 with held_receiver(residual,m['receiver_lock_status'],m['residual_scale']):yf,mf,tf,nf=call(qw)
 if not np.array_equal(y,yc) or not np.array_equal(y,yf) or any(nstate!=v for v in (nc,nw,nvstate,nf)):raise ValueError('Proxy replay/fixed-state or native noise-draw stream differs')
 for t in (tc,tw,tv,tf):
  if not np.array_equal(h,t['propagation_kernel']):raise ValueError('Physical H changed in quality counterfactual')
 def compact(a,t,yy):return dict(quality_snr_db=a['quality_snr_db'],lock=a['receiver_lock_status'],scale=a['residual_scale'],frequency_hz=a['output_frequency_hz'],phase_tracking=a['phase_tracking_applied'],EQ=a['channel_equalization_applied'],IQ_compensation=a['iq_compensation_applied'],agc_db=a['agc_gain_db'],agc_limited=a['agc_limited'],signal_to_added_noise_db=a['signal_to_added_noise_db'],pre_agc_signal_power=power(t['pre_agc_signal']),pre_agc_noise_power=power(t['pre_agc_noise']),output_power=power(yy))
 return dict(input_power=pin,propagation_energy=eh,noise_variance_reference=nv,filtered_power_whole=pw,filtered_power_interior=pv,kernel_length=len(h),numerical_filter_latency_samples=m['numerical_filter_latency_samples'],interior_start=start,interior_count=256-start,boundary=m['boundary'],noise_boundary=m['noise_boundary'],state=m['state_end'],receiver=asdict(receiver),native=compact(m,tr,y),whole=compact(mw,tw,yw),interior=compact(mv,tv,yv),fixed=compact(mf,tf,yf),proxy_replay_MSE=float(np.mean(abs(y-yc)**2)),whole_total_MSE=float(np.mean(abs(y-yw)**2)),interior_total_MSE=float(np.mean(abs(y-yv)**2)),fixed_receiver_MSE=float(np.mean(abs(y-yf)**2)),same_H=True,same_noise_draw_stream=True,native_process_calls=calls,true_total_snr_known=False)
