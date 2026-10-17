"""Same latent practical channel;only existing receiver equalizer changes."""
from dataclasses import replace
import numpy as np
from . import design as d

def triplet(x,ids,rx,day,scene):
 from leo_practical import channel as ch
 from leo_practical.residual import ResidualChannel
 from leo_practical.equalization import linear_filter
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all() or np.any(np.sum(x*x,axis=(1,2))<=0):raise ValueError('Finite nonzero IQ256 required')
 if scene not in d.SCENES or len(ids)!=len(x) or len(rx)!=len(x) or len(day)!=len(x) or len(set(ids))!=len(ids) or any(not isinstance(v,str) or not v for v in ids) or set(rx)-set(d.FIT) or set(day)-{1,2,3}:raise ValueError('Original opaque source IDs/RX/day and fixed scene required')
 cfg=ch.Config(fs_hz=25e6,fc_hz=2462e6,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
 outputs={k:[] for k in ('noEQ','MMSE24','MMSE64')};records=[]
 for row,pid,r,z in zip(x,ids,rx,day):
  source,_=ch.normalize_input(row[0]+1j*row[1]);seed=ch.stable_seed(d.PROBE_SEED,'MMSE_source_coupled',scene,pid);session=f'rx{r}:day{z}:eq1';hw=ch.receiver_for_session(cfg,d.RECEIVER_SEED,session)
  models={k:ResidualChannel(c,seed,hw) for k,c in [('noEQ',cfg),('MMSE24',replace(cfg,equalization_enabled=True,equalizer_method='mmse',equalizer_delay_samples=24)),('MMSE64',replace(cfg,equalization_enabled=True,equalizer_method='mmse',equalizer_delay_samples=64))]}
  values={k:m.process(source,return_traces=True) for k,m in models.items()};ref,meta,trace=values['noEQ'];rec=dict(id=pid,RX=int(r),day=int(z),scene=scene,state=meta['state_end'],lock=meta['receiver_lock_status'],quality=meta['quality_snr_db'],same_H=True,same_receiver_quality_and_compensation=True,noise_filter_verified=True,variants={})
  for k,(a,m,t) in values.items():
   for field in ('geometry','receiver','state_start','state_end','quality_snr_db','receiver_lock_status','residual_scale','orbital_doppler_hz','estimated_frequency_hz','output_frequency_hz','phase_tracking_applied','iq_compensation_applied','added_noise_variance_reference'):
    if m[field]!=meta[field]:raise ValueError('Latent/receiver dependency changed:'+field)
   if not np.array_equal(trace['propagation_kernel'],t['propagation_kernel']):raise ValueError('H changed')
   expected,_=linear_filter(trace['pre_agc_noise'],t['equalizer_kernel'],cfg.boundary)
   error=float(np.max(np.abs(expected-t['pre_agc_noise'])))
   if error>1e-13:raise ValueError('Base noise was not filtered by G once')
   if m['channel_equalization_applied']!=(k!='noEQ' and meta['receiver_lock_status']!='unlocked'):raise ValueError('Receiver lock dependency changed')
   if meta['receiver_lock_status']=='unlocked' and not np.array_equal(a,ref):raise ValueError('Unlocked IQ must stay exactly nominal')
   if not np.isfinite(a).all():raise ValueError('No nonfinite or selective discard')
   out=np.stack((a.real,a.imag));outputs[k].append(out);rec['variants'][k]=dict(EQ=m['channel_equalization_applied'],noise_filter_max_error=error,relative_IQ_change=float(np.linalg.norm(a-ref)/max(np.linalg.norm(ref),1e-30)),signal_to_added_noise_db=m['signal_to_added_noise_db'],noise_power=m['measured_added_noise_power_pre_agc'],AGC_db=m['agc_gain_db'],AGC_limited=m['agc_limited'],equalizer=m['equalizer'],residual_kernel_energy=m['residual_kernel_energy'])
  records.append(rec)
 return {k:np.ascontiguousarray(v,dtype=np.float32) for k,v in outputs.items()},records
