"""Same latent complete residual;replace N0 once,with paired fixed-quality controls."""
from dataclasses import replace
import math
import numpy as np
from . import design as d

def observations(x,ids,rx,day,scene):
 from leo_practical import channel as ch
 from leo_practical.residual import ResidualChannel
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all() or np.any(np.sum(x*x,axis=(1,2))<=0):raise ValueError('Finite nonzero IQ256 required')
 if scene not in d.SCENES or len(ids)!=len(x) or len(rx)!=len(x) or len(day)!=len(x) or len(set(ids))!=len(ids) or any(not isinstance(v,str) or not v for v in ids) or set(rx)-set(d.FIT) or set(day)-{1,2,3}:raise ValueError('Exact source IDs/RX/day/scene required')
 cfg=ch.Config(fs_hz=25e6,fc_hz=2462e6,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
 outputs={k:[] for k in d.VARIANTS};records=[]
 for row,pid,r,z in zip(x,ids,rx,day):
  source,_=ch.normalize_input(row[0]+1j*row[1]);seed=ch.stable_seed(d.PROBE_SEED,d.PROBE_NAMESPACE,scene,pid);hw=ch.receiver_for_session(cfg,d.RECEIVER_SEED,f'rx{r}:day{z}:eq1')
  base=ResidualChannel(cfg,seed,hw).process(source,return_traces=True);ref,meta,trace=base;values={'nominal':base}
  for key in d.VARIANTS[1:]:
   mult=d.FACTORS[key];c=replace(cfg,reference_snr_db=cfg.reference_snr_db-10*math.log10(mult));q=meta['quality_snr_db'] if key.endswith('fixedQ') else None
   values[key]=ResidualChannel(c,seed,hw).process(source,return_traces=True,quality_snr_db=q)
  rec=dict(id=pid,RX=int(r),day=int(z),scene=scene,state=meta['state_end'],nominal_lock=meta['receiver_lock_status'],nominal_quality=meta['quality_snr_db'],same_H=True,same_hardware=True,variants={})
  for key,(a,m,t) in values.items():
   mult=d.FACTORS[key];fixed=key.endswith('fixedQ');wantQ=meta['quality_snr_db'] if fixed else meta['quality_snr_db']-10*math.log10(mult)
   for field in ('geometry','receiver','state_start','state_end','orbital_doppler_hz','iq_compensation_applied'):
    if m[field]!=meta[field]:raise ValueError('Original latent changed:'+field)
   if not np.array_equal(t['propagation_kernel'],trace['propagation_kernel']):raise ValueError('H changed')
   if abs(m['added_noise_variance_reference']/meta['added_noise_variance_reference']-mult)>1e-12 or abs(m['quality_snr_db']-wantQ)>1e-10:raise ValueError('N0 or quality dependency differs')
   lock,scale=ch.receiver_quality(cfg,wantQ)
   if m['receiver_lock_status']!=lock or abs(m['residual_scale']-scale)>1e-12 or m['channel_equalization_applied'] or m['full_channel_waveform_synthesized']:raise ValueError('Original quality/NoEQ/single-residual rule differs')
   se=ne=None
   if fixed or key=='nominal':
    se=float(np.max(np.abs(t['pre_agc_signal']-trace['pre_agc_signal'])));ne=float(np.max(np.abs(t['pre_agc_noise']-math.sqrt(mult)*trace['pre_agc_noise'])))
    if se>1e-13 or ne>1e-13:raise ValueError('Fixed quality signal/noise causal control differs')
   rms=float(np.sqrt(np.mean(np.abs(t['pre_agc_signal']+t['pre_agc_noise'])**2)));wantAgc=float(np.clip(-20*np.log10(max(rms,1e-300)),-cfg.agc_limit_db,cfg.agc_limit_db)) if cfg.agc_enabled else 0.
   if abs(wantAgc-m['agc_gain_db'])>1e-10 or not np.isfinite(a).all():raise ValueError('Actual AGC/nonfinite handling differs')
   outputs[key].append(np.stack((a.real,a.imag)));rec['variants'][key]=dict(N0_factor=mult,noise_variance_reference=m['added_noise_variance_reference'],quality=m['quality_snr_db'],lock=lock,scale=scale,phase_tracking=m['phase_tracking_applied'],output_frequency_Hz=m['output_frequency_hz'],signal_to_added_noise_db=m['signal_to_added_noise_db'],noise_power=m['measured_added_noise_power_pre_agc'],AGC_db=m['agc_gain_db'],AGC_limited=m['agc_limited'],relative_IQ_change=float(np.linalg.norm(a-ref)/max(np.linalg.norm(ref),1e-30)),fixed_quality_signal_error=se,fixed_quality_noise_error=ne)
  if abs(.5*(rec['variants']['N0_half']['noise_variance_reference']+rec['variants']['N0_1p5']['noise_variance_reference'])-meta['added_noise_variance_reference'])>1e-15:raise ValueError('Expected reference noise power changed')
  records.append(rec)
 return {k:np.ascontiguousarray(v,dtype=np.float32) for k,v in outputs.items()},records
