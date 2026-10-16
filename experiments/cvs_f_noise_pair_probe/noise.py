"""Override one snapshot's primitive noise only; original receiver recomputes AGC."""
from dataclasses import asdict
import numpy as np
class FixedNormal:
 def __init__(self,value):
  a=np.asarray(value,dtype=np.float64)
  if a.ndim!=2 or a.shape[1]!=2 or len(a)<4 or not np.isfinite(a).all():raise ValueError('Finite primitive n x2 standard-normal draw required')
  self.value=a.copy();self.calls=0
 def normal(self,*,size):
  if self.calls or tuple(size)!=self.value.shape:raise ValueError('Exactly one snapshot noise draw of registered shape required')
  self.calls+=1;return self.value.copy()
def observe(residual,cfg,seed,receiver,x,z):
 if cfg.processing_route!='residual' or cfg.mode!='post_sync':raise ValueError('Original residual/post_sync required')
 engine=residual.ResidualChannel(cfg,seed,receiver);rng=FixedNormal(z)
 if len(x)!=len(rng.value):raise ValueError('Primitive noise length differs from IQ')
 engine.latent.rngs['noise']=rng
 y,meta,traces=engine.process(x,return_traces=True)
 if rng.calls!=1 or not np.isfinite(y).all():raise ValueError('Original noise injection was not consumed once')
 return y,meta,traces
class Channel:
 def __init__(self):
  from experiments.cvs_crossrx_satmix20.channel import imports
  from . import design as d
  self.ch,self.residual=imports();self.d=d;self.sessions={};self.calls=0
  self.configs={s:self.ch.Config(fs_hz=25_000_000.,fc_hz=2_462_000_000.,scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.SCENES}
 def batch(self,x,ids,rx,day):
  d=self.d;raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);arrays=np.empty((d.DRAWS,2,len(ids),2,256),dtype=np.float32);records=[]
  if raw.shape!=(len(ids),2,256) or not np.isfinite(raw).all() or len(set(ids))!=len(ids) or any(int(r) not in d.FIT for r in rx) or any(int(t) not in (1,2,3) for t in day):raise ValueError('Finite fitL only,heldRX1 excluded')
  for i,pid in enumerate(ids):
   ns='FcrossRX/noise_pair_diagnostic/fitL/v1';scene=str(np.random.default_rng(self.ch.stable_seed(d.STREAMS['channel'],ns,'scene',pid)).choice(d.SCENES));cfg=self.configs[scene];seed=self.ch.stable_seed(d.STREAMS['channel'],ns,scene,pid)
   session=f'rx{int(rx[i])}:day{int(day[i])}:eq1';hw=self.ch.receiver_for_session(cfg,d.STREAMS['receiver'],session);hardware=asdict(hw)
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('RX hardware varied within physical session')
   normalized,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);first_meta=None;first_signal=None;first_kernels=None
   for k in range(d.DRAWS):
    z=np.random.default_rng(self.ch.stable_seed(d.STREAMS['noise'],ns,'primitive',pid,k)).normal(size=(256,2));pair=[]
    for j,sign in enumerate((1.,-1.)):
     y,m,t=observe(self.residual,cfg,seed,hw,normalized,sign*z);self.calls+=1
     same_keys=('state_end','receiver_lock_status','quality_snr_db','residual_scale','output_frequency_hz','channel_equalization_applied','phase_tracking_applied','iq_compensation_applied','numerical_filter_latency_samples','residual_kernel_energy','boundary')
     state={key:m[key] for key in same_keys}
     if first_meta is None:first_meta=state;first_signal=t['pre_agc_signal'].copy();first_kernels={key:t[key].copy() for key in ('residual_kernel','equalizer_kernel')}
     if state!=first_meta or not np.array_equal(t['pre_agc_signal'],first_signal) or any(not np.array_equal(t[key],v) for key,v in first_kernels.items()):raise ValueError('Non-noise receiver/propagation state changed')
     if m['channel_equalization_applied'] or len(t['equalizer_kernel'])!=1 or t['equalizer_kernel'][0]!=1:raise ValueError('Actual EQoff must retain G=1')
     arrays[k,j,i]=np.stack((y.real,y.imag));pair.append((m,t))
     records.append(dict(sample_id=pid,RX=int(rx[i]),day=int(day[i]),scene=scene,draw=k,sign=int(sign),state=m['state_end'],lock=m['receiver_lock_status'],quality_snr_db=m['quality_snr_db'],signal_to_added_noise_db=m['signal_to_added_noise_db'],agc_gain_db=m['agc_gain_db'],agc_limited=m['agc_limited'],output_frequency_hz=m['output_frequency_hz'],residual_kernel_energy=m['residual_kernel_energy'],boundary=m['boundary']))
    if not np.array_equal(pair[0][1]['pre_agc_noise'],-pair[1][1]['pre_agc_noise']):raise ValueError('Primitive sign not preserved by RX/IQ/G noise processing')
  return arrays,records
 def audit(self):return dict(channel_file=self.ch.__file__,residual_file=self.residual.__file__,actual_configs={k:asdict(v) for k,v in self.configs.items()},session_count=len(self.sessions),session_hardware=self.sessions,observations=self.calls,AGC_per_view_recomputed=True,noise_marginals='z and -z both standard normal',white_input_quality_proxy=True,true_total_SNR_known=False,formal_test_changed=False)
