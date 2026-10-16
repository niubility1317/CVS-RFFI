"""Full practical residual;only cross-view primitive noise correlation changes."""
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch
from . import design as d
from experiments.cvs_crossrx_satmix20.channel import imports
from experiments.cvs_f_noise_pair_probe.noise import observe
KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
class TrainingChannel:
 def __init__(self):
  self.ch,self.residual=imports();self.sessions={};self.calls=0;self.physical_presentations=0;self.noise_observations=0;self.coverage=Counter()
  self.configs={s:self.ch.Config(fs_hz=25e6,fc_hz=2462e6,scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.SCENES}
 def apply(self,x,meta,*,step,mode):
  if mode not in ('INDEP','ANTI') or step not in range(1,4441) or x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Registered finite visibleL training IQ256/step/mode required')
  if not all(k in meta and len(meta[k])==len(x) for k in KEYS):raise ValueError('Actual physical source metadata required')
  vals={k:meta[k].detach().cpu().tolist() for k in KEYS};ids=list(zip(*(vals[k] for k in KEYS)))
  if any(p[0]!=1 or p[1] not in d.FIT or p[2] not in (1,2,3) for p in ids):raise ValueError('Only actual fitRX/day/eq sourceL allowed')
  epoch=(step-1)//222+1;ns=f'FcrossRX/noisepair/train/e{epoch}/step{step}/L';raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);outputs=[[],[]];records=[]
  for i,pid in enumerate(ids):
   scene=str(np.random.default_rng(self.ch.stable_seed(d.CHANNEL_SEED,ns,'scene',pid)).choice(d.SCENES));cfg=self.configs[scene];seed=self.ch.stable_seed(d.CHANNEL_SEED,ns,scene,pid);session=f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}';hw=self.ch.receiver_for_session(cfg,d.RECEIVER_SEED,session);hardware=asdict(hw)
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('RX hardware changed within RX/day/eq session')
   z0=np.random.default_rng(self.ch.stable_seed(d.NOISE_SEED,ns,'noise0',pid)).normal(size=(256,2));z1=-z0 if mode=='ANTI' else np.random.default_rng(self.ch.stable_seed(d.NOISE_SEED,ns,'noise1',pid)).normal(size=(256,2))
   normalized,input_meta=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);pair=[]
   for j,z in enumerate((z0,z1)):
    y,m,t=observe(self.residual,cfg,seed,hw,normalized,z)
    if m['channel_equalization_applied'] or len(t['equalizer_kernel'])!=1 or t['equalizer_kernel'][0]!=1:raise ValueError('Declared EQoff must retain G=1')
    pair.append((m,t));outputs[j].append(np.stack((y.real,y.imag)));records.append(dict(physical_id=list(pid),RX=pid[1],day=pid[2],step=step,epoch=epoch,view=j,scene=scene,session=session,state=m['state_end'],lock=m['receiver_lock_status'],quality_snr_db=m['quality_snr_db'],signal_to_added_noise_db=m['signal_to_added_noise_db'],output_frequency_hz=m['output_frequency_hz'],residual_scale=m['residual_scale'],residual_kernel_energy=m['residual_kernel_energy'],residual_kernel_length=m['residual_kernel_length'],numerical_filter_latency_samples=m['numerical_filter_latency_samples'],boundary=m['boundary'],agc_gain_db=m['agc_gain_db'],agc_limited=m['agc_limited'],added_noise_variance_reference=m['added_noise_variance_reference'],measured_added_noise_power_pre_agc=m['measured_added_noise_power_pre_agc'],phase_tracking_applied=m['phase_tracking_applied'],iq_compensation_applied=m['iq_compensation_applied'],EQ=False,G=1,nonnoise_latents_shared=True,AGC_recomputed_per_view=True))
    self.coverage[(pid[1],pid[2],scene,m['state_end'],m['receiver_lock_status'],j)]+=1
   common=('state_end','receiver_lock_status','quality_snr_db','residual_scale','output_frequency_hz','phase_tracking_applied','iq_compensation_applied','residual_kernel_energy','residual_kernel_length','boundary')
   if any(pair[0][0][k]!=pair[1][0][k] for k in common) or not np.array_equal(pair[0][1]['pre_agc_signal'],pair[1][1]['pre_agc_signal']) or not np.array_equal(pair[0][1]['residual_kernel'],pair[1][1]['residual_kernel']):raise ValueError('Propagation or receiver state changed across noise pair')
   if mode=='ANTI' and not np.array_equal(pair[0][1]['pre_agc_noise'],-pair[1][1]['pre_agc_noise']):raise ValueError('Primitive sign relation not retained by actual RX/IQ/G')
  values=[]
  for out in outputs:
   a=np.ascontiguousarray(out,dtype=np.float32);values.append(torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device))
  self.calls+=1;self.physical_presentations+=len(x);self.noise_observations+=2*len(x)
  return values,records
 def audit(self):return dict(channel_file=self.ch.__file__,residual_file=self.residual.__file__,actual_configs={k:asdict(v) for k,v in self.configs.items()},session_count=len(self.sessions),session_hardware=self.sessions,calls=self.calls,physical_presentations=self.physical_presentations,noise_observations=self.noise_observations,hardware_session_not_TX=True,original_AGC_each_view=True,true_total_SNR_known=False,formal_test_unchanged=True)
