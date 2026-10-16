"""One original full draw and its fixed-receiver no-added-noise intervention."""
from dataclasses import asdict
import numpy as np
import torch
from . import design as d
from experiments.cvs_crossrx_satmix20.channel import imports
from experiments.cvs_f_noise_pair_probe.noise import observe

def split_control(y,meta,traces):
 if meta['channel_equalization_applied'] or len(traces['equalizer_kernel'])!=1 or traces['equalizer_kernel'][0]!=1:raise ValueError('Original EQoffG1 required')
 signal=traces['pre_agc_signal'];noise=traces['pre_agc_noise'];gain=10**(meta['agc_gain_db']/20)
 if signal.shape!=noise.shape or signal.shape!=(256,) or not np.isfinite(signal).all() or not np.isfinite(noise).all() or not np.isfinite(gain) or gain<=0:raise ValueError('Finite original signal/noise/same AGC scalar required')
 reconstructed=np.asarray((signal+noise)*gain,dtype=np.complex64)
 if not np.array_equal(np.asarray(y,dtype=np.complex64),reconstructed):raise ValueError('Original full IQ not reproduced by native component sum andsameAGC')
 control=np.asarray(signal*gain,dtype=np.complex64)
 if not np.isfinite(control).all():raise FloatingPointError('Nonfinite controlled IQ')
 return control
class Channel:
 def __init__(self):
  self.ch,self.residual=imports();self.sessions={};self.calls=0
  self.cfg=self.ch.Config(fs_hz=25e6,fc_hz=2462e6,scenario='practical_low_urban',processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
 def batch(self,x,ids,rx,day):
  raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);full=[];control=[];physics=[]
  if raw.shape!=(len(ids),2,256) or not np.isfinite(raw).all() or len(set(ids))!=len(ids) or any(int(r) not in d.FIT for r in rx) or any(int(t) not in (1,2,3) for t in day):raise ValueError('Only unique finite actualfitL IQ256 required')
  for i,pid in enumerate(ids):
   ns='FcrossRX/postAs_noise_control/fitL/v1';seed=self.ch.stable_seed(d.STREAMS['channel'],ns,'practical_low_urban',pid);session=f'rx{int(rx[i])}:day{int(day[i])}:eq1';hw=self.ch.receiver_for_session(self.cfg,d.STREAMS['receiver'],session);hardware=asdict(hw)
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('RX hardware changed within actualRX/day/eq session')
   normalized,input_meta=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);z=np.random.default_rng(self.ch.stable_seed(d.STREAMS['noise'],ns,pid)).normal(size=(256,2));y,m,t=observe(self.residual,self.cfg,seed,hw,normalized,z);c=split_control(y,m,t);self.calls+=1
   full.append(np.stack((y.real,y.imag)));control.append(np.stack((c.real,c.imag)))
   physics.append(dict(physical_id=pid,RX=int(rx[i]),day=int(day[i]),scene='practical_low_urban',state=m['state_end'],lock=m['receiver_lock_status'],session=session,quality_snr_db=m['quality_snr_db'],signal_to_added_noise_db=m['signal_to_added_noise_db'],output_frequency_hz=m['output_frequency_hz'],residual_kernel_energy=m['residual_kernel_energy'],residual_kernel_length=m['residual_kernel_length'],propagation_kernel_length=len(t['propagation_kernel']),numerical_filter_latency_samples=m['numerical_filter_latency_samples'],boundary=m['boundary'],noise_boundary=m['noise_boundary'],agc_gain_db=m['agc_gain_db'],agc_limited=bool(m['agc_limited']),added_noise_variance_reference=m['added_noise_variance_reference'],measured_added_noise_power_pre_agc=m['measured_added_noise_power_pre_agc'],phase_tracking_applied=bool(m['phase_tracking_applied']),iq_compensation_applied=bool(m['iq_compensation_applied']),EQ=False,G=1,original_full_IQ_reconstructed_exact=True,same_receiver_compensation_AGC=True,control_new_noise_power=0.,receiver_state_reference='original_full_observation',true_total_SNR_known=False))
  def tensor(a):
   v=np.ascontiguousarray(a,dtype=np.float32);return torch.frombuffer(bytearray(v.tobytes()),dtype=torch.float32).reshape(v.shape)
  return tensor(full),tensor(control),physics
 def audit(self):return dict(channel_file=self.ch.__file__,residual_file=self.residual.__file__,actual_config=asdict(self.cfg),full_physical_draws=self.calls,session_count=len(self.sessions),session_hardware=self.sessions,control_receiver_and_AGC_recomputed=False,control_only_original_added_noise_removed=True,unknown_ground_TX_RX_untouched=True,formal_test_changed=False,true_total_SNR_known=False)
