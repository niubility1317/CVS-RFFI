"""Focused dependency,latency and input-negative tests;no real IQ/CP/NN."""
import sys
from pathlib import Path
import numpy as np
from . import design as d
from .coupled import triplet

def run():
 code=Path(__file__).resolve().parents[2]/'experiments/adv3b02_xuc/code'
 if code.exists():sys.path.insert(0,str(code))
 else:
  from experiments.cvs_phase1_overlay.model import native_modules
  native_modules()
 from leo_practical.channel import Config
 from leo_practical.equalization import design_equalizer
 rng=np.random.default_rng(928);x=rng.normal(size=(4,2,256));old=x.copy();ids=['opaque'+str(i) for i in range(4)];rx=[1,3,4,8];day=[1,2,3,1];state=np.random.get_state();pairs=0;maxerror=0.
 for scene in d.SCENES:
  out,records=triplet(x,ids,rx,day,scene);order=[3,1,0,2];rev,_=triplet(x[order],[ids[i] for i in order],[rx[i] for i in order],[day[i] for i in order],scene)
  for k in out:assert np.array_equal(rev[k],out[k][order])
  for r in records:
   for v in r['variants'].values():maxerror=max(maxerror,v['noise_filter_max_error'])
  pairs+=len(records)
 assert np.array_equal(x,old)
 after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
 from dataclasses import replace
 from leo_practical.channel import receiver_for_session
 from leo_practical.residual import ResidualChannel
 base=Config(fs_hz=25e6,scenario=d.SCENES[-1],processing_route='residual',reference_snr_db=-80.)
 hw=receiver_for_session(base,91,'synthetic-session');z=x[0,0]+1j*x[0,1]
 a,ma=ResidualChannel(base,17,hw).process(z)
 b,mb=ResidualChannel(replace(base,equalization_enabled=True,equalizer_delay_samples=24),17,hw).process(z)
 assert ma['receiver_lock_status']=='unlocked' and not mb['channel_equalization_applied'] and np.array_equal(a,b)
 for delay in (24,64):
  h=np.zeros(49,complex);h[24]=1.
  cfg=Config(fs_hz=25e6,equalization_enabled=True,equalizer_delay_samples=delay,channel_estimation_nmse_db=-300.)
  g,m=design_equalizer(h,cfg,np.random.default_rng(21),.001,1.)
  assert int(np.argmax(np.abs(np.convolve(h,g))))==delay
  assert np.max(np.abs(np.fft.fft(g,2048)))<=10+1e-9
 bad=[lambda:triplet(x[:,:,:255],ids,rx,day,d.SCENES[0]),lambda:triplet(x*float('nan'),ids,rx,day,d.SCENES[0]),lambda:triplet(x*0,ids,rx,day,d.SCENES[0]),lambda:triplet(x,ids[:-1],rx,day,d.SCENES[0]),lambda:triplet(x,ids,rx,day,'clean'),lambda:triplet(x,['same']*4,rx,day,d.SCENES[0]),lambda:triplet(x,ids,[0,3,4,8],day,d.SCENES[0]),lambda:triplet(x,ids,rx,[0,2,3,1],d.SCENES[0]),lambda:triplet(x,[1,2,3,4],rx,day,d.SCENES[0]),lambda:triplet(x,ids,rx[:-1],day,d.SCENES[0])]
 for f in bad:
  try:f()
  except ValueError:pass
  else:raise AssertionError('Negative not rejected')
 return dict(status='PASS',protocol_negatives=len(bad),synthetic_coupled_triplets=pairs,noise_filter_max_error=maxerror,known_identity_H_total_delay_peaks=[24,64],forced_unlocked_control_exact=True,batch_permutation_invariant=True,input_and_global_RNG_unchanged=True,real_IQ=0,CP=0,NN=0,training_updates=0)
if __name__=='__main__':print(run())
