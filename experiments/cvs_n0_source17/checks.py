"""Focused source-role and coupled-N0 tests;synthetic physics only."""
import sys
from pathlib import Path
import numpy as np
from . import design as d
from .coupled import observations

def run():
 code=Path(__file__).resolve().parents[2]/'experiments/adv3b02_xuc/code'
 if code.exists():sys.path.insert(0,str(code))
 else:
  from experiments.cvs_phase1_overlay.model import native_modules
  native_modules()
 rng=np.random.default_rng(928);x=rng.normal(size=(5,2,256));before=x.copy();ids=['opaque'+str(i) for i in range(5)];rx=list(d.FIT);day=[1,2,3,1,2];state=np.random.get_state();cases=0;maxsig=maxnoise=0.
 for scene in d.SCENES:
  out,rr=observations(x,ids,rx,day,scene);order=[4,1,0,3,2];rev,_=observations(x[order],[ids[i] for i in order],[rx[i] for i in order],[day[i] for i in order],scene)
  for key in d.VARIANTS:assert np.array_equal(out[key][order],rev[key]) and out[key].shape==(5,2,256)
  for r in rr:
   for key in d.VARIANTS:
    v=r['variants'][key];maxsig=max(maxsig,v['fixed_quality_signal_error'] or 0);maxnoise=max(maxnoise,v['fixed_quality_noise_error'] or 0)
   assert abs(.5*(r['variants']['N0_half']['noise_variance_reference']+r['variants']['N0_1p5']['noise_variance_reference'])-r['variants']['nominal']['noise_variance_reference'])<1e-15
  cases+=len(rr)
 assert np.array_equal(before,x);after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
 bad=[lambda:observations(x[:,:,:255],ids,rx,day,d.SCENES[0]),lambda:observations(x*float('nan'),ids,rx,day,d.SCENES[0]),lambda:observations(x*0,ids,rx,day,d.SCENES[0]),lambda:observations(x,ids[:-1],rx,day,d.SCENES[0]),lambda:observations(x,ids,rx,day,'clean'),lambda:observations(x,['same']*5,rx,day,d.SCENES[0]),lambda:observations(x,ids,[0,3,4,6,8],day,d.SCENES[0]),lambda:observations(x,ids,rx,[0,2,3,1,2],d.SCENES[0]),lambda:observations(x,list(range(5)),rx,day,d.SCENES[0]),lambda:observations(x,ids,rx[:-1],day,d.SCENES[0])]
 for f in bad:
  try:f()
  except ValueError:pass
  else:raise AssertionError('Protocol-negative not rejected')
 assert d.config()['expected_NN']==(1+3*5)*6300+2 and d.config()['training_updates']==0 and not d.config()['automatic_training']
 return dict(status='PASS',protocol_negatives=len(bad),synthetic_physical_pairs=cases,physical_processing_calls=cases*5*2,fixed_quality_signal_max_error=maxsig,fixed_quality_noise_max_error=maxnoise,mean_reference_N0_exact=True,input_and_global_RNG_unchanged=True,batch_permutation_invariant=True,real_IQ=0,CP=0,NN=0,training_updates=0)
if __name__=='__main__':print(run())
