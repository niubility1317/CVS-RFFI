"""Narrow protocol negatives and actual channel numerical examples;no real data/model."""
import copy,json,sys
import numpy as np
from . import design as d
from .core import propagated_quality,compare,summarize

def run():
 from leo_practical.channel import Config,receiver_for_session
 n=0
 def bad(f):
  nonlocal n
  try:f()
  except (ValueError,TypeError,KeyError):n+=1
  else:raise AssertionError('Invalid scope accepted')
 for k,v in [('L',5040),('source_RX',[3,4,6,8]),('source_RX_holdout',1),('source_days',[0,1]),('scenes',['practical_low_urban']),('reference_snr_db',35.),('EQ',True),('target_access',True),('network_forwards',1),('checkpoint_loads',1),('U_truth_read',True),('V_statistically_used',True),('automatic_training',True),('optimizer_updates',1),('same_raw_same_latent',False)]:
  c=d.config();c[k]=v;bad(lambda c=c:d.validate(c))
 cfg=Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=d.SCENES[0],processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');hw=receiver_for_session(cfg,d.RECEIVER_SEED,'synthetic')
 for x in (np.zeros(255,complex),np.full(256,np.nan+1j),np.zeros((2,256),complex)):
  bad(lambda x=x:compare(x,cfg,1,hw))
 t=np.arange(256);x=np.exp(.01j*t);assert abs(propagated_quality(x,np.ones(1),.001,'reflect')-30)<1e-12
 q=propagated_quality(x,np.array([1.,-1.])/np.sqrt(2),.001,'reflect');assert q<0
 cases=[]
 for scene in d.SCENES:
  cfg=Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
  for seed in (31,97):
   for x in (np.exp(.01j*t),np.exp(.9j*t)+.5*np.exp(-.4j*t)):
    a=compare(x,cfg,seed,hw);b=compare(x,cfg,seed,hw);assert a==b and a['same_input_latent'];cases.append(a)
 bad(lambda:summarize([]))
 return dict(status='PASS',protocol_negatives=n,analytical_power_cases=2,actual_channel_pairs=24,synthetic_realizations=48,real_IQ=0,checkpoint_loads=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':
 sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'));print(json.dumps(run()))
