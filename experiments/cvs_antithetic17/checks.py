"""Focused protocol negatives and native physical formula replay;synthetic only."""
import json
import numpy as np
from . import design as d
from .coupled import ExactNoise,process,three,variance,tensor_input
def run():
 import leo_practical.channel as ch
 import leo_practical.residual as residual
 from experiments.cvs_rdrop_sinc_units.startup import launch_once
 class Child:
  pid=17
  def poll(self):return None
 base=dict(cwd='/synthetic',argv=['python','worker'],gpu='0');observed=[];created=[]
 def create():created.append(Child());return created[-1]
 child,receipt=launch_once(create,base,lambda r:observed.append(dict(r)),reader=lambda pid:dict(base,pid=pid,start_ticks=42))
 assert child is created[0] and len(created)==1 and [x['status'] for x in observed]==['SUBMITTED','OBSERVED'] and receipt['start_ticks']==42
 failed=[]
 try:launch_once(create,base,lambda r:failed.append(dict(r)),reader=lambda pid:None,timeout=0)
 except TimeoutError:pass
 else:raise AssertionError('Expected observation timeout')
 assert len(created)==2 and [x['status'] for x in failed]==['SUBMITTED','OBSERVATION_FAILED'] and failed[0]['pid']==failed[1]['pid']==17
 a=(np.arange(15360,dtype=np.float32)/15361).reshape(30,2,256);b=tensor_input(a)
 assert b.tolist()==a.tolist() and str(b.dtype)=='torch.float32' and tuple(b.shape)==(30,2,256)
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError):negative+=1;return
  raise AssertionError('Expected rejection')
 for a in (np.zeros((255,2)),np.full((256,2),np.nan),np.zeros((256,3))):bad(lambda a=a:ExactNoise(a))
 z=ExactNoise(np.zeros((256,2)));bad(lambda:z.normal(size=128));z.normal(size=(256,2));bad(lambda:z.normal(size=(256,2)))
 x=np.exp(2j*np.pi*np.arange(256)*.03125)+.1*np.cos(np.arange(256)*.1);calls=0;max_error=0.
 for scene in d.SCENES:
  cfg=ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');hw=ch.receiver_for_session(cfg,d.RECEIVER_SEED,'rx1:day1:eq1')
  for seed in (21,22,23,24):
   native,meta=residual.ResidualChannel(cfg,seed,hw).process(x);e1=np.random.default_rng(ch.stable_seed(seed,'noise')).normal(size=(256,2));e2=np.random.default_rng(ch.stable_seed(seed,'other')).normal(size=(256,2));v=three(ch,residual,cfg,hw,x,seed,e1,e2);calls+=4;err=float(np.max(abs(v[0][0]-native)));max_error=max(max_error,err);assert err==0
 rng=np.random.default_rng(123);g=rng.normal(size=(24,17));v=variance(g@g.T,8)
 for name,k in [('independent',1),('antithetic',2)]:assert abs(v[name]-np.var((g[0::3]+g[k::3])*.5,axis=0,ddof=1).sum())<1e-12
 for a in (np.zeros((8,8)),np.full((24,24),np.nan),np.zeros((24,24))):bad(lambda a=a:variance(a,8))
 if d.config()['expected_NN']!=25922 or d.config()['training_updates'] or d.config()['target_access'] or d.config()['source_RX']!=[1,3,4,6,8]:raise ValueError('Fixed source-only budget required')
 return dict(status='PASS',protocol_negatives=negative,startup_fake_process_cases=2,tensor_buffer_bridge_exact=True,synthetic_native_channel_calls=calls,max_native_replay_error=max_error,synthetic_grad_Gram_recount_error_lt=1e-12,real_IQ=0,CP=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
