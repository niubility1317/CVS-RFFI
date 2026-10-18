"""Focused marginal/stratum/physical-chain and negative checks;synthetic IQ only."""
import json,sys
from pathlib import Path
import numpy as np
from .sampling import shadow_draws,apply_shadow

def run(require_torch=False):
 groups=np.array([0]*31+[1]*17+[2]*9+[3]*35+[8]*36);negative=0
 for mode in ('IID','LHS'):
  u,rank=shadow_draws(mode,groups,20261018941);u2,rank2=shadow_draws(mode,groups,20261018941)
  assert np.array_equal(u,u2) and np.array_equal(rank,rank2) and np.all((u>0)&(u<1))
  if mode=='LHS':
   for g in np.unique(groups):
    ix=np.flatnonzero(groups==g);assert sorted(np.floor(u[ix]*len(ix)).astype(int).tolist())==list(range(len(ix)))
 # Fresh random jitters/permutations;the full marginal,including both tails,remains covered.
 values=[]
 for seed in range(1024):values.append(shadow_draws('LHS',np.zeros(16,dtype=int),seed)[0][0])
 values=np.array(values);assert abs(values.mean()-.5)<.03 and abs(values.var()-1/12)<.01 and values.min()<.01 and values.max()>.99
 class Latent:
  def __init__(self,z):self.z=z
 a=[Latent(float(i)) for i in range(len(groups))];old=[v.z for v in a];records=apply_shadow('NATIVE',a,groups,9);assert [v.z for v in a]==old and all(x['replacement_applied'] is False for x in records)
 records=apply_shadow('LHS',a,groups,9);assert len(records)==128 and all(isinstance(v.z,float) and np.isfinite(v.z) for v in a);json.dumps(records,allow_nan=False)
 for mode,g in [('OTHER',groups),('LHS',np.array([-1])),('LHS',np.array([9])),('LHS',np.array([.5])),('LHS',np.array([]))]:
  try:shadow_draws(mode,g,1)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid sampling input accepted')
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/adv3b02_xuc/code'))
 from leo_practical.channel import Config,receiver_for_session,normalize_input
 from leo_practical.residual import ResidualChannel
 rng=np.random.default_rng(20261018942);x=rng.normal(size=256)+1j*rng.normal(size=256);x,_=normalize_input(x);physical=0
 for scene in ('practical_low_suburban','practical_mid_urban','practical_low_urban'):
  cfg=Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');hw=receiver_for_session(cfg,20261016423,'test:synthetic')
  for seed in (123,789):
   native=ResidualChannel(cfg,seed,hw);same=ResidualChannel(cfg,seed,hw);apply_shadow('NATIVE',[same.latent],np.array([0]),1);y,m=native.process(x);yy,mm=same.process(x);physical+=2;assert np.array_equal(y,yy) and m==mm
   test=ResidualChannel(cfg,seed,hw);test.latent.z=.75;expected,em=test.process(x);test2=ResidualChannel(cfg,seed,hw);apply_shadow('LHS',[test2.latent],np.array([0]),37);z=test2.latent.z;reference=ResidualChannel(cfg,seed,hw);reference.latent.z=z;expected,em=reference.process(x);actual,am=test2.process(x);physical+=3;assert np.array_equal(expected,actual) and em==am and am['quality_source']=='white_input_expected_power_proxy' and not am['channel_equalization_applied'] and am['added_noise_variance_reference']==.001
 from .channel import Practical
 n=128;raw=rng.normal(size=(n,2,256)).astype(np.float32);meta=dict(eq_i=[1]*n,rx_i=[(1,3,4,6,8)[i%5] for i in range(n)],day_i=[1+i%3 for i in range(n)],sig_i=list(range(n)),base_index=list(range(n)));outputs={};native_records=None
 for mode in ('NATIVE','IID','LHS'):
  ch=Practical(mode);out,rr=ch.process_numpy(raw,meta,21,4441);physical+=n;outputs[mode]=out
  assert out.shape==raw.shape and np.isfinite(out).all() and len(rr)==n and ch.packets==n and len(ch.sessions)==15 and ch.summary()['dropped']==0
  assert all(v['quality_source']=='white_input_expected_power_proxy' and v['added_noise_variance_reference']==.001 and not v['channel_equalization_applied'] for v in rr)
  if mode=='NATIVE':native_records=rr;assert all(v['shadow_sampling']['shadow_used']==v['shadow_sampling']['shadow_original'] for v in rr)
  if mode=='LHS':
   for key in set(v['shadow_sampling']['group'] for v in rr):
    a=[v['shadow_sampling'] for v in rr if v['shadow_sampling']['group']==key];assert sorted(v['stratum'] for v in a)==list(range(len(a)))
 assert np.any(outputs['LHS']!=outputs['IID']) and np.any(outputs['IID']!=outputs['NATIVE'])
 torch_adapter=False
 try:import torch
 except ModuleNotFoundError:
  if require_torch:raise
 else:
  from experiments.cvs_clock_factorial30.channel import Practical as Original
  x=torch.tensor(raw.tolist(),dtype=torch.float32);mt={k:torch.tensor(v) for k,v in meta.items()};orig,_=Original('train').apply(x,mt,21,4441);full,rr=Practical('NATIVE').apply(x,mt,21,4441);physical+=2*n;assert orig.equal(full);torch_adapter=True
 if require_torch and not torch_adapter:raise ValueError('RealTorch adapter check required on launchruntime')
 return dict(status='PASS',stratified_dimensions=1,conditional_groups=9,synthetic_marginal_replicates=1024,negative_cases=negative,native_bitwise_equal=True,whole_native_process_after_shadow_replacement=True,no_shadow_clipping=True,synthetic_physical_channel_calls=physical,full_batch_all128_retained=True,source15sessions_preserved=True,Torch_adapter_bitwise_checked=torch_adapter,real_IQ=0,CP=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
