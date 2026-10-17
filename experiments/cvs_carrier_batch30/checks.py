"""Native equivalence/RNG and marginal-law tests, without real data or a model."""
import copy,importlib.util,random,sys
from pathlib import Path
import numpy as np
import torch
from .carrier import install,CarrierPolicy
from experiments.cvs_fullsource_concat20.supcon import penalty
def native_module():
 path=Path(__file__).resolve().parents[2]/'experiments/adv3b02_xuc/code/DataAugmentation.py'
 if not path.is_file():
  import DataAugmentation
  return DataAugmentation
 spec=importlib.util.spec_from_file_location('_carrier_native_checks',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def seeds(n):torch.manual_seed(n);random.seed(n);np.random.seed(n)
def state():return (torch.get_rng_state().clone(),random.getstate(),np.random.get_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def equal_state(a,b):return torch.equal(a[0],b[0]) and a[1]==b[1] and a[2][0]==b[2][0] and np.array_equal(a[2][1],b[2][1]) and a[2][2:]==b[2][2:] and len(a[3])==len(b[3]) and all(torch.equal(x,y) for x,y in zip(a[3],b[3]))
def run(device='cpu'):
 torch.set_num_threads(2);m=native_module();configuration=dict(p_dac=0,p_pa=.14,enable_class_signature=False,p_time_shift=.35,max_time_shift=32,p_amp_scale=.45,amp_min=.9,amp_max=1.1,p_phase_rot=.45,p_cfo=.35,cfo_max=.0004,p_phase_noise=.3,phase_noise_sigma_max=.006,p_awgn=.4,snr_min_db=20,snr_max_db=36,p_multipath=.18,mp_taps_min=2,mp_taps_max=4,mp_delay_max=4,p_dc_offset=.3,dc_offset_max=.02,p_bandedge_taper=.25,taper_alpha_min=.02,taper_alpha_max=.1)
 seeds(2026101801);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;template=m.RFFIAugmentor(**configuration);records=[]
 for seed in (2026101811,2026101812,2026101813,2026101814,2026101815,2026101816,2026101817,2026101818):
  native,batch,packet=(copy.deepcopy(template) for _ in range(3));pb=install(batch,'BATCH',2026101821);pp=install(packet,'PACKET',2026101821)
  seeds(seed);a=native(x,labels=y,no_pa=False);sa=state();seeds(seed);b=batch(x,labels=y,no_pa=False);sb=state();seeds(seed);p=packet(x,labels=y,no_pa=False);sp=state()
  assert torch.equal(a,b),'Native batch selector output changed'
  assert equal_state(sa,sb) and equal_state(sa,sp),'Global RNG changed outside carrier selectors'
  assert p.shape==a.shape and torch.isfinite(p).all() and pb.phase_calls==pb.cfo_calls==pp.phase_calls==pp.cfo_calls==1
  records.append(dict(seed=seed,batch_bitwise_native=True,global_RNG_equal=True,packet_changed=not torch.equal(a,p),batch_record=pb.records[0],packet_record=pp.records[0]))
 assert sum(r['packet_changed'] for r in records)>0
 sampler={};z=torch.ones(128,1,dtype=torch.complex64)
 for mode in ('BATCH','PACKET'):
  owner=m.RFFIAugmentor(p_phase_rot=.45,p_cfo=.35,cfo_max=.0004);policy=CarrierPolicy(owner,mode,2026101831,m.__dict__);seeds(2026101832)
  for i in range(2048):policy.cfo(policy.phase(z))
  summary=policy.summary();sampler[mode]=summary
  for name,p in (('phase',.45),('cfo',.35)):
   q=summary[name];rate=q['selected']/summary['packets'];pair=q['selected_pairs']/q['pair_denominator'];assert abs(rate-p)<.04
   if mode=='BATCH':assert q['nontrivial_batches']==0 and abs(pair-rate)<1e-12
   else:assert q['nontrivial_batches']==2048 and abs(pair-p*p)<.02
 # The existing batch-standardized source contrast is not a separable sample risk.
 seeds(2026101841);h=torch.randn(8,4,8)*.2;h[:4,0]+=1.;h[4:,1]+=1.;labels=torch.arange(8)//4;rx=torch.tensor([1,3,4,6,1,3,4,6]);scores=[];means=[]
 for bits in range(256):
  mask=torch.tensor([(bits>>i)&1 for i in range(8)],dtype=torch.bool);rot=h.clone();rot[:,0::2]=-h[:,1::2];rot[:,1::2]=h[:,0::2];v=torch.where(mask[:,None,None],rot,h);score,_=penalty({'time_down':v,'t2.pool':v},labels,rx,'CROSS_RX');scores.append(float(score));means.append(float(mask.float().mean()))
 shared=(scores[0]+scores[-1])/2;independent=sum(scores)/256;assert abs(shared-independent)>.05 and sum(means)/256==.5
 negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,KeyError,TypeError):negative+=1
  else:raise AssertionError('Invalid carrier policy accepted')
 reject(lambda:CarrierPolicy(template,'INVALID',1,m.__dict__));double=copy.deepcopy(template);install(double,'BATCH',1);reject(lambda:install(double,'BATCH',1));bad=copy.deepcopy(template);bad.p_phase_rot=1.1;policy=CarrierPolicy(bad,'PACKET',1,m.__dict__);reject(lambda:policy.phase(z));policy=CarrierPolicy(template,'PACKET',1,m.__dict__);reject(lambda:policy.cfo(z))
 return dict(status='PASS',input_device=device,global_CUDA_RNG_checked=torch.cuda.is_available(),negative_cases=negative,native_equivalence_cases=records,selector_law=sampler,existing_contrast_counterexample=dict(shared_expected_loss=shared,packet_expected_loss=independent,marginal_mask_mean=.5,performance_claim=False),synthetic_full_augmented_packets=8*3*128,selector_only_complex_vectors=2*2048*128,toy_feature_loss_evaluations=256,real_IQ=0,model_NN=0,teacher_NN=0,optimizer_updates=0,target_access=False)
if __name__=='__main__':
 import json
 print(json.dumps(run(),allow_nan=False))
