"""Synthetic-only physical dependence/RNG/role checks; no model/CP/source IQ."""
import copy,json,random
import numpy as np
import torch
from .channel import GroupedPractical,mix_propagation
from experiments.cvs_crossrx_satmix20.channel import PairedChannel

def run():
 torch.set_num_threads(2);base=PairedChannel();B=12;x=torch.randn(B,2,256,generator=torch.Generator().manual_seed(202610172001));x=x/torch.sqrt(x.square().sum(1).mean(1))[:,None,None];saved=x.clone();rx=torch.tensor([1,3,4,8]*3);meta=dict(rx_i=rx,day_i=torch.ones(B,dtype=torch.long),eq_i=torch.ones(B,dtype=torch.long),sig_i=torch.arange(B),base_index=torch.arange(B),tx_i=torch.arange(B)%6);ns='train/epoch6/step1111';state=torch.get_rng_state();pystate=random.getstate();npstate=np.random.get_state();reports={};calls=0
 for policy in ('IID','SCENE_GROUP','PROP_GROUP'):
  view,detail=GroupedPractical(policy).apply(x,meta,namespace=ns);reports[policy]=(view,detail);calls+=B
  assert torch.equal(saved,x) and torch.equal(state,torch.get_rng_state()) and pystate==random.getstate()
  after=np.random.get_state();assert npstate[0]==after[0] and np.array_equal(npstate[1],after[1]) and npstate[2:]==after[2:]
 original,original_meta=base.apply(x,meta,namespace=ns);calls+=B
 assert torch.equal(original,reports['IID'][0])
 for expected,actual in zip(original_meta,reports['IID'][1]):assert all(actual[k]==v for k,v in expected.items())
 s=reports['SCENE_GROUP'][1];p=reports['PROP_GROUP'][1];assert [r['scene'] for r in s]==[r['scene'] for r in p]
 for r in (1,3,4,8):
  q=[z for z in p if z['RX']==r];assert len({z['environment_seed'] for z in q})==len({z['state'] for z in q})==len({z['kernel_energy'] for z in q})==1 and len({z['packet_seed'] for z in q})==3 and len({z['noise_stream_seed'] for z in q})==3
 assert all(a['packet_residual_hz']==b['packet_residual_hz'] and a['packet_phase']==b['packet_phase'] and a['packet_tracking_z']==b['packet_tracking_z'] and a['packet_coarse_error_hz']==b['packet_coarse_error_hz'] and a['noise_stream_seed']==b['noise_stream_seed'] for a,b in zip(s,p))
 altered=dict(meta,tx_i=(meta['tx_i']+3)%6);again,again_meta=GroupedPractical('PROP_GROUP').apply(x,altered,namespace=ns);calls+=B;assert torch.equal(again,reports['PROP_GROUP'][0]) and again_meta==p
 order=torch.arange(B-1,-1,-1);reordered,rr=GroupedPractical('PROP_GROUP').apply(x[order],{k:v[order] for k,v in meta.items()},namespace=ns);calls+=B;assert torch.equal(reordered[order],reports['PROP_GROUP'][0])
 fresh,fr=GroupedPractical('PROP_GROUP').apply(x,meta,namespace='train/epoch6/step1112');calls+=B;assert not torch.equal(fresh,reports['PROP_GROUP'][0]) and all(a['environment_seed']!=b['environment_seed'] for a,b in zip(p,fr))
 eval_meta=dict(meta,rx_i=torch.full((B,),6,dtype=torch.long))
 for scene in base.configs:
  expected,em=base.apply(x[:2],{k:v[:2] for k,v in eval_meta.items()},namespace='source_eval/RX6/final20',scene=scene);calls+=2
  for policy in ('IID','SCENE_GROUP','PROP_GROUP'):
   v,vm=GroupedPractical(policy).apply(x[:2],{k:z[:2] for k,z in eval_meta.items()},namespace='source_eval/RX6/final20',scene=scene);calls+=2;assert torch.equal(v,expected) and vm==em
  cfg=base.configs[scene];receiver=base.ch.receiver_for_session(cfg,20261016102,'rx1:day1:eq1');seed=202610172002
  a=base.residual.ResidualChannel(cfg,seed,receiver);b=base.residual.ResidualChannel(cfg,seed,receiver);e=base.residual.ResidualChannel(cfg,seed,receiver);before={k:copy.deepcopy(v.bit_generator.state) for k,v in a.latent.rngs.items()};mix_propagation(a,e)
  assert all(a.latent.rngs[k].bit_generator.state==v for k,v in before.items())
  wave=np.asarray(x[0,0].double().tolist())+1j*np.asarray(x[0,1].double().tolist());y,ym,yt=a.process(wave,return_traces=True);z,zm,zt=b.process(wave,return_traces=True);calls+=2;assert np.array_equal(y,z) and ym==zm and all(np.array_equal(yt[k],zt[k]) for k in yt)
 negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except ValueError:negative+=1;return
  raise AssertionError('Invalid physics/role/namespace accepted')
 reject(lambda:GroupedPractical('unknown'));reject(lambda:GroupedPractical('PROP_GROUP').apply(x,eval_meta,namespace=ns));reject(lambda:GroupedPractical('PROP_GROUP').apply(x,meta,namespace='target_eval'));reject(lambda:GroupedPractical('PROP_GROUP').apply(x,meta,namespace=ns,scene='practical_low_urban'));reject(lambda:GroupedPractical('PROP_GROUP').apply(x[:,:,:255],meta,namespace=ns))
 wrong=x.clone();wrong[0,0,0]=float('nan');reject(lambda:GroupedPractical('PROP_GROUP').apply(wrong,meta,namespace=ns))
 cfg=list(base.configs.values())[0];receiver=base.ch.receiver_for_session(cfg,20261016102,'rx1:day1:eq1');a=base.residual.ResidualChannel(cfg,1,receiver);a.used=True;b=base.residual.ResidualChannel(cfg,2,receiver);reject(lambda:mix_propagation(a,b))
 return dict(status='PASS',synthetic_channel_processes=calls,negative=negative,source_IQ=0,CP_reads=0,NN=0,updates=0,IID_exact_original=True,all6_eval_exact_original=True,no_TX_session_dependency=True,noise_phase_sync_packetwise_preserved=True,propagation_groups_shared=True,global_RNG_unchanged=True)
if __name__=='__main__':print(json.dumps(run()))
