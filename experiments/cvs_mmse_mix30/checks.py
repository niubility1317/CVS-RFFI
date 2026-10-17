"""Focused new training wiring negatives and direct original-core replay."""
import copy,json,sys
from dataclasses import replace
import numpy as np
from . import design as d
from .selector import choice
from .checkpoint_policy import load

def run():
 import torch
 from .channel import Practical
 negative=0
 def bad(fn):
  nonlocal negative
  try:fn()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid training contract accepted')
 c=d.config(d.rows()[0])
 for key,value in [('epochs',11),('label_epochs',1),('pseudo_epochs',9),('updates',2442),('cumulative_epochs',31),('parent_source','other'),('processing_seed',20261017942),('control_run','other'),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1),('physical_channel_prepasses',1),('model_seed',2026092702)]:
  q=copy.deepcopy(c);q[key]=value;bad(lambda q=q:d.validate(q))
 for key,value in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
  q=copy.deepcopy(c);q['data_contract'][key]=value;bad(lambda q=q:d.validate(q))
 for key,value in [('target_delay_samples',64),('probabilities',[.25,.75]),('max_gain_db',30.)]:
  q=copy.deepcopy(c);q['receiver_policy'][key]=value;bad(lambda q=q:d.validate(q))
 sentinel=object();calls=[]
 def actual(path,dev):calls.append((path,dev));return 'OWN_FINAL'
 assert load('/p/final','cpu','/p/final','/r/final',sentinel,actual) is sentinel and not calls
 assert load('/r/final','cpu','/p/final','/r/final',sentinel,actual)=='OWN_FINAL' and len(calls)==1
 for path in ('/other/final','/r/latest','/p/best'):bad(lambda path=path:load(path,'cpu','/p/final','/r/final',sentinel,actual))
 torch.set_num_threads(2);t=np.arange(256);adapter=Practical();meta={k:[] for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
 for r in d.FIT:
  for z in (1,2,3):
   for k,v in zip(meta,(1,r,z,0,len(meta['base_index']))):meta[k].append(v)
 xx=torch.tensor([[np.cos(.2*t).tolist(),np.sin(.2*t).tolist()] for _ in meta['eq_i']],dtype=torch.float32);before=xx.clone();ng=np.random.get_state();tg=torch.get_rng_state().clone();out,records=adapter.apply(xx,meta,epoch=21,step=4441)
 assert torch.equal(xx,before) and out.shape==xx.shape and len(records)==15 and adapter.receiver_calls==15 and len(adapter.sessions)==15 and set(adapter.receiver_counts)=={0,1}
 maxerror=0.;nominalreplays=0;MMSEreplays=0
 for i,r in enumerate(records):
  pid=tuple(meta[k][i] for k in meta);selected=choice(pid,21,4441);scene=r['scene'];cfg=adapter.configs[scene];seed=adapter.ch.stable_seed(d.CHANNEL_SEEDS['train'],r['namespace'],scene,pid);hw=adapter.ch.receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],r['session_id']);norm,_=adapter.ch.normalize_input(np.asarray(xx[i,0].tolist())+1j*np.asarray(xx[i,1].tolist()))
  if selected:cfg=replace(cfg,equalization_enabled=True,equalizer_method='mmse',equalizer_delay_samples=24);MMSEreplays+=1
  else:nominalreplays+=1
  y,m=adapter.residual.ResidualChannel(cfg,seed,hw).process(norm);expected=torch.tensor(np.stack((y.real,y.imag)).tolist(),dtype=torch.float32);error=float((out[i]-expected).abs().max());maxerror=max(error,maxerror);assert error==0 and r['receiver_choice']==selected
  assert r['channel_equalization_applied']==bool(selected and r['receiver_lock_status']!='unlocked')
 order=list(reversed(range(15)));again=Practical();rr,rm=again.apply(xx[order],{k:[v[i] for i in order] for k,v in meta.items()},epoch=21,step=4441);assert torch.equal(rr,out[order]) and rm==[records[i] for i in order]
 assert torch.equal(tg,torch.get_rng_state());after=np.random.get_state();assert ng[0]==after[0] and np.array_equal(ng[1],after[1]) and ng[2:]==after[2:]
 for ep,step in ((20,4441),(21,4440),(22,4441),(31,6661)):bad(lambda ep=ep,step=step:adapter.apply(xx,meta,epoch=ep,step=step))
 for key,val in (('eq_i',0),('rx_i',2),('day_i',4)):
  mm=copy.deepcopy(meta);mm[key][0]=val;bad(lambda mm=mm:adapter.apply(xx,mm,epoch=21,step=4441))
 for role in ('U','eval','test'):bad(lambda role=role:Practical(role))
 from . import model as api
 m=api.build(c,'cpu');m.eval()
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))
 assert z['tx_logits'].shape==(2,6) and z['id_feat_joint'].shape==(2,160) and torch.isfinite(z['tx_logits']).all() and sum(p.numel() for p in m.parameters())==900722
 return dict(status='PASS',protocol_negatives=negative,actual_adapter_cases=30,direct_core_replays=15,direct_replay_max_error=maxerror,NoEQ_replays=nominalreplays,MMSE24_replays=MMSEreplays,global_numpy_and_torch_RNG_unchanged=True,input_unchanged=True,batch_permutation_invariant=True,actual_F_parameters=900722,real_IQ=0,CP=0,synthetic_NN=2,updates=0,training_or_identity_gain_proved=False)
if __name__=='__main__':
 sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'));print(json.dumps(run()))
