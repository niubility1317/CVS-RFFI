"""Focused real-forward synthetic tests;zero realIQ/checkpoints or updates."""
import copy,json,math,random,subprocess,sys
import numpy as np
import torch
from . import design as d

def run():
 from .channel import Channel,split_control
 from .observer import run as observe,add_clean_cosines,FEATURES
 from experiments.cvs_f_noise_pair20 import design as pd,model as api
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);pd.parent.old.prior.SEED=pd.SEED;pd.parent.old.SEED=pd.SEED;pd.parent.SEED=pd.SEED;negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,KeyError,TypeError,FloatingPointError):negative+=1;return
  raise AssertionError('Invalid frozen-source diagnostic accepted')
 for row in d.rows():
  c=d.config(row);d.validate(c)
  for path,value in [(('source_only',),False),(('target_access',),True),(('model_seed',),2026092702),(('source_L_count',),6300),(('held_RX1_used',),True),(('optimizer_updates',),1),(('head_fit_updates',),1),(('model_epochs',),20),(('views',),['clean','full']),(('streams','noise'),4),(('checkpoint_source',),'unknown'),(('expected_NN',),30244)]:
   b=copy.deepcopy(c);node=b
   for k in path[:-1]:node=node[k]
   node[path[-1]]=value;reject(lambda b=b:d.validate(b))
 y=[0]*8;phys=[dict(state='BLOCKED' if i<4 else 'PARTIAL',lock='degraded' if i<4 else 'unlocked') for i in range(8)];views=dict(clean=[0]*8,full=[1]*8,no_added_noise_fixed_receiver=[0,0,1,1,0,0,1,1]);assert d.recoverability(y,views,phys)['supported']
 for i in (0,4):
  bad=copy.deepcopy(views);bad[d.VIEWS[-1]][i]=1;assert not d.recoverability(y,bad,phys)['supported'];negative+=1
 zero_gap=dict(views,full=[0]*8);assert not d.recoverability(y,zero_gap,phys)['supported'];negative+=1
 missing=[dict(state='LOS',lock='locked') for _ in y];assert not d.recoverability(y,views,missing)['supported'];negative+=1
 reject(lambda:d.recoverability([9]*8,views,phys));reject(lambda:d.recoverability(y,dict(clean=[0]*8,full=[1]*8),phys))
 carrier=dict(role_ids={k:[k+str(i) for i in range(n)] for k,n in [('L_s',6300),('U_s',56700),('V',27000)]});contract=dict(held_RX=1,source_receivers=list(d.FIT),role_ids={k:carrier['role_ids'][k][:n] for k,n in [('L_s',5040),('U_s',45360),('V',21600)]});init=dict(scratch_only=True,checkpoint_sources=[],target_contact=False);freeze=dict(epoch=20,selection='final_only',target_access=False);result=dict(actual_steps=4440,L_forward_packets=5683200,default_head_joint=True,head_fit_updates=0,target_access=False,checkpoint_sources=[],U_truth_read=False)
 expected=pd.config(next(v for v in pd.rows() if v['arm']=='ANTI'));done=dict(epoch=20,optimizer_steps=4440,initialization='scratch',checkpoint_sources=[],ancestors=[],head_fit_updates=0,target_access=False,target_evaluated=False,config=expected)
 d.verify_reference('ANTI',done,init,contract,freeze,carrier,result,expected)
 for key,value in [('epoch',200),('optimizer_steps',4441),('checkpoint_sources',['unknown']),('ancestors',['old']),('target_access',True),('head_fit_updates',1)]:
  bad=dict(done);bad[key]=value;reject(lambda bad=bad:d.verify_reference('ANTI',bad,init,contract,freeze,carrier,result,expected))
 badc=copy.deepcopy(contract);badc['role_ids']['L_s'][0]=badc['role_ids']['U_s'][0];reject(lambda:d.verify_reference('ANTI',done,init,badc,freeze,carrier,result,expected))
 reject(lambda:d.verify_reference('RAW10',done,init,contract,freeze,carrier,result,expected))
 g=torch.Generator().manual_seed(20261017604);x=torch.randn(12,2,256,generator=g);saved=x.clone();rx=[d.FIT[i%4] for i in range(12)];day=[i%3+1 for i in range(12)];ids=['synthetic-L'+str(i) for i in range(12)];labels=torch.arange(12)%6;domains=torch.tensor([3+3*d.FIT.index(r)+t-1 for r,t in zip(rx,day)]);rng=torch.get_rng_state().clone();prng=random.getstate();ch=Channel();full,ctrl,meta=ch.batch(x,ids,rx,day);full2,ctrl2,meta2=Channel().batch(x,ids,rx,day)
 assert torch.equal(x,saved) and torch.equal(full,full2) and torch.equal(ctrl,ctrl2) and meta==meta2 and not torch.equal(full,ctrl) and torch.equal(rng,torch.get_rng_state()) and prng==random.getstate() and ch.calls==12 and len(ch.sessions)==12
 assert all(p['G']==1 and not p['EQ'] and p['original_full_IQ_reconstructed_exact'] and p['same_receiver_compensation_AGC'] and p['added_noise_variance_reference']==.001 and p['numerical_filter_latency_samples']==24 for p in meta)
 reject(lambda:Channel().batch(x,ids,[1]*12,day));reject(lambda:Channel().batch(x,ids,rx,[0]*12));reject(lambda:Channel().batch(x,['same']*12,rx,day));reject(lambda:Channel().batch(x*float('nan'),ids,rx,day))
 sig=np.arange(256,dtype=np.float64)/256+1j*np.ones(256);noise=np.ones(256)*(.01+.02j);mm=dict(channel_equalization_applied=False,agc_gain_db=3.);tt=dict(pre_agc_signal=sig,pre_agc_noise=noise,equalizer_kernel=np.ones(1));yy=np.asarray((sig+noise)*10**(3./20),dtype=np.complex64);cc=split_control(yy,mm,tt);assert np.array_equal(cc,np.asarray(sig*10**(3./20),dtype=np.complex64))
 tzero=dict(tt,pre_agc_noise=np.zeros(256));yz=np.asarray(sig*10**(3./20),dtype=np.complex64);assert np.array_equal(split_control(yz,mm,tzero),yz)
 reject(lambda:split_control(yy+1,mm,tt));reject(lambda:split_control(yy,dict(mm,channel_equalization_applied=True),tt));reject(lambda:split_control(yy,dict(mm,agc_gain_db=float('nan')),tt));reject(lambda:split_control(yy,mm,dict(tt,equalizer_kernel=np.ones(2))))
 NN=0
 for arm in d.ARMS:
  torch.manual_seed(d.SEED);model=api.build(pd.config(next(v for v in pd.rows() if v['arm']==arm)),'cpu').eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
  with torch.no_grad(),fresh_parameter_caches(model):baseline=model.forward_identity_only(x,domain_labels=domains)['tx_logits'];NN+=len(x)
  clean=None
  for view,inp in zip(d.VIEWS,(x,full,ctrl)):
   rec,features=observe(model,inp,labels,domains);NN+=len(x)
   if clean is None:clean=features;assert torch.equal(torch.tensor([r['logits'] for r in rec]),baseline)
   add_clean_cosines(rec,features,clean);assert len(rec)==12 and all(set(r['feature_norms'])==set(FEATURES) and r['raw_reference_CFO_coordinate']==0 and all(v is not None and math.isfinite(v) and v>=0 for v in r['feature_gradient_L2'].values()) for r in rec)
   assert all(-1.000001<=v<=1.000001 for r in rec for v in r['feature_cosine_to_clean'].values())
   assert all(abs(r['feature_cosine_to_clean'][k]-1)<1e-8 for r in rec for k in FEATURES) if view=='clean' else True
   for r in rec:
    logits=r['logits'];yy=int(labels[rec.index(r)]);margin=(logits[yy]-max(v for i,v in enumerate(logits) if i!=yy))/30;assert abs(margin-r['margin_over30'])<1e-6
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in model.parameters()) and model._noise_calls==0 and model._noise_runtime is None and not model._supcon_handles and model._supcon_pending is None and model.id_backbone.encoder._view_eval_packets==48
  assert not any(v._forward_hooks for name,v in model.id_backbone.encoder.named_modules())
 direct=subprocess.run([sys.executable,str(__import__('pathlib').Path(__file__).with_name('runner.py')),'--target','anything'],capture_output=True,text=True);assert direct.returncode and 'No diagnostic data/views/stream/model/parameter overrides' in direct.stderr;negative+=1
 assert NN==96
 return dict(status='PASS',negative_cases=negative,synthetic_NN_packets=NN,real_IQ_reads=0,real_checkpoint_loads=0,optimizer_updates=0,head_fit_updates=0,target_access=False,controlled_full_reconstruction_exact=True,fixed_AGC_no_recompute=True,same_named_stream_exact=True,original_forward_logits_bitwise_equal=True,existing_branches_gates_gradient_observed=True,all_model_parameters_buffers_unchanged=True,no_parameter_grad_accumulation=True,all_hooks_removed=True,NoExplicitCFO_coordinate27_zero=True,integer_half_gap_boundary_PASS=True)
if __name__=='__main__':print(json.dumps(run()))
