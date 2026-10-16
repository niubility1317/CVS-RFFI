"""Focused changed-behavior checks; all IQ synthetic,zero optimizer updates."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess,sys,json
import numpy as np
import torch
from . import design as d
from .core import statistics,aggregate
from .analyze import from_gram
from .noise import FixedNormal,observe

def run(reference_metadata):
 negatives=0
 def rejects(call):
  nonlocal negatives
  try:call()
  except (ValueError,KeyError,ArithmeticError):negatives+=1;return
  raise AssertionError('Expected negative rejected')
 torch.set_num_threads(2)
 for row in d.rows():
  c=d.config(row);d.validate(c)
  for key,value in [('source_only',False),('target_access',True),('draws',4),('source_L_count',6300),('fit_RX',[1,3,4,6,8]),('optimizer_updates',1),('head_fit_updates',1),('automatic_training',True)]:
   bad=deepcopy(c);bad[key]=value;rejects(lambda bad=bad:d.validate(bad))
 x=torch.tensor([[float(i),float(3*i)] for i in range(1,9)])
 zero=statistics(x,-x);assert zero['antithetic_pair_trace']==0 and zero['trace_ratio']==0
 from_gram(dict(group='whole',batch=0,count=4,**zero))
 badgram=deepcopy(zero);badgram['raw_derivative_gram'][0][0]+=1.;rejects(lambda:from_gram(dict(group='whole',batch=0,count=4,**badgram)))
 same=statistics(x,x);assert abs(same['trace_ratio']-2)<1e-12
 assert not aggregate([dict(group='whole',**same)])['qualified_rationale']
 assert aggregate([dict(group='whole',**zero)])['qualified_rationale']
 for p,m in [(x,x[:2]),(x[:1],x[:1]),(x*float('nan'),x),(x.flatten(),x.flatten())]:rejects(lambda p=p,m=m:statistics(p,m))
 rejects(lambda:FixedNormal(np.zeros((3,2))));rejects(lambda:FixedNormal(np.zeros((8,3))));rejects(lambda:FixedNormal(np.full((8,2),np.nan)))
 r=FixedNormal(np.zeros((8,2)));r.normal(size=(8,2));rejects(lambda:r.normal(size=(8,2)))
 rejects(lambda:FixedNormal(np.zeros((8,2))).normal(size=(9,2)))
 from experiments.cvs_crossrx_satmix20.channel import imports
 ch,residual=imports();cfg=ch.Config(fs_hz=25e6,fc_hz=2462e6,scenario='practical_low_urban',processing_route='residual',mode='post_sync',equalization_enabled=False)
 hw=ch.receiver_for_session(cfg,d.STREAMS['receiver'],'rx3:day1:eq1');rng=np.random.default_rng(78453);iq=rng.normal(size=256)+1j*rng.normal(size=256);iq,_=ch.normalize_input(iq);noise=rng.normal(size=(256,2));copy=iq.copy()
 for config in (cfg,replace(cfg,equalization_enabled=True)):
  for seed in (78451,78452):
   p,pm,pt=observe(residual,config,seed,hw,iq,noise);m,mm,mt=observe(residual,config,seed,hw,iq,-noise)
   assert np.array_equal(pt['pre_agc_signal'],mt['pre_agc_signal']) and np.array_equal(pt['pre_agc_noise'],-mt['pre_agc_noise'])
   assert all(pm[k]==mm[k] for k in ('state_end','receiver_lock_status','quality_snr_db','output_frequency_hz','residual_scale','channel_equalization_applied'))
   for value,meta,trace in ((p,pm,pt),(m,mm,mt)):
    raw=trace['pre_agc_signal']+trace['pre_agc_noise'];requested=-20*np.log10(max(float(np.sqrt(np.mean(abs(raw)**2))),1e-300));gain=float(np.clip(requested,-config.agc_limit_db,config.agc_limit_db))
    assert gain==meta['agc_gain_db'] and np.array_equal(value,(raw*10**(gain/20)).astype(np.complex64))
  # Positive view exactly equals original stream with its own original primitive draw.
  engine=residual.ResidualChannel(config,78453,hw);z=engine.latent.rngs['noise'].normal(size=(256,2));a,am,at=observe(residual,config,78453,hw,iq,z)
  b,bm,bt=residual.ResidualChannel(config,78453,hw).process(iq,return_traces=True)
  assert np.array_equal(a,b) and am==bm and np.array_equal(at['equalizer_kernel'],bt['equalizer_kernel'])
 from .noise import Channel
 channel=Channel();from experiments.cvs_crossrx_satmix20.channel import PairedChannel
 original_config=json.loads(json.dumps(PairedChannel().audit()['actual_configs']));assert all(json.loads(json.dumps(__import__('dataclasses').asdict(c)))==original_config[k] for k,c in channel.configs.items())
 changed=deepcopy(original_config);changed['practical_low_urban']['reference_snr_db']+=1;assert not all(json.loads(json.dumps(__import__('dataclasses').asdict(c)))==changed[k] for k,c in channel.configs.items())
 test=torch.tensor(np.stack((np.stack((iq.real,iq.imag)),np.stack((iq.real,iq.imag)))),dtype=torch.float32)
 views,rows=channel.batch(test,['synthetic:1','synthetic:2'],[3,4],[1,2]);assert views.shape==(8,2,2,2,256) and len(rows)==32 and channel.calls==32
 assert np.array_equal(copy,iq)
 rejects(lambda:observe(residual,replace(cfg,processing_route='full'),78451,hw,iq,noise))
 rejects(lambda:observe(residual,cfg,78451,hw,iq,noise[:128]))
 R=Path(reference_metadata);done=d.read(R/'completion.json');init=d.read(R/'initialization.json');contract=d.read(R/'source_contract.json');freeze=d.read(R/'source_frozen.json');carrier=dict(role_ids=contract['role_ids']);d.verify_reference(2026092701,done,init,contract,freeze,carrier)
 for root,key,value in [('done','target_access',True),('done','epoch',200),('done','checkpoint_sources',['unknown']),('init','scratch_only',False),('freeze','selection','target_best'),('contract','held_RX',6)]:
  values=dict(done=deepcopy(done),init=deepcopy(init),contract=deepcopy(contract),freeze=deepcopy(freeze));values[root][key]=value
  rejects(lambda values=values:d.verify_reference(2026092701,values['done'],values['init'],values['contract'],values['freeze'],carrier))
 from experiments.cvs_additive_as20 import model as api,design as actual
 # Build existing default architecture;all new observation losses disabled via eval.
 actual.old.prior.SEED=2026092701;actual.old.SEED=2026092701;actual.SEED=2026092701
 c=actual.config(next(r for r in actual.rows() if r['arm']=='D_raw'));model=api.build(c,'cpu').eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
 from .probe import gradient
 x=torch.randn(4,2,256,generator=torch.Generator().manual_seed(78454));y=torch.arange(4);domains=torch.zeros(4,dtype=torch.long);named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 for value in (x,x+.01,x-.01,x.clone()):
  g,logits,ce=gradient(model,value,y,domains,named);assert logits.shape==(4,6) and sum(v.numel() for v in g.values())==sum(p.numel() for _,p in named)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for _,p in named)
 direct=subprocess.run([sys.executable,str(Path(__file__).with_name('runner.py')),'--seed','99'],capture_output=True,text=True)
 assert direct.returncode and 'No source/noise/budget/checkpoint overrides' in direct.stderr and 'Traceback' in direct.stderr
 return dict(status='PASS',negative_cases=negatives,exact_original_positive_view=True,original_AGC_each_sign=True,EQoff_and_EQon_noise_processing=True,variance_oracles=True,real_architecture_synthetic_NN=16,optimizer_updates=0,head_fit_updates=0,source_IQ_reads=0,checkpoint_reads=0,target_access=False)
