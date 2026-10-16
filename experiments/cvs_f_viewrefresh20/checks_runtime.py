"""Synthetic native parity,independent view construction and U isolation."""
import argparse,ast,copy,importlib.util,json,random,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any,Dict
import torch
from .iq import Runtime,AugmentorProxy

def run(native_file,schedule_file,actual):
 spec=importlib.util.spec_from_file_location('refresh_native',native_file);n=importlib.util.module_from_spec(spec);sys.modules[spec.name]=n;spec.loader.exec_module(n);source=Path(schedule_file).read_text();scope=dict(Any=Any,Dict=Dict)
 for name in ('ramp_value','build_aug_base_cfg','configure_augmentor_for_epoch'):
  fn=next(z for z in ast.parse(source).body if isinstance(z,ast.FunctionDef) and z.name==name);exec(compile(ast.get_source_segment(source,fn),name,'exec'),scope)
 recorded=json.loads(Path(actual).read_text());args=SimpleNamespace(**recorded['args'],epochs=20,late_stable_start=0,late_aug_min_scale=-1,seed=2026092701);cfg=scope['build_aug_base_cfg'](args)
 if cfg!=recorded['base_config'] or cfg['p_dac'] or cfg['enable_class_signature']:raise ValueError('Actual original ordinary config required')
 original=n.build_augmentor(**cfg);pre=[];old=original._apply_general_channel
 def capture(x):y=old(x);pre.append(y.clone());return y
 original._apply_general_channel=capture;augs={a:AugmentorProxy(n.build_augmentor(**cfg),Runtime(a)) for a in ('RAW','REPLAY','REFRESH')}
 torch.manual_seed(17962);x=torch.randn(128,2,256);snapshot=x.clone();labels=torch.arange(128)%6;meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor((3,4,6,8)*32),day_i=torch.arange(128)%3+1,sig_i=torch.arange(128),base_index=torch.arange(128));different=0
 for epoch in range(1,21):
  for a in (original,*augs.values()):scope['configure_augmentor_for_epoch'](a,cfg,min(epoch,13),args)
  torch.manual_seed(18000+epoch);random.seed(18000+epoch);start=(torch.get_rng_state(),random.getstate());expected=original(x,labels=labels,no_pa=False);end=(torch.get_rng_state(),random.getstate());fresh=[]
  for arm,aug in augs.items():
   aug.runtime.set_L(meta);torch.set_rng_state(start[0]);random.setstate(start[1]);out=aug(x,labels=labels,no_pa=False);assert torch.equal(out,expected) and torch.equal(end[0],torch.get_rng_state()) and end[1]==random.getstate() and torch.equal(x,snapshot);fresh.append(aug.runtime.fresh.clone());view=aug.runtime.take(x,labels,meta,epoch)
   assert torch.equal(view,{'RAW':x,'REPLAY':pre[-1],'REFRESH':fresh[-1]}[arm]) and not aug.runtime.pending
  assert all(torch.equal(fresh[0],z) for z in fresh[1:]);different+=not torch.equal(fresh[0],pre[-1])
 assert different>0 and all(a.runtime.calls==a.runtime.consumed==20 for a in augs.values())
 # U can carry nonnegative pseudo labels but is never a fresh labeled observation.
 aug=augs['REFRESH'];aug.runtime.set_U();count=aug.runtime.calls;torch.manual_seed(66);random.seed(66);start=(torch.get_rng_state(),random.getstate());one=original(x,labels=labels,no_pa=False);torch.set_rng_state(start[0]);random.setstate(start[1]);two=aug(x,labels=labels,no_pa=False);assert torch.equal(one,two) and aug.runtime.calls==count
 assert copy.deepcopy(aug.runtime) is None
 negatives=0
 def reject(fn):
  nonlocal negatives
  try:fn()
  except (ValueError,KeyError,TypeError):negatives+=1;return
  raise AssertionError('Invalid source view accepted')
 reject(lambda:Runtime('TXGROUP'))
 for xx,yy in ((x[:127],labels[:127]),(x[:,:,:255],labels),(x*float('nan'),labels),(x,labels.float()),(x,labels+6)):
  r=Runtime('REFRESH');r.set_L(meta);reject(lambda:r.begin(xx,yy))
 for k,v in [('rx_i',torch.ones(128,dtype=torch.long)),('eq_i',torch.zeros(128,dtype=torch.long)),('day_i',torch.zeros(128,dtype=torch.long))]:
  r=Runtime('REFRESH');r.set_L(dict(meta,**{k:v}));reject(lambda:r.begin(x,labels))
 r=Runtime('REFRESH');reject(lambda:r.begin(x,labels));reject(lambda:r.take(x,labels,meta,1));r.set_L(meta);r.begin(x,labels);reject(lambda:r.begin(x,labels));reject(lambda:r.finish_normal(original,old,old,x))
 return dict(status='PASS',protocol_negatives=negatives,actual_epochs=20,effective_epoch_cap=13,original_normal_IQ_and_CPU_Python_RNG_bitwise=True,three_arms_fresh_shadow_exact=True,fresh_nonidentical_epochs=different,RAW_and_REPLAY_exact=True,U_positive_pseudolabels_passthrough=True,teacher_runtime_deepcopy_None=True,primary_synthetic_native_normal_packets=128*(20*4+2),primary_synthetic_fresh_packets=128*20*3,NN=0,updates=0,realIQ=0,checkpoint_loads=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--native',required=True);p.add_argument('--schedule',required=True);p.add_argument('--actual',required=True);a=p.parse_args();print(json.dumps(run(a.native,a.schedule,a.actual)))
