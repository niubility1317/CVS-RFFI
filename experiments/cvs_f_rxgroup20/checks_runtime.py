"""Native augmentation parity and stream isolation;synthetic IQ only."""
import argparse,ast,copy,importlib.util,json,random,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any,Dict
import torch
from .coupling import Runtime,AugmentorProxy

def run(native_file,schedule_file,actual):
 module_name='coupling_native_augmentation';spec=importlib.util.spec_from_file_location(module_name,native_file);native=importlib.util.module_from_spec(spec);sys.modules[module_name]=native;spec.loader.exec_module(native)
 source=Path(schedule_file).read_text(encoding='utf8');scope=dict(Any=Any,Dict=Dict)
 for name in ('ramp_value','build_aug_base_cfg','configure_augmentor_for_epoch'):
  fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.get_source_segment(source,fn),name,'exec'),scope)
 recorded=json.loads(Path(actual).read_text(encoding='utf8'));args=SimpleNamespace(**recorded['args'],epochs=20,late_stable_start=0,late_aug_min_scale=-1,seed=2026092701);base_cfg=scope['build_aug_base_cfg'](args)
 if base_cfg!=recorded['base_config'] or base_cfg['p_dac'] or base_cfg['enable_class_signature']:raise ValueError('Actual original ordinary config required')
 original=native.build_augmentor(**base_cfg);batch=AugmentorProxy(native.build_augmentor(**base_cfg),Runtime('BATCH'));group=AugmentorProxy(native.build_augmentor(**base_cfg),Runtime('RXGROUP'))
 torch.manual_seed(20261017832);x=torch.randn(128,2,256);y=torch.arange(128)%6;meta={k:v for k,v in dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor((3,4,6,8)*32),day_i=torch.arange(128)%3+1,sig_i=torch.arange(128),base_index=torch.arange(128)).items()};snapshot=x.clone()
 paired_equal=0;nonzero=0
 for epoch in range(1,21):
  for aug in (original,batch,group):scope['configure_augmentor_for_epoch'](aug,base_cfg,epoch,args)
  torch.manual_seed(20261017900+epoch);random.seed(20261017900+epoch);ts=torch.get_rng_state();ps=random.getstate();expected=original(x,labels=y,no_pa=False);end=(torch.get_rng_state(),random.getstate())
  for aug in (batch,group):
   aug.runtime.set_L(meta);torch.set_rng_state(ts);random.setstate(ps);value=aug(x,labels=y,no_pa=False)
   assert torch.equal(torch.get_rng_state(),end[0]) and random.getstate()==end[1] and torch.equal(snapshot,x) and torch.isfinite(value).all()
   if aug is batch:assert torch.equal(value,expected);paired_equal+=1
   else:nonzero+=not torch.equal(value,expected)
 assert paired_equal==20 and nonzero>0 and batch.runtime.calls==group.runtime.calls==20 and group.runtime.stage_calls==40
 assert sum(v['actual_disagree'] for v in batch.runtime.summary()['pair_gate_counts'].values())==0 and sum(v['actual_disagree'] for v in group.runtime.summary()['pair_gate_counts'].values())>0
 # U normal and defect views do not enter grouping;no U label needed.
 count=group.runtime.calls;torch.manual_seed(44);random.seed(44);ts=torch.get_rng_state();ps=random.getstate();one=original(x,labels=None,no_pa=True);torch.set_rng_state(ts);random.setstate(ps);two=group(x,labels=None,no_pa=True);assert torch.equal(one,two) and group.runtime.calls==count
 # Nonnegative U pseudolabels still must not select the labeled coupling path.
 group.runtime.set_U();torch.set_rng_state(ts);random.setstate(ps);one=original(x,labels=y,no_pa=False);torch.set_rng_state(ts);random.setstate(ps);two=group(x,labels=y,no_pa=False);assert torch.equal(one,two) and group.runtime.calls==count
 # Group assignment and random seeds do not use visible TX labels.
 one=AugmentorProxy(native.build_augmentor(**base_cfg),Runtime('RXGROUP'));two=AugmentorProxy(native.build_augmentor(**base_cfg),Runtime('RXGROUP'))
 for aug in (one,two):scope['configure_augmentor_for_epoch'](aug,base_cfg,20,args);aug.runtime.set_L(meta)
 torch.manual_seed(45);random.seed(45);ts=torch.get_rng_state();ps=random.getstate();a=one(x,labels=y,no_pa=False);torch.set_rng_state(ts);random.setstate(ps);b=two(x,labels=(y+1)%6,no_pa=False);assert torch.equal(a,b)
 negatives=0
 def reject(call):
  nonlocal negatives
  try:call()
  except ValueError:negatives+=1;return
  raise AssertionError('Invalid training role accepted')
 reject(lambda:Runtime('TXGROUP'))
 for field,value in [('rx_i',torch.ones(128,dtype=torch.long)),('day_i',torch.zeros(128,dtype=torch.long)),('eq_i',torch.zeros(128,dtype=torch.long))]:
  r=Runtime('RXGROUP');r.set_L(dict(meta,**{field:value}));reject(lambda:r.begin(x,y))
 for xx,yy in ((x[:127],y[:127]),(x[:,:,:255],y),(x*float('nan'),y),(x,y.float()),(x,y+6)):
  r=Runtime('RXGROUP');r.set_L(meta);reject(lambda:r.begin(xx,yy))
 r=Runtime('RXGROUP');reject(lambda:r.begin(x,y));r.set_L(meta);r.begin(x,y);reject(lambda:r.begin(x,y));reject(lambda:r.finish(x))
 return dict(status='PASS',native_epoch_parity_checks=20,group_nonidentical_epochs=nonzero,protocol_negatives=negatives,synthetic_waveform_call_packets=128*(20*3+2*2+2),NN=0,model_checkpoints=0,optimizer_updates=0,actual_native_code=str(native_file),BASE_IQ_bitwise_native=True,CPU_Python_streams_bitwise_native=True,RDrop_input_generated_once=True,visible_TX_label_permutation_equivariant=True,U_passthrough_bitwise=True,ordinary_scalar_parameters_unchanged=True,original_PA_mask_and_parameters_stream_unchanged=True,base_crossRX_positive_gate_disagreement=0,group_crossRX_positive_gate_disagreement_positive=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--native',required=True);p.add_argument('--schedule',required=True);p.add_argument('--actual',required=True);a=p.parse_args();print(json.dumps(run(a.native,a.schedule,a.actual)))
