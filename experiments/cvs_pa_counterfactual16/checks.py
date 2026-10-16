"""Focused native-call/RNG counterfactual and decision negative cases,no real IQ."""
import ast,importlib.util,random,sys
from types import SimpleNamespace
from typing import Any,Dict
import torch
from . import design as d
from .core import paired,summary,qualify

def main():
 local=d.ROOT/'local_artifacts/F_crossrx_satmix_confirm20_20261016';spec=importlib.util.spec_from_file_location('actual_aug_check',local/'actual_DataAugmentation.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 code=d.read(local/'actual_callable_audit.json')['sources']['configure_augmentor'];functions={n.name:ast.get_source_segment(code,n) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)};scope=dict(Any=Any,Dict=Dict)
 for name in ('ramp_value','build_aug_base_cfg','configure_augmentor_for_epoch'):exec(compile(functions[name],name,'exec'),scope)
 args=d.read(local/'latest.json')['workers'][0]['native_args'];args['seed']=2026092701;args=SimpleNamespace(**args);base=scope['build_aug_base_cfg'](args);aug=module.build_augmentor(**base)
 x=torch.randn(32,2,256,generator=torch.Generator().manual_seed(110));y=torch.arange(32)%6;synthetic=0;triggers=0
 for epoch in d.EPOCHS:
  scope['configure_augmentor_for_epoch'](aug,base,epoch,args);random.seed(d.SEED+epoch);torch.manual_seed(d.SEED+epoch);a=random.getstate();b=torch.random.get_rng_state();expected=aug(x,labels=y,no_pa=False);end_a=random.getstate();end_b=torch.random.get_rng_state();random.setstate(a);torch.random.set_rng_state(b);full,bypass,info=paired(aug,x,y);synthetic+=64;triggers+=sum(info['triggered'])
  assert torch.equal(expected,full) and end_a==random.getstate() and torch.equal(end_b,torch.random.get_rng_state()) and torch.equal(full[~torch.tensor(info['triggered'])],bypass[~torch.tensor(info['triggered'])])
 assert triggers>0
 # Guaranteed-trigger synthetic counterexample:correct prefix matches no-PA at
 # the same start state,but bypassing a *new draw* must not replace the prefix.
 aug.p_pa=1.;random.seed(711);torch.manual_seed(711);a=random.getstate();b=torch.random.get_rng_state();full,bypass,info=paired(aug,x,y);random.setstate(a);torch.random.set_rng_state(b);independent=aug(x,labels=y,no_pa=True);synthetic+=64;assert torch.equal(independent,bypass) and all(info['triggered']) and any(info['changed'])
 negatives=0
 def rejects(call):
  nonlocal negatives
  try:call()
  except ValueError:negatives+=1
  else:raise AssertionError('Expected rejection')
 rejects(lambda:paired(aug,x[:,:,:255],y));rejects(lambda:paired(aug,x.double(),y));rejects(lambda:paired(aug,x,torch.full_like(y,6)));bad=x.clone();bad[0,0,0]=float('nan');rejects(lambda:paired(aug,bad,y))
 aug.p_dac=.1;rejects(lambda:paired(aug,x,y));aug.p_dac=0;aug.enable_class_signature=True;rejects(lambda:paired(aug,x,y));aug.enable_class_signature=False
 c=d.config();c['target_access']=True;rejects(lambda:d.validate(c))
 # Physically repeated observations are described,not treated as independent
 # trials for a significance test;the rule uses fixed effects and strata only.
 labels=[0]*2000;full=[1]*200+[0]*1800;bypass=[0]*2000;changed=[True]*2000;rx=list(d.FIT)*500;day=[1,2,3,1]*500;s=summary(labels,full,bypass,changed,rx,day);assert qualify(s)['decision']=='PASS' and s['changed']['repairs']==200
 import copy
 for change in ('count','recovery','RX','day'):
  v=copy.deepcopy(s)
  if change=='count':v['changed']['count']=999
  elif change=='recovery':v['changed']['recovery_pp']=1.999
  elif change=='RX':v['by_RX']['3']['recovery_pp']=-1.01
  else:v['by_day']['1']['recovery_pp']=-.01
  assert qualify(v)['decision']=='FAIL';negatives+=1
 value=dict(status='PASS',negative_cases=negatives,canonical_native_outputs_bitwise_equal=True,native_Python_Torch_RNG_end_states_equal=True,same_draw_noPA_equals_captured_prefix=True,all20_actual_schedules=True,synthetic_augmentation_packets=synthetic,real_IQ_reads=0,model_forwards=0,training_updates=0,target_access=False)
 d.write(d.ROOT/d.LOCAL/'local_checks.json',value);print(value)
if __name__=='__main__':main()
