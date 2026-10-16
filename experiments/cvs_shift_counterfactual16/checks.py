"""Actual native schedule/RNG/output controls and focused negative cases."""
import ast,copy,importlib.util,random,subprocess,sys
from types import SimpleNamespace
from typing import Any,Dict
import torch
from . import design as d
from .core import paired,summary,qualify,rng,restore,same

def main():
 torch.set_num_threads(2);local=d.ROOT/'local_artifacts/F_crossrx_satmix_confirm20_20261016'
 spec=importlib.util.spec_from_file_location('actual_shift_check',local/'actual_DataAugmentation.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 code=d.read(local/'actual_callable_audit.json')['sources']['configure_augmentor'];functions={n.name:ast.get_source_segment(code,n) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)};scope=dict(Any=Any,Dict=Dict)
 for name in ('ramp_value','build_aug_base_cfg','configure_augmentor_for_epoch'):exec(compile(functions[name],name,'exec'),scope)
 args=SimpleNamespace(**d.read(local/'latest.json')['workers'][0]['native_args']);base=scope['build_aug_base_cfg'](args);aug=module.build_augmentor(**base)
 x=torch.randn(32,2,256,generator=torch.Generator().manual_seed(110));y=torch.arange(32)%6;synthetic=0;nonzero=0
 original=module._apply_time_shift
 for epoch in d.EPOCHS:
  state=scope['configure_augmentor_for_epoch'](aug,base,epoch,args);random.seed(d.SEED+epoch);torch.manual_seed(d.SEED+epoch);begin=rng();expected=aug(x.clone(),labels=y,no_pa=False);end=rng();restore(begin);full,bypass,info=paired(aug,x,y);synthetic+=96;nonzero+=sum(info['nonzero_shift'])
  assert torch.equal(expected,full) and same(end,rng()) and module._apply_time_shift is original and info['nonzero_shift']==[v!=0 for v in info['shifts']]
  assert all(abs(v)<=aug.max_time_shift for v in info['shifts']) and torch.equal(full[~torch.tensor(info['nonzero_shift'])],bypass[~torch.tensor(info['nonzero_shift'])])
 assert args.label_epochs==13 and 3<=aug.max_time_shift<=8
 # Force gates only in synthetic control, never in the real source experiment.
 aug.p_time_shift=1.;aug.p_pa=1.;random.seed(711);torch.manual_seed(711);full,bypass,info=paired(aug,x,y,canonical=True);synthetic+=96
 assert info['canonical_checked'] and info['shift_calls']==1 and any(info['nonzero_shift']) and any(info['changed']) and all(v>0 for v in info['PA_strength'])
 aug.p_time_shift=0.;full,bypass,info=paired(aug,x,y,canonical=True);synthetic+=96
 assert full.equal(bypass) and not any(info['nonzero_shift']) and not any(info['changed']) and info['shift_inspection_scalar_draws']==0
 negatives=0
 def rejects(call):
  nonlocal negatives
  before=rng()
  try:call()
  except ValueError:negatives+=1
  else:raise AssertionError('Expected rejection')
  assert same(before,rng()) and module._apply_time_shift is original
 rejects(lambda:paired(aug,x[:,:,:255],y));rejects(lambda:paired(aug,x.double(),y));rejects(lambda:paired(aug,x,y.float()));rejects(lambda:paired(aug,x,torch.full_like(y,6)))
 bad=x.clone();bad[0,0,0]=float('nan');rejects(lambda:paired(aug,bad,y))
 aug.p_dac=.1;rejects(lambda:paired(aug,x,y));aug.p_dac=0;aug.enable_class_signature=True;rejects(lambda:paired(aug,x,y));aug.enable_class_signature=False
 c=d.config();c['target_access']=True;rejects(lambda:d.validate(c));c=d.config();c['epochs']=list(range(1,21));rejects(lambda:d.validate(c))
 s=summary([0]*2000,[1]*200+[0]*1800,[0]*2000,[True]*2000,list(d.FIT)*500,[1,2,3,1]*500);assert qualify(s)['decision']=='PASS'
 for mode in ('coverage','recovery','RXcoverage','RXpositive'):
  v=copy.deepcopy(s)
  if mode=='coverage':v['nonzero_shift']['count']=0
  elif mode=='recovery':v['nonzero_shift']['recovery_pp']=1.999999
  elif mode=='RXcoverage':v['by_RX']['3']['count']=0
  else:v['by_RX']['3']['recovery_pp']=0;v['by_RX']['4']['recovery_pp']=0
  assert qualify(v)['decision']=='FAIL';negatives+=1
 # Direct-script startup must reject unsupported args before any bootstrap,
 # even without site packages (regression against stdlib module shadowing).
 r=subprocess.run([sys.executable,'-S',str(d.ROOT/'experiments/cvs_shift_counterfactual16/runner.py'),'--unregistered'],capture_output=True,text=True)
 assert r.returncode!=0 and 'No role/seed/operator/budget overrides' in r.stderr and 'ModuleNotFoundError' not in r.stderr;negatives+=1
 value=dict(status='PASS',negative_cases=negatives,canonical_native_outputs_bitwise_equal=True,Python_Torch_final_RNG_equal=True,PA_strengths_same=True,all13_actual_L_schedules=True,natural_small_control_nonzero_shift_packets=nonzero,forced_gate_and_no_shift_controls=True,synthetic_augmentation_packets=synthetic,prior_local_attempt=dict(synthetic_augmentation_packets=1248,model_forwards=0,updates=0,failure='Test incorrectly required a gate in13small random batches;forced positive control tests mechanism without altering real experiment draws'),real_IQ_reads=0,model_forwards=0,training_updates=0,target_access=False)
 d.write(d.ROOT/d.LOCAL/'local_checks.json',value);print(value)
if __name__=='__main__':main()
