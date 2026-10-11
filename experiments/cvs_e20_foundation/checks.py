"""Negative metadata tests for E20-only inheritance, with no dataset access."""
import copy
from .verify import config,validate_metadata,EXECUTION_COMMIT

def check():
 c=config(2026092701)
 init=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],target_contact=False,source_roles='EXACT_MATCH',seed=c['model_seed'])
 done=dict(status='SOURCE_TRAINED',epoch=20,logged_steps=4440,optimizer_steps=4440,config=c,target_access=False,checkpoint_sources=[])
 resolved=dict(c,commit=EXECUTION_COMMIT)
 actual=dict(from_scratch=True,baseline_ckpt='',teacher_ckpt='',seed=c['model_seed'],label_epochs=20,pseudo_epochs=0,label_smoothing=0,
  use_unlabeled=False,use_ema_teacher=False,use_concat_sat_channel_aug=False,checkpoint_selection='final_only',wisig_pkl=c['dataset'],
  a1_source_screen_only=True,a1_periodic_target_inputs='',a1_periodic_target_truth='',a1_periodic_target_start=0,a1_periodic_target_interval=0,a1_final_weak_reference=False)
 validate_metadata(c,init,done,resolved,actual)
 cases=[(0,'target_contact',True),(0,'ancestors',['unknown']),(0,'seed',1),(1,'epoch',200),(1,'optimizer_steps',44400),
  (1,'target_access',True),(2,'commit','unverified'),(3,'use_unlabeled',True),(3,'use_ema_teacher',True),(3,'use_concat_sat_channel_aug',True),(3,'checkpoint_selection','best_test')]
 for index,key,value in cases:
  records=copy.deepcopy([init,done,resolved,actual]);records[index][key]=value
  try:validate_metadata(c,*records)
  except ValueError:pass
  else:raise AssertionError('Invalid E20 provenance accepted: '+key)
 return dict(status='PASS',negative_cases=len(cases),target_access=False)

if __name__=='__main__':
 import json
 print(json.dumps(check()))
