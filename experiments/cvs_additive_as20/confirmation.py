"""One fixed independent seed02 confirmation; no new loss, IQ or U rule."""
import math
from copy import deepcopy
from pathlib import Path
PROFILE='confirmation_seed02'
SEED=2026092702
FIRST='20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01'
RUN='20261016-phase1-F-crossRX-additiveAs-confirm-sourceRX1-e20-s2026092702-m3-r01'
RELEASE='cvs_F_crossRX_additiveAs_confirm_sourceRX1_e20_20261016_r01'
OWNER='codex/root/F-CROSS_RX-additiveAs-confirm-E20-20261016'
LOCAL='local_artifacts/F_additive_As_confirm20_20261016'
def activate():
 from . import design as d
 # The underlying timing/F configuration constructs the actual initialized graph.
 # Changing only native args.seed would leave its recorded model seed inconsistent.
 d.old.prior.SEED=SEED;d.old.SEED=SEED;d.SEED=SEED
 d.RUN=RUN;d.RELEASE=RELEASE;d.OWNER=OWNER;d.LOCAL=LOCAL
 d.BASE=Path(d.PROJECT)/'runs'/RUN
 d.SMOKE_SOURCE=Path(d.PROJECT)/'runs'/FIRST/'D00-RX1-s2026092701/source'
 return d
def equal(a,b):
 return a==b or (isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b))
def checks():
 import json,subprocess,sys,tempfile
 import torch
 from . import design as d,model as api
 torch.set_num_threads(2)
 before={r['arm']:d.config(r) for r in d.rows()}
 native_before={arm:vars(d.make_args(c,'cpu')).copy() for arm,c in before.items()}
 torch.manual_seed(2026092701);first=api.build(before['D00'],'cpu')
 first_state={k:v.detach().clone() for k,v in first.state_dict().items()};del first
 d=activate();assert d.base_model_config()['model_seed']==SEED
 first_result=d.read(d.ROOT/'automation_reports/CV-SincNet'/FIRST/'source_analysis.json')
 assert first_result['qualification']['qualified'] and first_result['source_only'] and not first_result['target_access'] and not first_result['joint_goal_complete']
 output=[];negative=0;states=[]
 for row in d.rows():
  c=d.validate(d.config(row));a=vars(d.make_args(c,'cpu'));old=before[row['arm']]
  assert c['model_seed']==a['seed']==SEED and c['updates']==4440 and c['epochs']==20
  assert c['run_id']==a['run_id']==RUN and c['row_id']==a['candidate_id'] and c['output_root']==a['output_dir'] and FIRST not in c['output_root']
  assert c['training_mechanism']==old['training_mechanism'] and c['additive']==old['additive'] and c['channel']==old['channel'] and c['source_selection']==old['source_selection'] and c['supcon']==old['supcon'] and c['fold']==old['fold']
  diff=sorted(k for k,v in a.items() if not equal(v,native_before[row['arm']].get(k)))
  assert set(diff)=={'seed','run_id','candidate_id','output_dir'} and not a['baseline_ckpt'] and not a['teacher_ckpt']
  torch.manual_seed(SEED);m=api.build(c,'cpu');state={k:v.detach().clone() for k,v in m.state_dict().items()};states.append(state)
  assert set(state)==set(first_state) and any(not torch.equal(v,first_state[k]) for k,v in state.items())
  assert m.id_backbone.encoder._view_train_packets==m.id_backbone.encoder._view_eval_packets==0 and m._additive_calls==0 and m._joint_calls==0
  for path,value in ((('model_seed',),2026092701),(('row_id',),row['row_id'].replace(str(SEED),'2026092701')),(('epochs',),50),(('target_access',),True),(('additive','extra_weight'),.5),(('training_mechanism','L_forwards_per_packet'),6),(('channel','selection'),'urban only')):
   bad=deepcopy(c);target=bad
   for key in path[:-1]:target=target[key]
   target[path[-1]]=value
   try:d.validate(bad)
   except ValueError:negative+=1
   else:raise AssertionError('Unregistered confirmation override accepted')
  output.append(dict(row_id=row['row_id'],native_model_seed=a['seed'],actual_F_model_config_seed=d.base_model_config()['model_seed'],native_argument_changes=diff,original_loss_IQ_U_channel_data_budget=True,no_checkpoint_initialization=True))
  del m
 assert all(all(torch.equal(v,s[k]) for k,v in states[0].items()) for s in states[1:]) and activate().rows()==d.rows()
 from . import runner,publish
 with tempfile.TemporaryDirectory() as temporary:
  fake=Path(temporary)/'experiments/cvs_additive_as20/runner.py';fake.parent.mkdir(parents=True)
  (Path(temporary)/'runtime.json').write_text(json.dumps(dict(backend='must_not_open',profile='unregistered')),encoding='utf8')
  original=runner.__file__;runner.__file__=str(fake)
  try:
   try:runner.bootstrap()
   except ValueError:negative+=1
   else:raise AssertionError('Unknown worker profile accepted')
  finally:runner.__file__=original
 try:publish.main(profile='unregistered')
 except ValueError:negative+=1
 else:raise AssertionError('Unknown publisher profile accepted')
 process=subprocess.run([sys.executable,'-S',str(d.ROOT/'experiments/cvs_additive_as20/runner.py'),'--unregistered'],capture_output=True,text=True)
 assert process.returncode and 'No budget/data/additive-weight overrides permitted' in process.stderr and 'ModuleNotFoundError' not in process.stderr
 negative+=1
 value=dict(status='PASS',profile=PROFILE,rows=output,negative_cases=negative,three_seed02_initial_states_equal=True,seed02_states_differ_from_seed01=True,unchanged_state_keys=True,first_source_rule_pass_required=True,source_IQ_reads=0,target_access=False,model_forward_packets=0,optimizer_updates=0,prior_mechanism_checks='Original20negative/3858syntheticNN/3syntheticupdates retained;not repeated')
 d.write(d.ROOT/LOCAL/'local_validation.json',value);print(json.dumps(value));return value
if __name__=='__main__':
 import sys
 if sys.argv[1:]==['--check']:checks()
 elif sys.argv[1:]==['--publish']:
  from .publish import main
  main(profile=PROFILE)
 else:raise ValueError('Use exactly --check or --publish')
