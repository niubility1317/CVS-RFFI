from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch30_baseline import design as b
from experiments.cvs_stochastic_ensemble import design as pilot
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write,SCENES,VIEWS,VIEWS_ROOT=(getattr(b,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write','SCENES','VIEWS','VIEWS_ROOT'))
RUN='20261012-phase1-ensemble-final30-manysig-m3-r01';RELEASE='cvs_ensemble_final30_20261012_r01';OWNER='codex/root/ensemble-final30-20261012'
BASE=Path(PROJECT)/'runs'/RUN;ARMS=('EQUAL_PROBABILITY_3',);READOUTS=('ENSEMBLE_HEAD',)
def rows():return [dict(row_id=f'EQUAL_PROBABILITY_3-s{s}',arm='EQUAL_PROBABILITY_3',model_seed=s) for s in SEEDS]
def baseline_config(seed):return b.config(next(r for r in b.rows() if r['arm']=='CE30' and r['model_seed']==seed))
def config(row):
 if row not in rows():raise ValueError('Unregistered final ensemble row')
 c=deepcopy(baseline_config(row['model_seed']));c.update(run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,
  goal=dict(primary='EQUAL_PROBABILITY_3',clean_three_seed_mean=.84,feedback_to_training=False),
  ensemble=dict(formula='uniform mean softmax(logits),temperature1,no calibration',member_count=3,
   initialization_seeds=[pilot.member_seed(row['model_seed'],i) for i in range(3)],sampler_seed=row['model_seed']+101,
   member0_reuse='verifiedmatchingfrozenCE30fullsourceweights,inferenceonly,notteacherorinitializer',
   new_models=2,new_updates=13320,reused_updates=6660,total_model_updates=19980,epochs_per_member=30,
   trainable_parameters_per_member=177025,ensemble_trainable_parameters=531075,
   source_qualification=(pilot.BASE/'source_decision.json').as_posix(),source_qualification_result='PASS before target access'),
  readouts=list(READOUTS),checkpoint_selection='allcomponentsfixed_final_epoch30;zero_fit_equal_probabilities',
  budget_authorization='userfromscratch30;eachnative30epochs,2new+1verifiedreusedperprimaryseed;3xensembletraining/params/inferencecostdisclosed',
  initialization='newmembersscratch;member0inferenceonlymatchingCE30reuse')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Final ensemble preregistration differs')
 return c
def member_config(row,member):
 if member not in (1,2):raise ValueError('Only new internal members train')
 c=deepcopy(config(row));c.update(member=member,member_init_seed=pilot.member_seed(row['model_seed'],member),
  output=(BASE/row['row_id']/'source'/f'member{member}').as_posix(),initialization='scratchnativeidentity',checkpoint_selection='fixed_final_epoch30',training_target_access=False)
 return c
def validate_member(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c.get('member') not in (1,2) or c!=member_config(row,c['member']):raise ValueError('New final member config differs')
 return c
