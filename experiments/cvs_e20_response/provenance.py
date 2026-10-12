from pathlib import Path
from . import design as d
from .model import expected_parent
from experiments.cvs_nocfo_engineering.teacher import expected_proof,check_proof
def expected_sources(c):return [expected_parent(c)['checkpoint'],*[expected_proof(s)['checkpoint'] for s in c['teacher']['seeds']]]
def teacher_provenance(c,value):
 wanted={str(s):expected_proof(s) for s in c['teacher']['seeds']}
 if value!=wanted:raise ValueError('Fixed peer ancestry differs')
 return value
def initialization(c,parent,teachers,response):
 if parent!=expected_parent(c):raise ValueError('CE parent proof differs')
 teacher_provenance(c,teachers)
 return dict(parent_provenance=parent,teacher_provenance=teachers,response_provenance=response,checkpoint_sources=expected_sources(c),source_roles='EXACT_MATCH',parent_training_epochs=20,incremental_epochs=20,total_epochs=40,scratch_only=False,reference_E20_loaded=True,optimizer_reset=True,target_training_or_selection=False)
def checkpoint_provenance(c,ck):
 wanted=dict(config=c,model_seed=c['model_seed'],epoch=20,total_epochs=40,parent_training_epochs=20,incremental_epochs=20,optimizer_steps=4440,lifetime_optimizer_steps=8880,run_id=d.RUN,candidate_id=c['row_id'],parent_checkpoint=c['parent_checkpoint'],checkpoint_selection='final_only',target_access=False,scratch_only=False,reference_E20_loaded=True)
 if any(ck.get(k)!=v for k,v in wanted.items()) or ck.get('parent_provenance')!=expected_parent(c) or 'model' not in ck or 'baseline_args' not in ck:raise ValueError('Own continuation checkpoint differs')
 teacher_provenance(c,ck.get('teacher_provenance',{}))
 if ck.get('checkpoint_sources')!=expected_sources(c):raise ValueError('Own checkpoint ancestor list differs')
 validate_response(c,ck.get('response_provenance'))
def validate_response(c,proof):
 if not c['response']['enabled']:
  if proof is not None:raise ValueError('Unexpected response checkpoint')
  return
 wanted=dict(status='VERIFIED',run_id=c['response']['source_run'],execution_commit=d.FINITE_EXECUTION_COMMIT,seed=c['model_seed'],split='PID80_20',source_only=True,identity_updates=0,target_access=False,reference_parent=expected_parent(c),tasks=c['response']['tasks'],source_root=c['response']['source_root'],auxiliary_trainable_parameters=0)
 if not isinstance(proof,dict) or any(proof.get(k)!=v for k,v in wanted.items()):raise ValueError('Response model ancestry differs')
 if len(proof.get('state_paths',[]))!=4 or proof.get('observations_path')!=str(Path(c['response']['source_root'])/'source_observations.pt'):raise ValueError('Response artifacts differ')
 for path,task in zip(proof['state_paths'],c['response']['tasks']):
  if path!=str(Path(c['response']['source_root'])/f"PID80_20-{task[0]}-{task[1].replace('.','_')}.pt"):raise ValueError('Wrong response state')
