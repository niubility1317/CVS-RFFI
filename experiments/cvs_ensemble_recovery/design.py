from copy import deepcopy
from pathlib import Path
from experiments.cvs_ensemble_final30 import design as original
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write,SCENES,VIEWS,VIEWS_ROOT,ARMS,READOUTS=(getattr(original,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write','SCENES','VIEWS','VIEWS_ROOT','ARMS','READOUTS'))
RUN='20261012-phase1-ensemble-final30-recovery-manysig-m3-r02';RELEASE='cvs_ensemble_recovery_20261012_r02';OWNER='codex/root/ensemble-recovery-20261012'
BASE=Path(PROJECT)/'runs'/RUN
def rows():return original.rows()
def baseline_config(seed):return original.baseline_config(seed)
def config(row):
 if row not in rows():raise ValueError('Unregistered frozen recovery row')
 source=original.config(row);c=deepcopy(source);c.update(run_id=RUN,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,
  source_config=source,source_output=source['output'],recovery=dict(new_training_epochs=0,new_model_updates=0,source_member_epochs=30,models=3,
   defect='aggregate source contract copied reference instead of runtime evidence',fix='actual member runtime contracts verified against unchanged reference; no fabricated flag/no source mutation'),
  budget_authorization='prediction-and-independent-score-only recovery;reuse all9fixedfinal30 weights;no new training or dataset change')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Recovery fixed matrix differs')
 return c
