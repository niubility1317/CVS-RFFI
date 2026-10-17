"""Finish the five pre-fixed E30 checkpoints;no training or checkpoint selection."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01'
RELEASE='cvs_F_clock_E30_score_recovery_20261017_r01';LOCAL='local_artifacts/F_clock_e30_score_recovery_20261017';OWNER='codex/root/F-E30-score-recovery'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE=trained.AF_RELEASE
CLASSES,FULL_FP32_POLICY,SCENES,VIEWS,FIT=trained.CLASSES,trained.FULL_FP32_POLICY,trained.SCENES,trained.VIEWS,trained.FIT
BASE=Path(PROJECT)/'runs'/RUN;SEED=trained.SEED;ARMS=trained.ARMS
TRAIN_RUN='20261017-phase1-F-crossRX-clock-factorial-e20plus10-s2026092701-m5-r03'
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return trained.rows()
def train_rows():return rows()
def training_config(row):return trained.config(row)
def config(row):
 c=deepcopy(training_config(row));c.update(run_id=RUN,schema='E30_FIXED_CHECKPOINT_SCORE_RECOVERY_V1',training_run_id=TRAIN_RUN,training_config=training_config(row),source_root=training_config(row)['output_root'],launch_owner=OWNER,recovery='only final checkpoint default-head predictions;0newtraining',new_optimizer_updates=0,new_model_epochs=0)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source checkpoint/evaluation contract differs')
 return c
