"""Evaluation-only continuation of fixed frequency checkpoints; no retraining."""
from pathlib import Path
from experiments.cvs_frequency_context50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,SEEDS,CLASSES,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','SEEDS','CLASSES','FULL_FP32_POLICY','read','write'))
SOURCE_RUN=parent.RUN
SOURCE_BASE=parent.BASE
RUN='20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01'
RELEASE='cvs_frequency_context_eval_recovery_20261013_r01'
OWNER='codex/root/frequency-context-eval-recovery-20261013'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE_CONFIGS={r['row_id']:parent.config(r) for r in parent.rows()}
ARMS=parent.ARMS;PRIMARY=parent.PRIMARY;VIEWS=parent.VIEWS;QUERY_COUNT=parent.QUERY_COUNT
def rows():return parent.rows()
def config(row):
 if row not in rows():raise ValueError('Unregistered recovery row')
 return SOURCE_CONFIGS[row['row_id']]
def configure_runtime():
 # Source proofs/checkpoint metadata continue to use the original exact config.
 # Only prediction/score locations move to the new immutable evaluation run.
 parent.BASE=BASE;parent.config=config
