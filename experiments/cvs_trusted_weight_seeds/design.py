"""Unchanged predefined02/03, only after the frozen first source verdict."""
import copy
from pathlib import Path
from experiments.cvs_trusted_weight50 import design as first
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(first,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s0203-m6-r01';RELEASE='cvs_trusted_weight_seeds_20261014_r01';OWNER='codex/root/trusted-weight-seeds-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092702,2026092703);VERIFY_INITIAL_EVAL_EQUAL=True
FEATURE_CONTROL=first.FEATURE_CONTROL;FEATURE_PARAMETER_COUNT=first.FEATURE_PARAMETER_COUNT
ARMS,BASELINE_ARM,PRIMARY,PAIRS,READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(first,k) for k in ('ARMS','BASELINE_ARM','PRIMARY','PAIRS','READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
baseline_config=first.baseline_config
def config(row):
 if row not in rows():raise ValueError('Unregistered remaining trusted weight row')
 baseline=row['arm']==BASELINE_ARM;skeleton=row if baseline else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=first.parent.configuration(skeleton,run=RUN,base=BASE,owner=OWNER);template=first.config(next(r for r in first.rows() if r['arm']==row['arm']))
 for key in ('feature_combination','trusted_weight','hypothesis_basis','source_falsification','goal'):c[key]=copy.deepcopy(template[key])
 c.update(**row,output_root=(BASE/row['row_id']/'source').as_posix(),source_qualification_run=first.RUN,source_expansion_rule='Exact fixed firstsource PASS only; no targetmetric read; unchanged loss/configuration/epochs',first_seed_retrained=False,combined_model_seeds=[2026092701,2026092702,2026092703]);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed remaining trusted weight config differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical baseline cannot be retrained')
 a=first.parent.first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
