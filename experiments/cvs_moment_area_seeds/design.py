"""Unchanged seed02/03 expansion, conditional on frozen first-seed source rules."""
from pathlib import Path
from experiments.cvs_moment_area50 import design as first
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(first,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s0203-m8-r01'
RELEASE='cvs_moment_area_seeds_20261014_r01';OWNER='codex/root/moment-area-seeds-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092702,2026092703)
FEATURE_CONTROL=first.FEATURE_CONTROL;FEATURE_PARAMETER_COUNT=first.FEATURE_PARAMETER_COUNT
ARMS,BASELINE_ARM,PRIMARY,PAIRS,READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(first,k) for k in ('ARMS','BASELINE_ARM','PRIMARY','PAIRS','READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
baseline_config=first.baseline_config
def config(row):
 if row not in rows():raise ValueError('Unregistered remaining signature-statistics row')
 return dict(first.configuration(row,run=RUN,base=BASE,owner=OWNER),source_qualification_run=first.RUN,source_expansion_rule='Only exact frozen first-seed source PASS, no target metric reads; all methods/configuration/epochs unchanged',first_seed_retrained=False,combined_model_seeds=[2026092701,2026092702,2026092703])
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed remaining signature-statistics config differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical baseline cannot be retrained')
 a=first.first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
