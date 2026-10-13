"""Matched temporal coverage x branch nonlinearity, original head only."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
from experiments.cvs_pa_cross_delay import design as pa
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-time-factorial-oldgate-cos-e50-clean-manysig-m15-r01'
RELEASE='cvs_time_factorial_20261014_r01'
OWNER='codex/root/time-factorial-20261014'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('OLD_GATE_COS50','COMPACT_LINEAR','LONG_LINEAR','COMPACT_SILU','LONG_SILU')
PRIMARY='LONG_SILU';BASELINE_ARM='OLD_GATE_COS50'
PAIRS=[('LONG_LINEAR','COMPACT_LINEAR'),('LONG_SILU','COMPACT_SILU'),('COMPACT_SILU','COMPACT_LINEAR'),('LONG_SILU','LONG_LINEAR')]+[(a,BASELINE_ARM) for a in ARMS[1:]]
READOUTS=('ORIGINAL_HEAD',);VIEWS=('clean',);QUERY_COUNT=168000
SOURCE_DOMAINS=15
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
BASELINE_RUN=old.RUN;BASELINE_EVAL_RUN=old.RUN
LEGACY_FIELDS=dict(pa.LEGACY_FIELDS)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def baseline_config(row):return old.config(dict(row_id=f"{BASELINE_ARM}-s{row['model_seed']}",arm=BASELINE_ARM,model_seed=row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered TIME factorial row')
 prior=baseline_config(row);c=deepcopy(prior);arm=row['arm']
 c.update(schema='time_factorial_default_v1',run_id=RUN,**row,method='cvs_time_factorial',stage='evaluation_reuse' if arm==BASELINE_ARM else 'time_factorial',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_config=prior if arm==BASELINE_ARM else None,
   model_initialization='historical own final50 comparison only' if arm==BASELINE_ARM else 'scratch',
   time_factorial=dict(enabled=arm!=BASELINE_ARM,layers=['t1','t2','t3'],settings=[[3,1],[5,1],[5,2 if arm.startswith('LONG') else 1]],branch_activation='silu' if arm.endswith('SILU') else 'none',preserve=['PA','frequency','reference','GroupNorm','output_ReLU','pool','dropout','default_head'],private_init_offset=62103,added_identity_parameters=60672,same_parameter_values_across_four_arms=True,same_initial_function_across_activations=False),
   readouts=list(READOUTS),classifier_training='default jointly trained only; separate head updates0',source_V='health only; no gradients or selection',
   source_falsification=dict(first_seed=SEEDS[0],maximum_final_L_CE=.5,metric='mean train/loss_tx_labeled over final222updates; no target or V feedback'),
   goal=dict(primary=PRIMARY,readout='ORIGINAL_HEAD',clean_three_seed_mean=.84,feedback_to_training=False),
   historical_TIME_reuse=False,historical_TIME_reuse_reason='Matched private initialization preserves outer Torch RNG; historical module initialization consumed outer RNG. Fresh LONG_LINEAR control required.',
   classifier='unchanged ResidualPhysicalHead cosine6x160 scale30; joint normal training; no separate fit')
 c.pop("feature_extractor",None)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row) or c['epochs']>50 or c['readouts']!=['ORIGINAL_HEAD']:raise ValueError('Fixed TIME/default-head factorial config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['arm']==BASELINE_ARM:raise ValueError('Historical comparison is never retrained')
 a=old.make_args(baseline_config(c),device)
 a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.seed=c['model_seed']
 return a
