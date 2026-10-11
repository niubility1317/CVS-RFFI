from pathlib import Path
from experiments.cvs_e20_budget_fishr import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SEEDS=parent.PROJECT,parent.SEEDS
read,write=parent.read,parent.write
RUN='20261012-phase1-e20-error-audit-manysig-m3-r01'
RELEASE='cvs_e20_error_audit_20261012_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/e20-error-audit-20261012'

def rows():return [dict(row_id=f'E20-s{s}',model_seed=s) for s in SEEDS]

def config(row):
 if row not in rows():raise ValueError('Unregistered E20 error audit row')
 seed=row['model_seed'];pc=parent.config(next(r for r in parent.rows() if r['arm']=='JOINT' and r['model_seed']==seed))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
  parent_build_config=pc,source_checkpoint=pc['parent_checkpoint'],source_only=True,target_access=False,
  identity_updates=0,head_warmup_steps=200,roles=dict(L=6300,U=56700,V=27000),batch_size=256,
  endpoint_run=parent.RUN,endpoint_arms=list(parent.ARMS),endpoint_selection='all eight fixed rows, no target- or V-based subset',
  confidence=.95,confidence_bins=[0.,.5,.8,.9,.95,.98,.99,1.],
  cohorts=['high_correct','high_error','low_correct','low_error'],
  gradient_rounds=24,gradient_step_start=1332,gradient_step_stride=4,
  gradient_scope='same original E20+L-warmed heads; all8 objectives, independent next-day L reference; actual train dropout; no optimizer or identity update',
  evaluation_scope='all L and V fixed-PID parent/end readouts; V eval/no_grad only; no U truth or target access',
  claim_scope='known-source error correction and gradient compatibility only; all five source RX were seen by parent; not unknown-RX or training-benefit proof',
  output_policy='exclusive new source audit directory; save source probabilities and summaries, no IQ or deployable checkpoint')

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('E20 error audit configuration differs')
 return c
