from pathlib import Path
from experiments.cvs_e20_cross_style import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SEEDS=parent.PROJECT,parent.SEEDS
read,write=parent.read,parent.write
RUN='20261012-phase1-e20-direction-audit-manysig-m3-r01'
RELEASE='cvs_e20_direction_audit_20261012_r01'
OWNER='codex/root/e20-direction-audit-20261012'
BASE=Path(PROJECT)/'runs'/RUN

def rows():return [dict(row_id=f'DIRECTION-s{s}',model_seed=s) for s in SEEDS]

def config(row):
 if row not in rows():raise ValueError('Unregistered source direction audit')
 seed=row['model_seed'];ce=parent.config(next(r for r in parent.rows() if r['arm']=='CE' and r['model_seed']==seed))
 ml=parent.config(next(r for r in parent.rows() if r['arm']=='F_BOTH' and r['model_seed']==seed))['multilevel']
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
  parent_build_config=ce,multilevel=ml,models=['P0','CE20'],checkpoint_sources=[ce['parent_checkpoint'],(Path(ce['output_root'])/'final_ssdg.pth').as_posix()],
  endpoint_run=parent.RUN,source_only=True,target_access=False,identity_updates=0,head_warmup_steps_per_model=200,
  source_role='L_s',source_L_count=6300,U_forward=0,V_forward=0,
  levels=['time_down','t2.pool'],rounds=36,strengths=[.3,1.],level_sets=[['time_down'],['t2.pool'],['time_down','t2.pool']],
  style=dict(p=.5,alpha=.1,same_TX=True,same_day=True,cross_RX=True),
  gradient_scope='full identity parameters and fixed shallow/middle parameter groups; actual autograd, no virtual-affine substitution',
  dropout_control='primary eval; secondary frozen-buffer train dropout with paired identical RNG and independent-mask control',
  calibration='fresh source-L heads200 plus old-pair gradient fixed calibration12; original identity unchanged',
  claim_scope='source controlled identity difficulty/gradient interaction; all source RX seen, not unknown-RX or target improvement proof',
  output_policy='exclusive new diagnostic output; statistics and source logs only; no checkpoints/IQ/maps saved')

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Source diagnostic config differs')
 return c
