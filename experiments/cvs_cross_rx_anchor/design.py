from pathlib import Path
from experiments.cvs_stage2_teacher_factorial import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT=parent.PROJECT;SEEDS=parent.SEEDS
RUN='20261011-phase1-cross-rx-anchor-manysig-m3-r01'
RELEASE='cvs_cross_rx_anchor_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/cross-rx-anchor-20261011'
read,write=parent.read,parent.write
def rows():return [dict(row_id=f'E200-s{s}',model_seed=s) for s in SEEDS]
def config(row):
 if row not in rows():raise ValueError('Unregistered cross-RX row')
 pc=parent.config(next(r for r in parent.rows() if r['arm']=='O_DIRECT' and r['model_seed']==row['model_seed']))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),parent_build_config=pc,source_checkpoint=pc['parent_checkpoint'],source_only=True,target_access=False,identity_updates=0,launch_owner=OWNER,
  anchor=dict(k_per_RX=32,temperature=.07,classes=6,metric_modes=['cosine','within_class_whitened'],covariance_shrinkage=.1,fit_role='L_s only',query_roles=['U_s','V'],same_RX_excluded_for_cross=True,equal_RX_weight=True,mix=.5,consensus_required='all other four RX vote same TX',consensus_min_confidence=.95),
  role_counts=dict(L=6300,U=56700,V=27000),interpretation='Parent already saw all five source RX; this is held-out anchor RX, not unseen-RX generalization',selection='all fixed readouts reported; no target access or automatic training choice')
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Cross-RX config differs')
