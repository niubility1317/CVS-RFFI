from pathlib import Path
from experiments.cvs_stage2_rx_risk import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS=parent.PROJECT,parent.SOURCE,parent.SEEDS
read,write=parent.read,parent.write
RUN='20261011-phase1-rx-update-utility-manysig-m3-r01'
RELEASE='cvs_rx_update_utility_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN;OWNER='codex/root/rx-update-utility-20261011'
POLICIES=('clean','uniform','risk_weighted','hard_margin','L_selected')
def rows():return [dict(row_id=f'E200-s{s}',model_seed=s) for s in SEEDS]
def config(row):
 if row not in rows():raise ValueError('Unregistered utility row')
 pc=parent.config(next(r for r in parent.rows() if r['arm']=='RFIX' and r['model_seed']==row['model_seed']))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,parent_build_config=pc,
  source_only=True,target_access=False,identity_updates=0,roles=dict(L=6300,V=27000,U_used=0),
  budget=dict(batches=30,update_per_class=5,probe_per_class_RX_day=2,relative_steps=[1e-4,1e-3],candidates=8),
  sampling='cycle all15 source RX/day cells twice; 30 update L; each probe96, other4RX x other2days x6TX x2; L/V disjoint; fixed seeded random draws',
  update='deterministic eval gradients; dense identity only, native Hz excluded; .5 clean CE+.5 action CE; group LR ratio1:3; normalize dense displacement to fixed relative L2; restore all parameters and buffers after every virtual update',
  policies=list(POLICIES),risk='same-seed pretrained frozen RANK_STATE with fixed L normalization; no new risk training',
  utility_selection='select one batch-level candidate by smallest L-probe post-update CE separately per fixed step size; V never used for selection or gradients',
  scope='mechanistic finite-step source transfer diagnostic, not AdamW replay, actual training, or unknown-RX validation; all fixed policies and scales reported')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Utility configuration differs')
