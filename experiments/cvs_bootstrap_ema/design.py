from copy import deepcopy
from pathlib import Path
from experiments.cvs_quality_film import design as q
from experiments.cvs_scratch30_baseline.design import U_CACHE
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(q,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261012-phase1-bootstrap-ema-source-manysig-s2026092701-r01';RELEASE='cvs_bootstrap_ema_20261012_r01';OWNER='codex/root/bootstrap-ema-20261012';BASE=Path(PROJECT)/'runs'/RUN
SEED=q.SEED;RX=q.RX;ARMS=('SELF_EMA','BOOTSTRAP_EMA')
def rows():return [dict(row_id=f'{a}-RX{rx}-s{SEED}',arm=a,held_rx=rx,model_seed=SEED) for rx in RX for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered source EMA row')
 c=deepcopy(q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==row['held_rx'])));c.pop('quality_film')
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_U=True,U_cache=U_CACHE.as_posix(),source_only=True,test_views=[],goal=dict(primary='BOOTSTRAP_EMA',scope='source_only_not84',feedback_to_training=False),
  semi=dict(warmup_steps=1110,ramp_steps=1110,weight=.5,confidence=.90,min_passes=2,noise_std=.015,EMA_decay=.99,EMA_updates=6660,EMA_initialization='current random student, no checkpoint',
   anchor=row['arm']=='BOOTSTRAP_EMA',anchor_weight='0.8 at step1111, linear to0 at step4440; exactly0 afterwards',anchor_formula='three frozen matched fitRX native softmax T1 equal mean; U only, no labels',prediction='fixed final30 student, not EMA'),
  scientific_rule=dict(mean_gain_CE=.01,mean_gain_SELF=.005,positive_RX_CE=4,positive_RX_SELF=4,worst_RX_CE_noninferiority=.01,worst_RX_SELF_noninferiority=.01),
  source_validation='heldRX V once after30; heldRX excluded from fitL and all U/teacher consumption',sampling='fitL5040,fitU45360,heldV5400;original20percentphysicalroles;wholepass domain balanced',
  initialization='random native student;frozen fitRX ensemble only pseudo target ancestor in bootstrap;EMA currentstudent',budget_authorization='user scratch30;2arms x5RX x6660 updates;teacher historical90modelepochs perRX explicitly additional')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source EMA configuration differs')
 return c
