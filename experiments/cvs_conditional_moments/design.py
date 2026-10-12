from copy import deepcopy
from pathlib import Path
from experiments.cvs_quality_film import design as q
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SEED,RX,LEVELS=(getattr(q,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SEED','RX','LEVELS'))
RUN='20261012-phase1-conditional-moments-source-manysig-s2026092701-r01';RELEASE='cvs_conditional_moments_20261012_r01';OWNER='codex/root/conditional-moments-20261012';BASE=Path(PROJECT)/'runs'/RUN;ARMS=('UNCONDITIONAL','TX_CONDITIONAL');REFERENCE=q.BASE
def rows():return [dict(row_id=f'{a}-RX{rx}-s{SEED}',arm=a,held_rx=rx,model_seed=SEED) for rx in RX for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered conditional moment source row')
 c=deepcopy(q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==row['held_rx'])));c.pop('quality_film')
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
  alignment=dict(levels=list(LEVELS),descriptor='per-packet channel time mean and population std, concatenated',normalization='per current legal L batch detached mean/std, denominator floor1e-4; training loss only',group='TX then RX' if row['arm']=='TX_CONDITIONAL' else 'RX',min_group_occurrences=2,weight=.1,warmup_epochs=5,ramp_epochs=10,day='pooled training days; no same-day guarantee',penalty='equal eligible TX mean of equal RX centroid squared deviation; equal channel and level mean',persistent_state=False),
  classifier=dict(kind='single ordinary cosine',formula='30 * linear(normalize(joint160,eps1e-4),normalize(weight6x160,eps1e-4))',RX_heads=False,routing=False,ensemble=False,Phase2='discard class weight; retain identity representation; support-only refit'),
  scientific_rule=dict(mean_gain_vs_CE=.01,mean_gain_vs_UNCONDITIONAL=.005,nonnegative_RX_each=4,worst_RX_noninferiority_each=.01),
  goal=dict(primary='TX_CONDITIONAL',scope='source_only_not84',feedback_to_training=False),
  budget_authorization='user scratch30 per model, ordinarycos head;10 source models66600 updates; no teacher/ensemble/new classifier or inherited initialization')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed conditional-moment matrix differs')
 return c
def coefficient(epoch):return .1*min(1.,max(0.,(epoch-5)/10))
