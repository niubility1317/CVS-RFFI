from pathlib import Path
from copy import deepcopy
from experiments.cvs_quality_film import design as q
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(q,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261012-phase1-stochastic-ensemble-source-manysig-s2026092701-r01';RELEASE='cvs_stochastic_ensemble_20261012_r01'
OWNER='codex/root/stochastic-ensemble-20261012';BASE=Path(PROJECT)/'runs'/RUN;REFERENCE=q.BASE
SEED=q.SEED;RX=q.RX;ARMS=('MEMBER1','MEMBER2')
def member_seed(primary,member):return primary+10000*member
def baseline_id(rx):return f'CE30-RX{rx}-s{SEED}'
def rows():return [dict(row_id=f'MEMBER{m}-RX{rx}-s{SEED}',arm=f'MEMBER{m}',held_rx=rx,model_seed=SEED,member=m) for rx in RX for m in (1,2)]
def config(row):
 if row not in rows():raise ValueError('Unregistered ensemble source row')
 c=deepcopy(q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==row['held_rx'])));c.pop('quality_film')
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,member_init_seed=member_seed(SEED,row['member']),
  ensemble=dict(members=[0,1,2],member0='matched frozen CE30 comparison and inference only; never teacher or initialization',
   formula='uniform mean softmax(logits),temperature1,no fitted weights/calibration',model_count=3,
   primary_seed=SEED,initialization_seeds=[member_seed(SEED,i) for i in (0,1,2)],sampler_seed=SEED+101,
   dropout_seed='actual member initialization seed; distinct member streams',scientific_rule=dict(mean_gain_vs_mean_component=.01,nonnegative_RX=4,worst_RX_noninferiority=.01)),
  goal=dict(primary='fixed equal probability ensemble',scope='single ensemble source pilot not three replicated seeds or84',feedback_to_training=False),
  initialization='fresh native identity per member, derived internal initialization seed; no inherited checkpoint',
  baseline_ref=(REFERENCE/baseline_id(row['held_rx'])).as_posix(),
  budget_authorization='user scratch30;10 fresh members66600updates,5 matched members33300 historical updates;ensemblecost3xsingle,notmatchedcost advantage')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source ensemble config differs')
 return c
