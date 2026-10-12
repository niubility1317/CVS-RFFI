from pathlib import Path
from copy import deepcopy
from experiments.cvs_quality_film import design as q
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(q,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261012-phase1-relation-complement-source-manysig-s2026092701-r02'
RELEASE='cvs_relation_complement_20261012_r02';OWNER='codex/root/relation-complement-20261012-r02'
BASE=Path(PROJECT)/'runs'/RUN;REFERENCE=q.BASE
RX=q.RX;SEED=q.SEED;ARMS=('RAW_REL_BRANCH','COMPLEX_REL_BRANCH')
RULE=dict(mean_gain_vs_CE=.01,nonnegative_RX_vs_CE=4,worst_RX_noninferiority=.01,
 invariant_attribution_mean_gain_vs_RAW=.005,invariant_attribution_nonnegative_RX=4)
def rows():return [dict(row_id=f'{a}-RX{rx}-s{SEED}',arm=a,held_rx=rx,model_seed=SEED) for rx in RX for a in ARMS]
def baseline_id(rx):return f'CE30-RX{rx}-s{SEED}'
def config(row):
 if row not in rows():raise ValueError('Unregistered complementary source row')
 c=deepcopy(q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==row['held_rx'])))
 c.pop('quality_film');c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
  goal=dict(primary='source performance selection among eligible branches',scope='source_only_not84',feedback_to_training=False),
  relation_branch=dict(projection_width=32,projection_seed=2026101207,channels=4,conv_channels=[16,24],pool_grid=2,
   feature_width=160,fusion='native joint160 plus .2*tanh(gain160)*relation160, shared original cosine head',initialization='zero fusion gain; complete native initial function',
   representation='normalized projected IQ outer products' if row['arm']=='RAW_REL_BRANCH' else 'normalized projected complex Gram real/imag repeated; weak constant complex gain/phase invariant channel only',
   frequency_handling='both branches same fixed per-packet lag20 phase slope compensation on window80:160 before full256 projection; no frequency scalar to classifier; fixed numerical convention, not verified preamble',
   whole_model_invariant=False,physical_RX_invariant=False,source_stats_fit=False,auxiliary_loss=False),
  baseline_ref=(REFERENCE/baseline_id(row['held_rx'])).as_posix(),scientific_rule=RULE,
  budget_authorization='user scratch30;10 fresh full native+branch models;5 matched CE30 source artifacts reused only for comparison, no initialization',
  initialization='full native identity and relation branch scratch; no loaded weights')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Complementary relation source config differs')
 return c
