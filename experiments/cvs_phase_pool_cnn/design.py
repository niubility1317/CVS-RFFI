from copy import deepcopy
from pathlib import Path
from experiments.cvs_complement_cnn import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SEED,RX,REFERENCE=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SEED','RX','REFERENCE'))
RUN='20261012-phase1-phase-pool-cnn-source-manysig-s2026092701-r01';RELEASE='cvs_phase_pool_cnn_20261012_r01';OWNER='codex/root/phase-pool-cnn-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('POOL_ONLY','FLAT_PHASE','POOL_PHASE')
def rows():return [dict(row_id=f'{a}-RX{rx}-s{SEED}',arm=a,held_rx=rx,model_seed=SEED) for rx in RX for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered phase/poolsource row')
 c=deepcopy(old.config(next(r for r in old.rows() if r['held_rx']==row['held_rx'])));c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
  pool=row['arm'] in ('POOL_ONLY','POOL_PHASE'),phase_train=row['arm'] in ('FLAT_PHASE','POOL_PHASE'),
  mechanism=dict(pooling='concat perchannel time mean and populationstd256 ->256->160' if row['arm'] in ('POOL_ONLY','POOL_PHASE') else 'originalflatten2048->256->160',phase='independent uniform[-pi,pi] commoncomplexunitrotation eachLpacket/step ifenabled;no frequency/rx/TX/targetlabels',augmentation_count=852480 if row['arm'] in ('FLAT_PHASE','POOL_PHASE') else 0),
  goal=dict(primary='fixedsourcequalified phase/pool CNN mixture',scope='source_only_not84',feedback_to_training=False),
  source_selection='eachmixture vsnative3 mean+1pp,4of5nonnegative,worstnoninferior1pp;amongqualifying choosehighest source mean,tie minparameters thenarmname;completeall15 first',
  budget_authorization='user scratch30;15newCNN99900updates;reusefrozen5FLATcontrol33300+15native99900,notequalcost mainclaim')
 c['cnn'].update(pool='moments' if c['pool'] else 'flatten',flatten_width=256 if c['pool'] else 2048);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed phase/poolsource matrix differs')
 return c
