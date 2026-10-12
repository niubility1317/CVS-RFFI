from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch30_baseline import design as b
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(b,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261012-phase1-quality-film-source-manysig-s2026092701-r01';RELEASE='cvs_quality_film_20261012_r01';OWNER='codex/root/quality-film-20261012';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;RX=(1,3,4,6,8);ARMS=('CE30','FREE_FILM','QUALITY_FILM');LEVELS=('time_down','t2.pool')
def rows():return [dict(row_id=f'{a}-RX{rx}-s{SEED}',arm=a,held_rx=rx,model_seed=SEED) for rx in RX for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered quality source row')
 c=deepcopy(b.config(next(r for r in b.rows() if r['arm']=='CE30' and r['model_seed']==SEED)))
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_only=True,test_views=[],goal=dict(primary='QUALITY_FILM',scope='source_only_not84',feedback_to_training=False),held_rx_excluded_before_training=True,quality_film=dict(levels=list(LEVELS),hidden=16,log_scale_bound=.2,shift_bound=.1,quality='log packetRMS, FFT powerflatness, abs lag16coherence; no CFO or verified-preamble claim',quality_normalization='mean/populationstd from fourRX fitL only, fixed; clip4',quality_clip=4.,all_TX_share_condition_transform=True,initialization='zero final generator makes exact identity',control='same bounded FiLM, learned3dim context from local channelmean/std; slightly more params',scientific_rule=dict(mean_gain_vs_CE=.01,mean_gain_vs_FREE=.005,positive_RX_vs_CE=4,positive_RX_vs_FREE=4,worst_RX_noninferiority_CE=.01,worst_RX_noninferiority_FREE=.01)),source_validation='heldRX V only after final30; no target; originalroles preserved',sampling='fourRX/day balanced fitL5040; no heldRX/teacher/direction/statistics leakage',budget_authorization='user from scratch30epochs; single-seed sourcefalsification of report conditional-feature hypothesis')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Quality-FiLM source config differs')
 return c
