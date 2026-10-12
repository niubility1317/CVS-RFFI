from pathlib import Path
from experiments.cvs_nocfo_foundation import design as f
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(f,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261012-phase1-wave-relation-source-manysig-s2026092701-r01'
RELEASE='cvs_wave_geometry_20261012_r01'
OWNER='codex/root/wave-geometry-20261012'
BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701
ARMS=('IQ','RAW_REL','COMPLEX_REL','GL2_REL')
RX=(1,3,4,6,8)
def rows():return [dict(row_id=f'RX{r}-s{SEED}',held_rx=r,model_seed=SEED) for r in RX]
def config(row):
 if row not in rows():raise ValueError('Unregistered source holdout')
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,arms=list(ARMS),source_contract=SOURCE,source_only=True,target_access=False,checkpoint_sources=[],backbone=False,teacher=False,U_access=False,V_updates=False,source_loader_config=f.config(f.rows()[0]),epochs=20,batch_size=128,peak_lr=.001,min_lr=.000001,weight_decay=.0001,gradient_clip=10.,projection_seed=2026101207,projection_width=32,feature_width=4096,head_hidden=[96,64],selection='fixed epoch20; no early selection',held_rx_excluded='all L fitting, normalization, backbone, teacher, direction and head; only source heldRX L/V evaluation',roles='existing L6300/U56700/V27000 unchanged; omit one RX from L fitting; U not opened',scientific_rule=dict(mean_gain_vs_RAW_REL=.01,nonnegative_folds=4,worst_RX_noninferiority=.01,mean_accuracy_floor=.30),budget='20 actual full L-fit passes, ceil(Nfit/128) updates each; all arms same ordering')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Source holdout config differs')
 return c
