from copy import deepcopy
from pathlib import Path
from experiments.cvs_finite_margin import design as f
from experiments.cvs_nocfo_foundation import design as n
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,RX=(getattr(f,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','RX'))
RUN='20261012-phase1-cosine-prototype-confirm-source-manysig-m10-r01';RELEASE='cvs_proto_confirmation_20261012_r01';OWNER='codex/root/prototype-confirmation-20261012';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092702,2026092703);ARMS=('NATIVE_HEAD','COSINE_PROTO')
def rows():return [dict(row_id=f'PROTO-RX{rx}-s{s}',held_rx=rx,model_seed=s,arm='CLEAN_REPLAY') for s in SEEDS for rx in RX]
def config(row):
 if row not in rows():raise ValueError('Unregistered prototype confirmation row')
 c=deepcopy(f.config(next(r for r in f.rows() if r['held_rx']==row['held_rx'] and r['arm']=='CLEAN_REPLAY')))
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_loader_config=n.config(next(r for r in n.rows() if r['model_seed']==row['model_seed'])),source_validation='source heldRX only; fixedE20 followed by fitL cosine prototype; no target',prototype=dict(fit='same fourRX L5040 only after sourceE20 frozen',features='actual normalized joint160',classifier='cosine to normalized class mean, identical formula all classes',learned_temperature=False,head_optimizer=False),scientific_rule=dict(each_seed_mean_gain=.02,total_nonnegative_RX=8,worst_RX_noninferiority_each_seed=.01))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Prototype confirmation config differs')
 return c
