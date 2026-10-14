from pathlib import Path
from experiments.cvs_neighbor_pair50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-style-profile-manysig-s01-m1-r01'
RELEASE='cvs_source_style_profile_20261014_r01';OWNER='codex/root/source-style-profile-20261014'
BASE=Path(PROJECT)/'runs'/RUN
SEED=parent.SEEDS[0]
def rows():return [dict(row_id='SOURCE_STYLE-s'+str(SEED),model_seed=SEED)]
def source_config(row):return parent.config(dict(row_id='OLD_GATE_COS50-s'+str(SEED),arm='OLD_GATE_COS50',model_seed=SEED))
def config(row):
 if row not in rows():raise ValueError('Unregistered source style row')
 return dict(schema='source_style_profile_v1',run_id=RUN,**row,source_config=source_config(row),output_root=(BASE/row['row_id']).as_posix(),source_role='L_s',source_count=6300,layers=['t1','t2'],batch=128,shuffle_seed=SEED+95001,p=.5,alpha=.1,strength=1.,donor_rule='same true L TX/day, different source RX/real PID, fallback self',source_only=True,optimizer_steps=0,head_fit=False,query_read=False,U_truth_read=False,V_forward=False,V_gradients=False,V_cache_materialized=True)
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Frozen source style profile differs')
