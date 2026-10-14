"""Frozen-source report route2 continuous-window geometry; no fitting."""
from pathlib import Path
from experiments.cvs_moment_area50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-hierarchy-geometry-manysig-s01-m1-r01'
RELEASE='cvs_source_hierarchy_20261014_r01';OWNER='codex/root/source-hierarchy-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;ROW='SOURCE_HIERARCHY-s2026092701';RX=(1,3,4,6,8);DAYS=(1,2,3)
def rows():return [dict(row_id=ROW,arm='SOURCE_HIERARCHY',model_seed=SEED)]
def source_config():return parent.config(next(r for r in parent.rows() if r['arm']=='SIG_TIME'))
def config(row):
 if row!=rows()[0]:raise ValueError('Unregistered source Dropout row')
 return dict(schema='source_hierarchy_geometry_v1',run_id=RUN,**row,source_config=source_config(),source_execution_commit='512c10a267b5e654cfd2c983ab1debdeebc7f626',output_root=(BASE/ROW).as_posix(),launch_owner=OWNER,source_role='L_s',source_pool_count=6300,source_sample_count=720,samples_per_TX_RX_day=8,source_selection='physical-ID sorted first8 per6TXx5RXx3day cell',observation_layer='t2 eval32positions, actual final6D Conv+tanh',hierarchy=[4,2,1],boundary_increments=3,time_coordinate='one global linspace0to1 over32 samples; no local resets in correct composition',statistics='Corrected continuous skew-area composition vs direct32 and direct16 paths; missing bridges/reset clock/omitted cross-area controls, with full/RX/day RMS and absolute errors',model_mode='all eval, frozen; CPU observation does not set train mode or update buffers',training_epochs=0,optimizer_updates=0,head_fit=False,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,goal_claim=False,stop_rule='Role/provenance/nonfinite/shape/state mutation failure preserves outputs; no retry, target or hyperparameter feedback')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Frozen source Dropout config differs')
