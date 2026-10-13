"""Fixed first-seed source derivatives; no training or target test."""
from pathlib import Path
from experiments.cvs_neighbor_pair50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-feature-gradient-manysig-s01-m3-r01'
RELEASE='cvs_source_feature_gradient_20261014_r01';OWNER='codex/root/source-feature-gradient-20261014'
BASE=Path(PROJECT)/'runs'/RUN
SEED=SEEDS[0]
def rows():return [r for r in parent.rows() if r['model_seed']==SEED]
def source_config(row):return parent.config(row)
def config(row):
 if row not in rows():raise ValueError('Unregistered fixed first-seed source derivative row')
 return dict(schema='source_feature_gradient_v1',run_id=RUN,**row,source_config=source_config(row),output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_role='L_s',source_count=6300,source_sampling='PAIRS full L6300 once, native metadata fields only',shuffle_seed=SEED+75001,strong_seed=SEED+85001,batch=256,model_mode='eval for student and ownEMA; source derivative observer, not recorded train/dropout gradients',losses=dict(L_CE='clean true source label CE mean, smoothing .01',U_CE='strong ownEMA pseudo CE mean over full original gate mask, .16 weight',entropy='strong entropy mean over all source proxies, .01 weight'),feature_boundary='Actual identity subtree176065, exclude only class W960; keep head feature norms/gain/reference',training_epochs=0,optimizer_updates=0,new_head_fit=False,U_IQ_read=False,U_truth_read=False,V_forward=False,V_gradients=False,V_cache_materialized=True,query_read=False,stop_rule='Any provenance/role/shape/nonfinite/scope/state mutation failure preserves outputs and stops remaining rows; no automatic retry')
def validate(c):
 if c!=config(next((r for r in rows() if r['row_id']==c.get('row_id')),{})):raise ValueError('Frozen source derivative plan differs')
