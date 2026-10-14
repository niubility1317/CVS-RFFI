"""Frozen-source report route3 read-position diagnostic; no training or fitting."""
from pathlib import Path
from experiments.cvs_moment_area50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01'
RELEASE='cvs_source_dropout_20261014_r01';OWNER='codex/root/source-dropout-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;ROW='SOURCE_DROPOUT-s2026092701';RX=(1,3,4,6,8);DAYS=(1,2,3);MASK_SEEDS=tuple(SEED+62001+i for i in range(8))
def rows():return [dict(row_id=ROW,arm='SOURCE_DROPOUT',model_seed=SEED)]
def source_config():return parent.config(next(r for r in parent.rows() if r['arm']=='SIG_TIME'))
def config(row):
 if row!=rows()[0]:raise ValueError('Unregistered source Dropout row')
 return dict(schema='source_dropout_descriptor_v1',run_id=RUN,**row,source_config=source_config(),source_execution_commit='512c10a267b5e654cfd2c983ab1debdeebc7f626',output_root=(BASE/ROW).as_posix(),launch_owner=OWNER,source_role='L_s',source_pool_count=6300,source_sample_count=720,samples_per_TX_RX_day=8,source_selection='physical-ID sorted first8 per6TXx5RXx3day cell',mask_seeds=list(MASK_SEEDS),mask_samples=8,native_dropout_probability=.1,observation_layer='t2.pool vs t2 complete eval equality; frozen t2.pool activation with isolated inverted Bernoulli .1 replay',projection='Actual fixed final SIG_TIME single6D Conv1d+tanh',statistics='4x8 windows: feature increment6, feature-feature area15, feature-time area6, full SIG_TIME28; COV_SHAPE27 and AREA_SHAPE27 on the same path; added residual160',model_mode='all eval, frozen; independent CPU replay does not set train mode or update buffers',training_epochs=0,optimizer_updates=0,head_fit=False,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,goal_claim=False,stop_rule='Role/provenance/nonfinite/shape/state mutation failure preserves outputs; no retry, target or hyperparameter feedback')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Frozen source Dropout config differs')
