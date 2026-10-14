"""Balanced-source derivative variance of real shallow/middle f_theta weights."""
from pathlib import Path
from experiments.cvs_identity_mixstyle50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-gradient-variance-manysig-s01-m1-r01'
RELEASE='cvs_source_gradient_variance_20261014_r01';OWNER='codex/root/source-gradient-variance-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;ROW='SOURCE_GRADVAR-s2026092701';RX=(1,3,4,6,8);DAYS=(1,2,3)
PARAMETERS=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight')
def rows():return [dict(row_id=ROW,arm='SOURCE_GRADVAR',model_seed=SEED)]
def source_config():return parent.baseline_config(next(r for r in parent.rows() if r['arm']==parent.BASELINE_ARM))
def config(row):
 if row!=rows()[0]:raise ValueError('Unregistered source variance row')
 return dict(schema='source_gradient_variance_v1',run_id=RUN,**row,source_config=source_config(),output_root=(BASE/ROW).as_posix(),launch_owner=OWNER,source_role='L_s',source_pool_count=6300,samples_per_TX_RX_day=2,source_sample_count=180,shuffle_seed=SEED+97501,domains='5 actual source RX; 36 packets each, balanced6TXx3daysx2',gradient_parameters=list(PARAMETERS),gradient_elements=[3456,6912],loss='per-example default cosine CE label_smoothing .01; no batch-mean scaling',statistics='unbiased coordinate gradient variance within RX; raw and TX/day-centered residual; no cross-layer raw-unit ranking',model_mode='eval; read-only derivative observer',training_epochs=0,optimizer_updates=0,head_fit=False,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,stop_rule='Role/provenance/shape/nonfinite/state mutation failure preserves outputs and stops; no automatic retry; no target feedback')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Frozen source variance config differs')
