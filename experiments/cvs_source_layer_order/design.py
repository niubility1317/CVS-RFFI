"""Full legal L coverage, fixed shallow/middle semantic relation observer."""
from pathlib import Path
from experiments.cvs_trusted_weight50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-layer-order-manysig-s01-m1-r01'
RELEASE='cvs_source_layer_order_20261014_r01';OWNER='codex/root/source-layer-order-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;ROW='SOURCE_LAYER_ORDER-s2026092701';RX=(1,3,4,6,8);DAYS=(1,2,3)
LEVELS=('t1','t2','joint160');MARGIN=.1;TEMPERATURE=.2;COEFFICIENT=.1
PARAMETERS=tuple('id_backbone.encoder.core.id_backbone.'+n+'.pw.weight' for n in ('t1','t2','t3'))
def rows():return [dict(row_id=ROW,arm='SOURCE_LAYER_ORDER',model_seed=SEED)]
def source_config():return parent.config(next(r for r in parent.rows() if r['arm']=='SIG_TIME'))
def config(row):
 if row!=rows()[0]:raise ValueError('Unregistered source layer order row')
 return dict(schema='source_layer_order_v1',run_id=RUN,**row,source_config=source_config(),output_root=(BASE/ROW).as_posix(),launch_owner=OWNER,source_role='L_s',source_count=6300,cohort_count=70,cohort_packets=90,sampling='fixed shuffled rank within each TX/RX/day; all70 ranks once, one physical packet per90cell cohort',shuffle_seed=SEED+118011,margin=MARGIN,temperature=TEMPERATURE,coefficient=COEFFICIENT,levels=list(LEVELS),local_view='adaptive average pool to4 ordered temporal bins, flatten existing t1/t2 output, per-packet L2 eps1e-4; no projection/statistical fit',relations='same-day i/p sameTX crossRX; n differentTX and sameRX/day as p; all1800 triples percohort, satisfied terms stay in denominator',supcon='per anchor/referenceRX, six true-TX reference packets including sameTX positive; normalized cosine logsumexp/temperature; all360 reference groups percohort',gradient_parameters=list(PARAMETERS),training_epochs=0,optimizer_updates=0,head_fit=False,U_IQ_read=False,U_truth_read=False,query_read=False,V_cache_materialized=True,V_forward=False,early_middle='not available from frozen final checkpoint; real epochs1/17/33/50 to be observed only in subsequent preregistered training',stop_rule='Role/provenance/shape/nonfinite/state mutation failure stops, preserves outputs; no automatic retry or target feedback')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Frozen source layer order plan differs')
