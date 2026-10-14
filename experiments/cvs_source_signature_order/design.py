"""Full legal L coverage, fixed shallow/middle semantic relation observer."""
from pathlib import Path
from experiments.cvs_local_order50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-signature-order-manysig-s01-m1-r01'
RELEASE='cvs_source_signature_order_20261014_r01';OWNER='codex/root/source-signature-order-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;ROW='SOURCE_SIGNATURE_ORDER-s2026092701';RX=(1,3,4,6,8);DAYS=(1,2,3)
LEVELS=('T2_WINDOW','SIG_RAW','SIG_DYNAMIC','JOINT160');MARGIN=.1;TEMPERATURE=.2;COEFFICIENT=.1
PARAMETERS=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.views.0.local_projection.weight')
def rows():return [dict(row_id=ROW,arm='SOURCE_SIGNATURE_ORDER',model_seed=SEED)]
def source_config():return parent.config(next(r for r in parent.rows() if r['arm']=='SIG_TIME'))
def config(row):
 if row!=rows()[0]:raise ValueError('Unregistered source layer order row')
 return dict(schema='source_signature_order_v1',run_id=RUN,**row,source_config=source_config(),output_root=(BASE/ROW).as_posix(),launch_owner=OWNER,source_role='L_s',source_count=6300,cohort_count=70,cohort_packets=90,sampling='fixed shuffled rank within each TX/RX/day; all70 ranks once, one physical packet per90cell cohort',shuffle_seed=SEED+118011,margin=MARGIN,temperature=TEMPERATURE,coefficient=COEFFICIENT,levels=list(LEVELS),local_view='actual deployed SIG112 vs remove4constant time-step coordinates only SIG108, four-bin t2control384 and deployedjoint160, all L2eps1e-4; no fitted projection/statistics',relations='same-day i/p sameTX crossRX; n differentTX and sameRX/day as p; all1800 triples percohort, satisfied terms stay in denominator',supcon='per anchor/referenceRX, six true-TX reference packets including sameTX positive; normalized cosine logsumexp/temperature; all360 reference groups percohort',gradient_parameters=list(PARAMETERS),training_epochs=0,optimizer_updates=0,head_fit=False,U_IQ_read=False,U_truth_read=False,query_read=False,V_cache_materialized=True,V_forward=False,early_middle='not available from frozen final checkpoint; real epochs1/17/33/50 to be observed only in subsequent preregistered training',stop_rule='Role/provenance/shape/nonfinite/state mutation failure stops, preserves outputs; no automatic retry or target feedback')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Frozen source layer order plan differs')
