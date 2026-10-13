"""Source-only probes on all nine fixed feature extractors, without target feedback."""
from pathlib import Path
from experiments.cvs_frequency_context50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,CLASSES,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','SEEDS','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-frozen-feature-rehead-source-manysig-m9-r01'
RELEASE='cvs_source_transfer_probe_20261013_r01'
OWNER='codex/root/source-transfer-probe-20261013'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE_RUN=parent.RUN
ARMS=parent.ARMS
SUPPORT_SEEDS=(2026101301,2026101302,2026101303)
K_VALUES=(5,10)
HEAD_EPOCHS=20
SOURCE_RX=(1,3,4,6,8)
def rows():return parent.rows()
def config(row):
 if row not in rows():raise ValueError('Unregistered source probe')
 return dict(schema='source_transfer_probe_v1',run_id=RUN,**row,source_config=parent.config(row),source_run=SOURCE_RUN,output_root=(BASE/row['row_id']).as_posix(),head=dict(type='ordinary cosine W6x160',scale=30.,bias=False,epsilon=1e-4,optimizer='Adam',lr=.01,weight_decay=0.,epochs=HEAD_EPOCHS,batch_size=256,initialization='matched seeded Xavier, no old W'),feature_updates=0,readouts=['ORIGINAL_HEAD','FRESH_COSINE_ALL_L','COSINE_PROTO_ALL_L','FRESH_COSINE_CROSS_RX_K'],support_seeds=list(SUPPORT_SEEDS),K=list(K_VALUES),source_rx=list(SOURCE_RX),training_labels='L_s only',validation='V only; no fit or candidate selection',target_access=False,U_opened=False,claim='Head-fit RX holdout only; every RX/TX already seen by backbone; no new-TX or Phase2 claim')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source transfer probe differs')
 return c
