from pathlib import Path
from experiments.cvs_evidence_relation import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,read,write,FULL_FP32_POLICY=(getattr(parent,k) for k in ('PROJECT','SOURCE','read','write','FULL_FP32_POLICY'))
RUN='20261014-diagnostic-relation-signal-source-manysig-s01-m4-r01'
RELEASE='cvs_relation_signal_source_20261014_r01'
OWNER='codex/root/relation-signal-source-20261014'
BASE=Path(PROJECT)/'runs'/RUN

def rows():
 return [dict(row_id='SIGNAL-'+r['row_id'],parent_row=r) for r in parent.rows() if r['model_seed']==parent.SEEDS[0]]

def config(row):
 if row not in rows():raise ValueError('Unregistered frozen source signal row')
 c=parent.config(row['parent_row'])
 return dict(schema='relation_signal_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']).as_posix(),parent_config=c,checkpoint=(Path(c['output_root'])/'final_ssdg.pth').as_posix(),source_role='L_s',anchors=180,disjoint_probe_packets=720,batch_size=120,views=['CLEAN_EVAL','STRONG_NATIVE'],anchor_probe_seed=parent.SEEDS[0]+93217,probe_order_seed=parent.SEEDS[0]+93613,strong_rng_seed=parent.SEEDS[0]+94119,label_smoothing=.01,relation_modes=['ALL_REL','CROSS_RESP'],diagnostic_labels='Legal L TX only in exact-default-head CE direction, same-TX mass and true-class margin measurements; never supplied to relation/response objective.',source_forward_packets=2340,head_fit_updates=0,optimizer_updates=0,EMA_updates=0,U_IQ_read=False,U_truth_read=False,target_access=False,threshold_fit=False,parameter_selection='All trainable id_backbone parameters excluding default class W and two fusion gains; actual same-packet CE versus R/E gradients',claim_scope='Frozen first-seed source L seen during training. Gradient dot products describe local steepest descent, not actual AdamW updates or unseen-RX performance. No method or weight selection from target results.')

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source signal config differs')
 return c
