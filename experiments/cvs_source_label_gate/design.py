"""Read-only labeled-source proxy; no new backbone or fitted classifier."""
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,read,write,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','SEEDS','read','write','FULL_FP32_POLICY'))
RUN='20261014-diagnostic-source-labeled-gate-oldgate-manysig-m3-r02'
RELEASE='cvs_source_label_gate_20261014_r02';OWNER='codex/root/source-label-gate-20261014'
BASE=Path(PROJECT)/'runs'/RUN
SEED_OFFSET=79021
FORBIDDEN_TX=('tx','tx_i','true_tx_i','tx_label','y_tx')
def rows():return [dict(row_id=f'GATE_PROXY-s{s}',model_seed=s) for s in SEEDS]
def baseline_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed))
def config(row):
 if row not in rows():raise ValueError('Unregistered source proxy row')
 return dict(schema='phase1_source_gate_proxy_v1',run_id=RUN,**row,launch_owner=OWNER,source_config=baseline_config(row['model_seed']),role='L_s',count=6300,batch_size=256,drop_last=False,shuffle='native u_unlabeled_shuffle setting',sampling_seed=row['model_seed']+SEED_OFFSET,models='own historical fixedfinal student and own EMA; exact same source contract',student_modes=['train_dropout_no_grad','eval_no_grad'],native_gate='unchanged confidence + batch_neighbor + weakEMA/strongstudent agreement',structural_bound='all-one predictions/confidence using identical blind metadata',fit_updates=0,EMA_updates=0,head_fit_updates=0,U_forward_count=0,target_access=False,output_root=(BASE/row['row_id']).as_posix(),interpretation='Seen labeled training source proxy, not measured U precision or unseen RX generalization; no rule change from result')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Source gate proxy config differs')
 return c
