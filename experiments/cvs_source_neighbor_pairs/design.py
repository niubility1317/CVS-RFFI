from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01'
RELEASE='cvs_source_neighbor_pairs_20261014_r01';OWNER='codex/root/source-neighbor-pairs-20261014'
BASE=Path(PROJECT)/'runs'/RUN
ROLES=('L_s','V')
def rows():return [dict(row_id='SOURCE_PAIR-s'+str(s),model_seed=s) for s in SEEDS]
def source_config(row):return old.config(dict(row_id='OLD_GATE_COS50-s'+str(row['model_seed']),arm='OLD_GATE_COS50',model_seed=row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered source gate row')
 return dict(schema='source_neighbor_pairs_v1',run_id=RUN,**row,source_config=source_config(row),roles=list(ROLES),samplers=['RANDOM','PAIRS'],pair_fields=['base_index','day_i','eq_i','rx_i','sig_i'],source_criterion=dict(each_V_full_coverage_min=.5,each_V_full_precision_min=.995,each_V_full_precision_drop_max=.0025),source_role_counts=dict(L_s=6300,V=27000),batch=256,shuffle_seeds=dict(L_s=2026101410,V=2026101420),strong_dropout_seeds=dict(L_s=2026101411,V=2026101421),observer_labels='L/V only; labels absent from gate calls',teacher='actual own EMA final50',student='actual own final50 identity train-mode dropout for strong view, frozen parameters; no BN in identity path',head='existing default jointly trained cosine6x160scale30; no new fitting',strong_noise_std=.015,source_only=True,target_access=False,U_IQ_read=False,U_truth_read=False,optimizer_steps=0,model_state_updates=0,output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER)
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed L/V source diagnostic differs')
 return c
