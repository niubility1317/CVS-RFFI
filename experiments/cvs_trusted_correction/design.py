from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-diagnostic-trusted-correction-source-manysig-m3-r02';RELEASE='cvs_trusted_correction_source_20261014_r02';OWNER='codex/root/trusted-correction-source-20261014';BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN='20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02'
def rows():return [dict(row_id=f'TRUST_PROXY-s{s}',model_seed=s) for s in SEEDS]
def baseline_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed))
def reference_source(seed):return Path(PROJECT)/'runs'/REFERENCE_RUN/f'PAIRED_ORDER-s{seed}'/'source'
def config(row):
 if row not in rows():raise ValueError('Unregistered trust diagnostic row')
 return dict(schema='trusted_correction_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']).as_posix(),checkpoint=(reference_source(row['model_seed'])/'final_ssdg.pth').as_posix(),source_role='V',source_count=27000,model_mode='ownEMA eval, student train/no_grad for actual strong Dropout proxy; state/buffers frozen',PAIR_sampling=True,order_seed=row['model_seed']+99301,strong_rng_seed=row['model_seed']+99307,weak_view_seed_offset=99317,min_teacher_confidence=.99,second_weak_noise_RMS=.003,correction_count_cap_ratio=.1,correction_CE_mass_cap_ratio=.25,normalization='original mean over selected accepted packets',lambda_U_unchanged=.16,label_phase_epochs=33,pseudo_phase_epochs=17,threshold_fit=False,source_criterion=dict(pooled_admitted_count_min=100,each_seed_admitted_count_min=20,pooled_admitted_precision_min=.99,each_seed_admitted_precision_min=.98),U_IQ_read=False,U_truth_read=False,target_access=False,head_fit_updates=0,optimizer_updates=0,claim_scope='Source fixed-rule proxy, not hidden-U precision. Shared weak views are correlated; evaluate trust and value separately. Gate passes before B/D training, without threshold search.')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen trust source diagnostic differs')
 return c
