"""Fixed five-condition feature experiment; all weights are frozen before target."""
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-restitution-competition-oldgate-e50-clean-manysig-m15-r01'
RELEASE='cvs_feature_restitution_20261013_r01'
OWNER='codex/root/restitution-competition-20261013'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BASE_REF','ORDINARY','RESTORE','COMP','RESTORE_COMP')
READOUTS=('NATIVE_HEAD','REFIT_COSINE');VIEWS=('clean',)
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
PROBE=dict(epochs=10,batch_size=256,lr=.001,weight_decay=.0001,scale=30.,seed_offset=43019,embedding=160,optimizer='AdamW',selection='final_only',fit_role='L_s',backbone_frozen=True)
SOURCE_GATE=dict(final_L_CE_max=.5,leave_RX_mean_drop_max=.01,leave_RX_worst_drop_max=.02,TX_V_drop_max=.01,selection='entire firstseed matrix must pass source health/noninferiority; no candidate ranking or target feedback; not an effectiveness claim')
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed))
def config(row):
 if row not in rows():raise ValueError('Unregistered restitution row')
 ref=baseline_config(row['model_seed']);a=row['arm']
 return dict(ref,schema='feature_restitution_competition_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=ref if a=='BASE_REF' else None,baseline_reuse=a=='BASE_REF',normalization='ordinary' if a=='ORDINARY' else 'restore' if a in ('RESTORE','RESTORE_COMP') else 'none',competition=a in ('COMP','RESTORE_COMP'),feature_extractor=dict(layer='t1',groupnorm_channels=72,groupnorm_groups=8,relative_statistics='group mean/RMS and log(group std/RMS)',alpha_initial=.05,alpha_bound=.2,mask_probability=.1,warmup_steps=1110,mask_groups=['time','frequency','PA','reference'],mask_role='L_s student only',deployed_embedding=160,classifier_changed=False),probe=dict(PROBE),source_falsification=dict(SOURCE_GATE),test_views=list(VIEWS),readouts=list(READOUTS),goal=dict(primary='RESTORE_COMP',clean_three_seed_mean=.84,feature_evidence='positive paired REFIT_COSINE gain over BASE_REF; native primary mean>=84%; no classifier architecture change',feedback_to_training=False),cost_claim='12 new scratch50 backbones plus3 existing scratch50 references; 10epoch fixed ordinary probes are separate head-only cost, never backbone updates')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed restitution config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(baseline_config(c['model_seed']),device)
 a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.seed=c['model_seed']
 return a
