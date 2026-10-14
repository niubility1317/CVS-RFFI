"""Fixed source-first sequence-feature matrix; original default classifier only."""
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-localdistribution-pathsignature-oldgate-e50-clean-manysig-m24-r01'
RELEASE='cvs_feature_sequence50_20261013_r01';OWNER='codex/root/sequence-features-20261013'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BASE_REF','MEAN','MEAN_VAR','NONLINEAR_MEAN','RFF','COV','SIG','SIG_TIME')
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);PRIMARY=('RFF','SIG_TIME')
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02,selection='all firstseed rows must satisfy fixed source health/noninferiority; no learned probe, candidate ranking or target feedback')
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed))
def config(row):
 if row not in rows():raise ValueError('Unregistered sequence-feature row')
 ref=baseline_config(row['model_seed']);a=row['arm']
 return dict(ref,schema='phase1_sequence_features_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=ref if a=='BASE_REF' else None,baseline_reuse=a=='BASE_REF',feature_extractor=dict(arm=a,layer='t2',local_dimension=6,local_projection='learned 1x1 then tanh; shared across controls',windows=4,random_directions=14,random_direction_seed=2026101307,random_direction_scales=[.5,1.,2.],descriptor_width=28,output_embedding=160,parameter_matched=a!='BASE_REF',intrinsic_descriptor_rank_matched=False,classifier_changed=False,gain_initial=.05,gain_bound=.2,time_coordinate=a=='SIG_TIME',per_packet=True),classifier='unchanged native ResidualPhysicalHead/6x160 cosine/scale30; jointly trained with backbone; no separate training or replacement',source_falsification=dict(SOURCE_GATE),test_views=list(VIEWS),readouts=list(READOUTS),separate_head_training=False,goal=dict(co_primary=list(PRIMARY),clean_three_seed_mean=.84,readout='NATIVE_HEAD',feedback_to_training=False),cost_claim='21 new scratch50 backbones and3 reused same-seed scratch50 references; no extra head optimization')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row) or c['separate_head_training'] or c['readouts']!=['NATIVE_HEAD']:raise ValueError('Fixed default-head feature config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(baseline_config(c['model_seed']),device)
 a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.seed=c['model_seed']
 return a
