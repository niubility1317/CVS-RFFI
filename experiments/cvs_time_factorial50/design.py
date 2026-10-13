"""Coverage x branch nonlinearity, fixed source-first Phase1 matrix."""
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-time-coverage-nonlinearity-oldgate-e50-clean-manysig-m15-r01'
RELEASE='cvs_time_factorial50_20261014_r01';OWNER='codex/root/time-factorial-20261014'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BASE_REF','COMPACT_LINEAR','LONG_LINEAR','COMPACT_SILU','LONG_SILU')
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);PRIMARY=('LONG_SILU',)
TITLE='时间覆盖与分支内非线性：默认分类头完整结果'
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02)
PAIRS=[(a,'BASE_REF') for a in ARMS[1:]]+[('LONG_LINEAR','COMPACT_LINEAR'),('COMPACT_SILU','COMPACT_LINEAR'),('LONG_SILU','LONG_LINEAR'),('LONG_SILU','COMPACT_SILU')]
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed))
def config(row):
 if row not in rows():raise ValueError('Unregistered temporal factorial row')
 ref=baseline_config(row['model_seed']);a=row['arm'];long=a.startswith('LONG');nonlinear=a.endswith('SILU')
 return dict(ref,schema='phase1_time_factorial_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=ref if a=='BASE_REF' else None,baseline_reuse=a=='BASE_REF',feature_extractor=dict(time_layers=['t1','t2','t3'],PA_changed=False,frequency_changed=False,reference_changed=False,settings=[[3,1],[5,1],[5,2 if long else 1]],branch_activation='SiLU' if nonlinear else 'identity',postprocessing='unchanged GN/ReLU/pool/dropout',skip='unchanged learned1x1 before original GN',output_embedding=160,extra_parameters=0 if a=='BASE_REF' else 60672,classifier_changed=False,matched_new_arm_parameters=True,initial_functions_equal=False if nonlinear else 'compact/long dilation differs',initialization_seed_offset=48213,RNG_policy='CPU fork_rng during added module init; restore native CPU/global loader/augmentation stream; no new stochastic forward ops'),classifier='unchanged native ResidualPhysicalHead6x160 scale30; joint CE/old pseudo training only',source_falsification=dict(SOURCE_GATE),test_views=['clean'],readouts=['NATIVE_HEAD'],separate_head_training=False,goal=dict(primary='LONG_SILU',clean_three_seed_mean=.84,feedback_to_training=False),cost_claim='12 new independent scratch50 and3 historical same-seed fixed50 references; no inherited initialization or extra head fitting')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed temporal factorial config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(baseline_config(c['model_seed']),device)
 a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.seed=c['model_seed'];return a
def result_metadata():return dict(rows=15,new_backbones=12,existing_backbone_references=3,readouts=15,prediction_count=2520000,reused_native_decisions=504000,new_prediction_decisions=2016000,metric_records=270,backbone_updates_new=133200,historical_baseline_updates=33300,factorial=True)
