"""Single temporal basis versus completed compact multibasis, parameter matched."""
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
from experiments.cvs_time_factorial50 import design as temporal
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-single-basis-residual-oldgate-e50-clean-manysig-m6-r01'
RELEASE='cvs_time_basis50_20261014_r01';OWNER='codex/root/time-basis-20261014'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BASE_REF','SINGLE_BASIS');VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);PRIMARY=('SINGLE_BASIS',)
TITLE='单时间滤波基残差与相近参数量控制：默认头结果'
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02)
PAIRS=[('SINGLE_BASIS','BASE_REF')]
HIDDEN=(90,127,148)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed))
def compact_config(seed):return temporal.config(next(r for r in temporal.rows() if r['arm']=='COMPACT_LINEAR' and r['model_seed']==seed))
def source_reference_metrics(seed):
 from experiments.cvs_time_factorial50.provenance import source_proof
 p=source_proof(compact_config(seed));return read(p/'completion.json')['final_source_metrics']
def config(row):
 if row not in rows():raise ValueError('Unregistered single-basis row')
 ref=baseline_config(row['model_seed']);a=row['arm']
 return dict(ref,schema='phase1_time_basis_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=ref if a=='BASE_REF' else None,baseline_reuse=a=='BASE_REF',feature_extractor=dict(time_layers=['t1','t2','t3'],kernel_size=[5,5,3] if a=='BASE_REF' else 5,bases_per_input=1,preGN_rank_bound=1 if a=='BASE_REF' else 2,pointwise='original single1x1' if a=='BASE_REF' else 'two linear1x1 factors without intervening activation; hidden90/127/148',skip='none' if a=='BASE_REF' else 'learned1x1 before original GN',postprocessing='unchanged GN/ReLU/pool/dropout',time_parameters=21000 if a=='BASE_REF' else 81744,matched_compact_time_parameters=81672,parameter_difference_to_compact=72,extra_parameters=0 if a=='BASE_REF' else 60744,PA_changed=False,frequency_changed=False,Sinc_changed=False,reference_changed=False,output_embedding=160,classifier_changed=False,initialization_seed_offset=48213,RNG_policy='CPU default_generator fork only; no additional CUDA/global loader/augmentation draw; same native forward stochastic shape'),source_control=compact_config(row['model_seed']),classifier='unchanged native ResidualPhysicalHead6x160 scale30; normal joint CE/old pseudo',source_falsification=dict(SOURCE_GATE,comparison='fixedfinal COMPACT_LINEAR same seed'),test_views=['clean'],readouts=['NATIVE_HEAD'],separate_head_training=False,goal=dict(primary='SINGLE_BASIS',clean_three_seed_mean=.84,feedback_to_training=False),cost_claim='3 new scratch50; original baseline and completed compact controls are historical comparison only; no weight inheritance or headfit')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed single-basis config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(baseline_config(c['model_seed']),device);a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.seed=c['model_seed'];return a
def result_metadata():return dict(rows=6,new_backbones=3,existing_backbone_references=3,readouts=6,prediction_count=1008000,reused_native_decisions=504000,new_prediction_decisions=504000,metric_records=108,backbone_updates_new=33300,historical_baseline_updates=33300,external_compact_references=3,external_compact_decisions=504000,external_compact_metric_records=54)
