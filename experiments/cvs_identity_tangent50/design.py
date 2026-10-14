from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
SEEDS=(2026092701,)
RUN='20261014-phase1-identity-tangent-oldgate-e50-clean-manysig-s01-m4-r01'
RELEASE='cvs_identity_tangent50_20261014_r01';OWNER='codex/root/identity-tangent-20261014';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BASE_REF','ZERO','RAW_MATCHED','IDSAFE');VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);PRIMARY=('IDSAFE',)
TITLE='身份保护接收机切向平坦化：固定首种子'
PAIRS=[('ZERO','BASE_REF'),('RAW_MATCHED','ZERO'),('IDSAFE','ZERO'),('IDSAFE','RAW_MATCHED')]
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed))
def config(row):
 if row not in rows():raise ValueError('Unregistered identity tangent row')
 ref=baseline_config(row['model_seed']);arm=row['arm']
 return dict(ref,schema='identity_tangent50_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=ref if arm=='BASE_REF' else None,baseline_reuse=arm=='BASE_REF',feature_extractor=dict(time_layers=['t1','t2'],structural_changes=False,extra_parameters=0,output_embedding=160,classifier_changed=False,training_mechanism='identity-protected receiver tangent flatness'),tangent=dict(mode=arm if arm!='BASE_REF' else None,interval=4,warmup_updates=1110,ramp_updates=1110,layers=['t1','t2'],layer_selection='alternate per active auxiliary call',source_role='L_s',batch_size=60,samples_per_TX_RX_day=2,day_selection='cycle actual source days1/2/3',epsilon=.1,identity_subspace='rank<=5 class-mean statistics span averaged over all5RX; detached SVD relative1e-6',direction='sameTX/day crossRX cell-mean shift; cyclic RX1/3/4/6/8',energy_control='RAW direction matches projected direction L2 per packet, shared max-coordinate bound1',statistic='channel mu/sourceRMS(sigma), log sigma; source RMS only current L batch',regularizer='squared symmetric derivative of L2normalized deployed joint160, no classW',gradient_cap=.2,coefficient='linear ramp * min(1000, .2*main LCE ftheta gradient norm/(rawR ftheta norm+1e-12)); zero R coefficient0; exclude classW',auxiliary_forward_calls=3,auxiliary_calls=2498,extra_L_forward_packets=449640,teacher_intervention=False,U_intervention=False,sampling_seed_offset=98413,query_access=False),source_falsification=dict(SOURCE_GATE),test_views=['clean'],readouts=['NATIVE_HEAD'],separate_head_training=False,goal=dict(primary='IDSAFE',clean_three_seed_mean=.84,first_seed_is_not_three_seed=True,feedback_to_training=False),cost_claim='3 new scratch50/11100; original oldgate same-seed fixed50 reference only;150 new modelepochs,33300updates; no headfit')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed identity tangent configuration differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(baseline_config(c['model_seed']),device);a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];return a
def result_metadata():return dict(rows=4,new_backbones=3,existing_backbone_references=1,readouts=4,prediction_count=672000,reused_native_decisions=168000,new_prediction_decisions=504000,metric_records=72,backbone_updates_new=33300,historical_baseline_updates=11100)
