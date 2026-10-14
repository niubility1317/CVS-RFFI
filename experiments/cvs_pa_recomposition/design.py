from pathlib import Path
from experiments.cvs_evidence_relation import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01'
RELEASE='cvs_PA_recomposition50_20261014_r01';OWNER='codex/root/PA-recomposition-20261014';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('ZERO','SAME_RX','CROSS_RX');PRIMARY=('CROSS_RX',);VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
TITLE='同身份证据跨 RX 重组：PA 实际增量匹配验证'
PAIRS=[('CROSS_RX','ZERO'),('SAME_RX','ZERO'),('CROSS_RX','SAME_RX')]
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02,aux_gradient_ratio_min=1e-5,mean_common_cohort_coverage_min=.25)
PROBE_STEPS=(1112,3556,7328,10880)

def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return parent.baseline_config(seed)
def active(step):return 1110<step<=11100 and step%4==0
def config(row):
 if row not in rows():raise ValueError('Unregistered PA recomposition row')
 return dict(baseline_config(row['model_seed']),schema='source_PA_recomposition50_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=None,baseline_reuse=False,feature_extractor=dict(structural_changes=False,extra_parameters=0,output_embedding=160,classifier_changed=False,training_mechanism='L_s PA additive increment recomposition, actual joint160; detached currentdefaultW and auxgains only'),recomposition=dict(mode=row['arm'],alpha=.2,coefficient=.1,warmup_steps=1110,frequency=4,auxiliary_calls=2498,role='L_s',old_L_CE_unchanged=True,U_training_unchanged=True,teacher_training_unchanged=True,aux_head_and_gain_detach=True,internal_feature_gradients=True,cosine_eps=.0001,cosine_scale=30.,aux_label_smoothing=0.,sampling='ExistingL128batch only; eachTX/day balancedqualifiedRXcells>=2 distinctphysicalpackets/cell and>=2cells, privateCPUgenerator; SAMEwithinRXcycle/CROSSbetweenRXcycle; everycohortpacket1anchor+1donor inbothconditions',sampling_seed_formula='model_seed+91273+optimizer_step*701',no_extra_model_forward=True,extra_L_forward_packets=0,extra_U_forward_packets=0,extra_actual_optimizer_steps=0,gradient_and_update_probe_steps=list(PROBE_STEPS),bank=False,loss_basis='GroundtruthLidentity, not teachergeometry; do not treat differentpackets as samewaveform hardwarecounterfactual'),source_neighbor_sampling=dict(mode='PAIRS',original_gates=True),source_falsification=dict(SOURCE_GATE),test_views=['clean'],readouts=['NATIVE_HEAD'],separate_head_training=False,goal=dict(primary='CROSS_RX',clean_three_seed_mean=.84,feedback_to_training=False),cost_claim='9new scratch50/99900updates; first3seed01 sourcehealth then6pendingrows. ExistingLforward only,2498lowcost recompositioncalls/model,0extraIQforwards/0additionalparameters/0extraactualupdates; fourgradient/currentAdamW/updateprobes',budget_authorization='User c718649d report explicitlyrequests sameTX/day crossRXPAincrement CE with detacheddefaultW/gains and sameRX/originalcontrols. Fixedprototype alpha.2/coefficient.1 established beforeUcalendar target; new mechanism independent originaltail33+17 baseline, no timing/R/E/Sinc/temporal stacking.')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed PA mechanism differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.old.make_args(baseline_config(c['model_seed']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def result_metadata():return dict(rows=9,new_backbones=9,existing_backbone_references=0,readouts=9,prediction_count=1512000,reused_native_decisions=0,new_prediction_decisions=1512000,metric_records=162,backbone_updates_new=99900,historical_baseline_updates=0)
