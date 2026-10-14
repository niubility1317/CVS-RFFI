from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-evidence-relation-pair-oldgate-e50-clean-manysig-m12-r01';RELEASE='cvs_evidence_relation50_20261014_r01';OWNER='codex/root/evidence-relation-20261014';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('ZERO','ALL_REL','CROSS_REL','CROSS_RESP');PRIMARY=('CROSS_RESP',);VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
TITLE='跨 RX 样本关系与 CVS 增量证据响应：完整三种子'
PAIRS=[('ALL_REL','ZERO'),('CROSS_REL','ALL_REL'),('CROSS_RESP','CROSS_REL'),('CROSS_RESP','ZERO')]
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==seed))
def config(row):
 if row not in rows():raise ValueError('Unregistered evidence relation row')
 ref=baseline_config(row['model_seed'])
 return dict(ref,schema='evidence_relation50_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=None,baseline_reuse=False,feature_extractor=dict(structural_changes=False,extra_parameters=0,output_embedding=160,classifier_changed=False,training_mechanism='source sample relations and additive CVS evidence response'),relation=dict(mode=row['arm'],tau=.2,alpha=.5,lambda_relation=0. if row['arm']=='ZERO' else 1.,lambda_response=1. if row['arm']=='CROSS_RESP' else 0.,response_norm='SmoothL1 beta.02 sum anchor coordinates, mean RX/branches/packets',all_legal_U=True,U_TX_read=False,feature='actual id_feat_joint joint160; auxiliary reconstructs exact values with detached PA/ref gains',anchor_role='L_s only',anchor_pool_count=180,anchors_per_TX_RX_day=2,anchors_per_query=120,anchors_per_query_TX=20,anchor_day_counts_per_TX=[7,7,6],RX_groups=4 if row['arm'] in ('CROSS_REL','CROSS_RESP') else 5,teacher='ownEMA only, detached',refresh_updates=222,refresh_first_update=7327,refresh_calls=17,extra_L_forward_packets=3060,auxiliary_calls=3774,U_packet_observations=963900,weakstrong='original EMA clean/native strong noise.015 and Dropout',gradient_reweight=False,probability_thresholds_added=False,pseudo_gate_changed=False,early_U=False,global_entropy_changed=False,class_W_used=False,aux_gain_gradients=False,deployment_reference_bank=False),source_neighbor_sampling=dict(mode='PAIRS',original_gates=True),source_falsification=dict(SOURCE_GATE,nonzero_relation_fraction_min=.5,mean_aux_to_L_feature_gradient_min=1e-5),test_views=['clean'],readouts=['NATIVE_HEAD'],separate_head_training=False,goal=dict(primary='CROSS_RESP',clean_three_seed_mean=.84,feedback_to_training=False),cost_claim='4newfirstseed50, remaining8 only after frozen source criterion; complete12scratch50/133200updates plus3060Lanchorforward each; no headfit')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed evidence relation config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(baseline_config(c['model_seed']),device);a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];return a
def result_metadata():return dict(rows=12,new_backbones=12,existing_backbone_references=0,readouts=12,prediction_count=2016000,reused_native_decisions=0,new_prediction_decisions=2016000,metric_records=216,backbone_updates_new=133200,historical_baseline_updates=0)
def source_reference_metrics(seed):return read(Path(baseline_config(seed)['output_root'])/'completion.json')['final_source_metrics']
