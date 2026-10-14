from pathlib import Path
from experiments.cvs_evidence_relation import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-U-calendar-pair-oldgate-e50-clean-manysig-m9-r02'
RELEASE='cvs_U_schedule50_20261014_r02';OWNER='codex/root/U-calendar-20261014';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('TAIL','MID','SPREAD');PRIMARY=('MID',);VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
TITLE='等 U 曝光量的训练时序：默认头固定三种子'
CALENDARS=dict(TAIL=list(range(34,51)),MID=list(range(17,34)),SPREAD=list(range(17,50,2)))
PAIRS=[('MID','TAIL'),('SPREAD','TAIL'),('SPREAD','MID')]
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02)
PROBE_U_ORDINALS=(1,6,12,17)

def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return parent.baseline_config(seed)
def config(row):
 if row not in rows():raise ValueError('Unregistered U calendar row')
 base=baseline_config(row['model_seed']);calendar=CALENDARS[row['arm']]
 return dict(base,schema='source_U_calendar50_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=None,baseline_reuse=False,feature_extractor=dict(structural_changes=False,extra_parameters=0,output_embedding=160,classifier_changed=False,training_mechanism='same17Uepochs moved within original50epochcosine; no additional objective'),calendar=dict(unlabeled_epochs=calendar,label_only_epoch_count=33,U_epoch_count=17,backbone_updates=11100,L_packet_observations=1420800,U_packet_observations=963900,pseudo_and_entropy_weights_unchanged=True,old_PAIR_gate_unchanged=True,ownEMA_all11100updates=True,U_gradient_scope='Original .16 selected-hard-pseudo CE plus .01 allU entropy; fullbase is actual total loss minus these two U terms.',optimizer_probe_U_ordinals=list(PROBE_U_ORDINALS),optimizer_probe_global_steps=[(calendar[i-1]-1)*222+1 for i in PROBE_U_ORDINALS],source_V_proxy_at_U_ordinals=list(PROBE_U_ORDINALS),proxy='CurrentownEMA eval weak/currentstudent identity-only trainDropout strong; completeoriginal27000V,labels downstream counts only; no fitting',source_V_proxy_forward_packets=216000,calendar_basis='Latestuser explicitly proposes17..33 mid; originalfinalsource curves establish source task learns in labelstage. Cutpointsfixed before thisrun target. No finalcheckpoint teacher used early.'),source_neighbor_sampling=dict(mode='PAIRS',original_gates=True),source_falsification=dict(SOURCE_GATE),test_views=['clean'],readouts=['NATIVE_HEAD'],separate_head_training=False,goal=dict(primary='MID',clean_three_seed_mean=.84,feedback_to_training=False),cost_claim='9new scratch50/99900updates; first3seed01 sourcehealth then6pendingrows, equal50L+17Uepochs;216000extraVproxyforward/model and4two-shadow optimizer probes,0extraactual optimizersteps',budget_authorization='User currentreport prioritizes equalUbudget timing experiment, same50epoch/backboneupdate/Lexposure/Uexposure/defaulthead/oldgate; Agentchoosefixedmid17..33 andspread17..49odd, no new relationshiploss orSinc.')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed U calendar differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.old.make_args(baseline_config(c['model_seed']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.source_unlabeled_epochs=list(c['calendar']['unlabeled_epochs']);return a
def result_metadata():return dict(rows=9,new_backbones=9,existing_backbone_references=0,readouts=9,prediction_count=1512000,reused_native_decisions=0,new_prediction_decisions=1512000,metric_records=162,backbone_updates_new=99900,historical_baseline_updates=0)
