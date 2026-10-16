"""One matched scratchE20 source falsification, after fixed IQmix identity guard."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-crossRX-sameTX-IQmix-sourceRX1-e20-s2026092701-m3-r01'
RELEASE='cvs_F_sameTX_IQmix_sourceRX1_e20_20261016_r01'
OWNER='codex/root/F-sameTX-IQmix-E20-20261016'
LOCAL='local_artifacts/F_sameTX_IQmix_E20_20261016';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01';BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SMOKE_SOURCE=Path(PROJECT)/'runs'/old.RUN/'B00-RX1-s2026092701/source'
GUARD='20261016-diagnostic-F-crossRX-sameTX-IQmix-fitL-manysig-m2-r01'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('B00','C_same','B10');UPDATES=4440;SCENES=old.SCENES
base_model_config=old.base_model_config;source_validation=old.source_validation;macro_f1=old.macro_f1

def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def original_config():return old.config(old.rows()[0])
def config(row):
 if row not in rows():raise ValueError('Fixed three source IQmix rows only')
 c=deepcopy(original_config());arm=row['arm'];mode={'B00':'raw','C_same':'sameRX','B10':'crossRX'}[arm]
 c.update(schema='F_CROSS_RX_sameTX_IQmix_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_sameTX_IQmix',baseline_reference='Matched raw-slot B00,original whole base retained;C_same controls averaging and three donor variants',updates=4440)
 c['training_mechanism'].update(Ag=arm=='B10',As=False,base_raw_total_weight=.75,raw_total_weight=1. if arm=='B00' else .75,ordinary_weight=.25,raw_weight=.5,s_slot_weight=.25,s_slot_name='Common third pair raw in all rows',extra_pair_weight=.25,extra_pair_input=mode,L_forwards_per_packet=8,expected_L_packets=4546560,no_new_U_augmentation=True)
 c['iqmix']=dict(mode=mode,coefficient=.1,base_pair_weight_sum=1.,pair_weight_sum=1.25,extra_pair_weight=.25,extra_loss='Original smoothedCE.01 mean plus within-IQ RDrop1KL;no new loss',bank='Actual filtered native train_loader.dataset visibleL5040 only,materialized once perrow,no G0 cache/weights',pairing='SameTX/day;sort existingopaqueIDs in each70cell;cross next-rank+1 in each of3otherRX;sameRX ranks+1/+2/+3;visitmod3 cycles all3 variants peranchor',control_refinement='sameRX has3distinctdonors to match3crossRX variants;G0sameRX checked+1 only,cross candidate unchanged',transform='Exactly G0 mix .1:donorRMS-to-anchor,relative lag20 frequency and80:160 phase alignment,.9anchor+.1donor,restoreanchorRMS;all256,no crop/shift/channel/PA',normalization='Perphysical mean within each pair;fullraw.75/ordinary.25 and SupCon unchanged;no divide by4;pairweight1.25 in allrows',same_IQ_within_each_pair=True,bank_persistence='Training-module global only;never model/teacher/buffer/checkpoint/export state',U='Original13L+7U/ownEMA/hiddentruth unchanged;no new U or teacher views',donor_read_packets=0 if arm=='B00' else 568320,generated_packets=0 if arm=='B00' else 568320,extra_L_NN_packets=1136640,source_identity_guard=GUARD)
 c['channel']['input']='No As training;only fixed heldsource final20 sixpractical readouts';c['channel']['composition']=False
 c['training_recipe']='ScratchE20/F+CROSS_RX/defaultcos/NoExplicitCFO/originalPA+shift/U;fullbase raw.75+ordinary.25+SupCon plus .25 raw/sameRX/crossRX IQmix;8Lforwards'
 c['source_selection']=dict(primary='B10',references=['B00','C_same'],fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5,final_base_L_objective_max=.5,on_fail='Close fixed sameTX IQmix recipe;no coefficient/alignment/window/donor/RX/seed or budget rescue',on_pass='Only supports independent source-seed confirmation;not formal target or fulljoint proof')
 c['budget_authorization']='Active Ag/As goal,latestF+CROSS_RX/E20;three matched scratchE20/13320updates,60modelepochs;no parent initialization/continuation/target'
 c['cost_claim']='Each4440updates/4546560LNN incl1136640extra;all3nativeordinary568320;mixrow568320 IQmix/donor gathers;bank5040directfitIQreads allrows;originalU/EMA/V and8820held readouts separately actual-counted;dispatcher realCPsmoke2NN;no extra teacher/headfit'
 return c

def validate(c):
 row=next((v for v in rows() if v['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source IQmix roles/budget/operator/weights/controls differ')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(original_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All three source IQmix rows required')
 s=results['B10'];conditions={};comparisons={}
 for ref in ('B00','C_same'):
  b=results[ref];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy'])
  part=dict(clean=delta['clean']>=1.,clean_macroF1=f1>=0.,each_practical=all(delta[v]>=-.5 for v in SCENES),seen_V=dv>=-.5)
  comparisons[ref]=dict(conditions=part,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seen_V_gain_pp=dv);conditions.update({ref+'_'+k:v for k,v in part.items()})
 conditions['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['IQmix_steps']==4440 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and r['IQmix_bank_unique_L']==5040 and r['IQmix_generated_packets']==(0 if arm=='B00' else 568320) and not r['As_training_observations'] and not r['teacher_IQmix_calls'] and not r['head_fit_updates'] and not r['target_access'] for arm,r in results.items())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,target_access=False,Ag_efficacy_proved=False,goal_complete=False)
