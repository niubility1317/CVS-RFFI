"""Fixed matched teacher-only U qualification and quarter complete practical U slots."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m3-r02'
RELEASE='cvs_FcrossRX_UPractical_E30_20261019_r02'
OWNER='codex/root/practical-UPractical';LOCAL='local_artifacts/F_UPractical_E30_20261019_r02'
FAILED_R01_COST=dict(status='FAILED_PREFLIGHT',run_id='20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m2-r01',code_commit='17b5939f440e370dacb6e92b6b523940d726115b',NN=770,optimizer_updates=0,real_IQ=0,propagation_simulations=0,worker_count=0,preserved=True,error='missing module trusted hook;actual model featurehook used inR02',real_parent_forward_packets=2,synthetic_L_forward_packets=768,U_forward_packets=0)
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
REFERENCE_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=2220
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660
ALLOCATION_SEED=20261016421

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf8')
_runtime=ROOT/'runtime.json';PROJECT=read(_runtime)['project'] if _runtime.is_file() else 'PROJECT'
def resolved(v):
 if isinstance(v,dict):return {k:resolved(x) for k,x in v.items()}
 if isinstance(v,list):return [resolved(x) for x in v]
 return v.replace('PROJECT',PROJECT) if isinstance(v,str) else v
_REFERENCE=resolved(read(Path(__file__).with_name('reference_config.json')))
globals().update(_REFERENCE['constants'])
SOURCE=Path(SOURCE);FORMAL_VIEWS=Path(FORMAL_VIEWS);TRUTH=Path(TRUTH);PANEL=Path(PANEL);SOURCE_CACHE=Path(SOURCE_CACHE);BASE=Path(PROJECT)/'runs'/RUN
TRAIN_SCENES=tuple(_REFERENCE['configs']['B01']['30']['channel']['train_scenes'])
CHANNEL_SEEDS=dict(_REFERENCE['configs']['B01']['30']['channel']['seeds'])
ARMS=('N01','C01','S01');REFERENCE_ROW='B01-s'+str(SEED)
def reference_row(row):return ('B00' if row['arm']=='N01' else 'B01')+'-s'+str(SEED)
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only actual original sameAs20/30 reference')
 return deepcopy(_REFERENCE['configs']['B00' if row['arm']=='N01' else 'B01'][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only registered sameAs20+10 fixed seed01')
 c=reference_config(row,30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/reference_row(row)/'stage20';sat=row['arm']=='S01';as_l=row['arm']!='N01'
 c.update(schema='FcrossRX_UPractical30_v1',run_id=RUN,**row,launch_owner=OWNER,method='teacher_only_qualification_plus_U_quarter_practical' if sat else 'matched_teacher_only_original_Gaussian_U',output_root=str(BASE/row['row_id']/'source'),reference_row_id=reference_row(row),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict corresponding original ownFINAL20 student and EMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=as_l,training_recipe='OriginalP43 L6/RD/ORD128 unchanged;both rows teacher-only qualification;S replaces quarter native U strong slots with full practical(raw U)',source_selection='fixedFINAL30;source failure still all7;no automatic extension',budget_authorization='three own20+10,totalE30 each;NownNoAs20;C/SsameownAs20;2220newupdates each;actual_m3 fixed matrix;fresh r02 after preserved r01 technical failure',cost_claim='unchanged6L768NN/1.25L andoneUstudent+teacher perphysicalU;S141750 extra completepractical simulations;0additionalNN;CPU routing/logging separately',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='NmatchedNoAs teacher-only control ownB00;C/S sharedownB01As20+EMA;original30 oldreadouts',benchmark_context='C/S common teacher-only qualification differs originalAs30 strong-agreement;S-C is practical U incremental effect,not originalrecipe-only comparison')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.5RD(mixed128)+.25RD(raw128)+.1CROSS_RX(ordinary128);originalU .16maskedCE+.01allUentropy',pair_weights=[.5,.5,.25],same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['crossview'].update(mode='ORD128',ordinary_anchors=128,additional_anchors=0,additional_view='none',schedule='same original cumulativeE21..30/full0.1')
 c['u_practical']=dict(mode='QUARTER_PRACTICAL' if sat else 'GAUSSIAN_CONTROL',slot_fraction=.25,actual_As_fraction=.25 if sat else 0.,U_batch_size=256,last_U_batch=124,U_batches=2220,U_packets=567000,U_As_packets=141750 if sat else 0,selected_slots=141750,slot_namespace='FcrossRX/UPractical/slots/v1',channel_namespace='FcrossRX/train/e{epoch}/step{step}/U_S',augmentation_seed=ALLOCATION_SEED,receiver_seed=CHANNEL_SEEDS['receiver'],selection='private uniform permutation B//4 positions withoutreplacement;no labels/confidence/quality',raw_input_to_practical=True,ordinary_before_practical=False,original_strong_draw_all_packets=True,Gaussian_std=.015,teacher_input='original raw clean U',teacher_only_eligibility=True,pseudo_strong_agreement=False,pseudo_domain_gate=False,pseudo_temporal_gate=True,pseudo_temporal_mode='batch_neighbor',teacher_quantile_unchanged=True,batch_neighbor_unchanged=True,lambda_unlabeled=.16,lambda_unlabeled_entropy=.01,CE_normalization='mean over teacher-qualified slots acrossallU;differentiablezero ifnone',entropy_normalization='mean over all U slots includingunqualified/practical',teacher_qualification_before_student_view=True,teacher_trajectory_equality_claim=False,U_truth_used=False,extra_NN=0,new_losses=0,L_changed=False,formal_test_changed=False)
 c['source_promotion']=dict(fullV_S_minus_N_clean_min_pp=-.1,panel_midlow_S_minus_C_min_pp=.5,source_failure_skips_formal=False,automatic_extension=False,on_fail='close fixedquarterUpractical;noE50/proportion/seed rescue')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed UPractical/data/parent/seed/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 row=next(r for r in rows() if r['row_id']==c['row_id']);original=b.config(next(r for r in b.rows() if r['row_id']==reference_row(row)),30)
 if original!=reference_config(row,30):raise ValueError('Pinned actual P43 recipe differs from saved reference')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';a.pseudo_strong_agreement=False
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled or a.pseudo_strong_agreement or a.pseudo_domain_gate or not a.pseudo_temporal_gate or a.pseudo_temporal_mode!='batch_neighbor' or a.strong_noise_std!=.015 or a.lambda_u!=.16 or a.lambda_ent!=.01:raise ValueError('Original L/Uteacher flags differ beyond common teacher-only eligibility')
 return a
