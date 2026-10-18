"""Two ownAs20+10 rows: IID versus risk-preserving stratification of one source-L bank."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-qualityRisk-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_qualityRisk_E30_20261018_r03'
OWNER='codex/root/practical-qualityRisk';LOCAL='local_artifacts/F_qualityRisk_E30_20261018'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
REFERENCE_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
REFERENCE_ROW='B01-s2026092701'
SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=2220
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660
BANK_SEED=20261016421;BANK_SIZE=252000;BIN_EDGES=(3.,8.,15.)
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
SOURCE=Path(SOURCE);FORMAL_VIEWS=Path(FORMAL_VIEWS);TRUTH=Path(TRUTH);PANEL=Path(PANEL);SOURCE_CACHE=Path(SOURCE_CACHE);BASE=Path(PROJECT)/'runs'/RUN;BANK=BASE/'proposal_bank'
TRAIN_SCENES=tuple(_REFERENCE['configs']['B01']['30']['channel']['train_scenes']);CHANNEL_SEEDS=dict(_REFERENCE['configs']['B01']['30']['channel']['seeds'])
ARMS=('I01','S01')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only original B01 ownAs20/30 reference')
 return deepcopy(_REFERENCE['configs']['B01'][str(segment)])
def bank_contexts():return [dict(cumulative_epoch=e,replica=r,cumulative_step=(e-1)*222+1+r) for e in range(21,31) for r in range(4)]
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only two fixed ownoriginalAs20+10 seed01 rows')
 c=reference_config(row,30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/REFERENCE_ROW/'stage20';mode='IID' if row['arm']=='I01' else 'STRATIFIED'
 c.update(schema='FcrossRX_qualityRisk30_v1',run_id=RUN,**row,launch_owner=OWNER,reference_row_id=REFERENCE_ROW,method='finite_bank_'+mode,output_root=(BASE/row['row_id']/'source').as_posix(),parent_source=parent.as_posix(),checkpoint_sources=[(parent/'final_ssdg.pth').as_posix()],model_initialization='same strict original B01 AsFINAL20 student and ownEMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=True,training_recipe='OriginalP43 .5RDordinary128+.5RDmixed128+.25RDraw128+.1ORD128;selected64 from shared complete-practical sourceL bank;perpacket CE and KL both importance weighted',source_selection='fixedFINAL30;scientific failure completes all7;no automatic M3/E50',budget_authorization='two independent ownAs20+10 rows,totalE30each;4440newupdates;reuse originalNoAs30/As30',cost_claim='6L768NN and1.25 nominal Lscale/step;1704960 LNN+142080cachedAs/row;252000 shared propagation simulations;sampling/cachecopy/weightedloss CPU wall separate',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='I01 andS01 same originalAs20;original liveAs30 separate reference;NoAs30 has its own NoAs20 trajectory',benchmark_context='Finite-bank expected-risk equivalence only IID versus STRATIFIED;not online-risk or commonU trajectory equivalence')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.5weightedRD(mixed128)+.25RD(raw128)+.1CROSS_RX(ordinary128)',pair_weights=[.5,.5,.25],second_pair='ordinary64 currentL + bankAs64 own true labels/domain/meta;original alternating slots',same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['factorial'].update(absent_slot='originalNoAs selectedraw64;readout reused',As_input='completepractical(native rawL) bank;not necessarily currentL physicalID')
 c['crossview'].update(mode='ORD128',ordinary_anchors=128,additional_anchors=0,additional_view='none',schedule='same original cumulativeE21..30/full0.1')
 c['quality_risk']=dict(mode=mode,bank=BANK.as_posix(),bank_size=BANK_SIZE,source_L=6300,replicas_per_epoch=4,cumulative_epoch_contexts=list(range(21,31)),contexts=bank_contexts(),each_L_bank_views=40,bin_field='signal_to_added_noise_db',bin_edges=list(BIN_EDGES),bin_labels=['<3','[3,8)','[8,15)','>=15'],nominal_bank_mass='N_bin/N_bank;allL equal40;original perPID IIDscene;no rejection',sample_with_replacement=True,bank_packets_per_step=64,per_bin_packets=None if mode=='IID' else 16,bank_weight='1' if mode=='IID' else '4*N_bin/N_bank',ordinary_weight=1.,mixed_denominator=128,mixed_outer_weight=.5,As_expected_outer_weight=.25,all_loss_scale=1.25,CE_label_smoothing=.01,KL_same_weights_as_CE=True,sampling_seed=BANK_SEED,sampling_namespace='FcrossRX/qualityRisk/bank_sampling/v1',private_RNG=True,global_RNG_used=False,original_alternating_mask=True,mixed_labels_replaced=True,mixed_domain_metadata_replaced=True,ORD128_currentL_only=True,online_bank_risk_equivalence_claim=False,common_U_trajectory_equality_claim=False,extra_NN=0,extra_CE_operator_calls=0,CE_operator_widths=[128,128],weighted_reduction_new=True,formal_test_changed=False)
 c['source_promotion']=dict(hard_equal_midlow_S_minus_I_min_pp=.5,easy_equal_midlow_S_minus_I_min_pp=-.2,fullV_S_minus_originalAs_min_pp=-.1,hard='signal_to_added_noise_db<8',easy='signal_to_added_noise_db>=8',equal_scene_mean=True,source_failure_skips_formal=False,automatic_extension=False,M3_implemented=False)
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed two-row qualityRisk/data/parent/seed/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 row=next(r for r in rows() if r['row_id']==c['row_id']);original=b.config(next(r for r in b.rows() if r['row_id']==REFERENCE_ROW),30)
 if original!=reference_config(row,30):raise ValueError('Pinned actual P43 B01 recipe differs')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled or not a.a1_source_screen_only:raise ValueError('Original tail L/RDrop/U/source-only args differ')
 return a
