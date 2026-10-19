"""Fixed own20+10: supplemental raw or complete HQ label-only CE."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-signalBridge-e30-s2026092701-m3-r01'
RELEASE='cvs_FcrossRX_signalBridge_E30_20261019_r01'
OWNER='codex/root/practical-signalBridge';LOCAL='local_artifacts/F_signalBridge_E30_20261019'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
REFERENCE_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=2220
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660
EXTRA_WEIGHT=.25;HQ_REFERENCE_SNR_DB=36.

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

def rows():return [dict(arm=a,row_id=n+'-s'+str(SEED),model_seed=SEED,extra_view=v) for n,a,v in [('N01','B00','RAW'),('C01','B01','RAW'),('S01','B01','HQ')]]
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only original own20/30 source references')
 return deepcopy(_REFERENCE['configs'][row['arm']][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed originalown20+10 three arms')
 c=reference_config(row,30);parentrow=row['arm']+'-s2026092701';parent=Path(PROJECT)/'runs'/REFERENCE_RUN/parentrow/'stage20'
 c.update(schema='FcrossRX_signalBridge30_v1',run_id=RUN,**row,launch_owner=OWNER,method='extra_completeHQ_labelCE_only',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),parent_row_id=parentrow,checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict original ownFINAL20 student andownEMA;resetoptimizer',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=row['arm']=='B01',training_recipe='OriginalP43 .5RDordinary+.5RDmixed+.25RDraw+.1ORD128CROSS_RX andoldU;add .25 mean of2CE64 onraworHQ,noextraKL/teacher/align',source_selection='fixedFINAL30;all7formaleven sourcefailure;futureE50sourceonly',budget_authorization='Three ownoriginal20+10 tails,totalE30 each;6660new updates;extra128NNperstep matched all3',cost_claim='8Lcalls=896NN/step;1989120LNN;oldUstudent/teacher567000each;0extraVJP;HQ shadow matched C/S physics',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='OriginalP43 NoAs30/As30 frozenreadouts reused;newN01/C01 matched896LNN and1.5Lscale')
 c['training_mechanism'].update(L_forwards_per_sample=8,expected_L_forward_packets=UPDATES*896,original_L_identity_forwards=UPDATES*768,extra_L_identity_forwards=UPDATES*128,objective='Original1.25L plus .25*mean(CEextra64pass0,CEextra64pass1)',pair_weights=[.5,.5,.25],extra_CE_weight=.25,same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['crossview'].update(mode='ORD128',ordinary_anchors=128,additional_anchors=0,additional_view='none',schedule='same cumulativeE21..30/.1')
 c['signal_bridge']=dict(extra_view=row['extra_view'],extra_CE_weight=.25,extra_packets=64,extra_passes=2,extra_NN_per_step=128,total_L_nominal_scale=1.5,original_clean_raw_supervision_absolute_weights_unchanged=True,original_fullAs_unchanged=True,extra_label_smoothing=.01,extra_KL=0,extra_teacher_NN=0,new_cross_view_consistency=0,HQ_reference_snr_db=36.,original_reference_snr_db=30.,same_exogenous_propagation=True,same_hardware_values=True,same_standard_noise_rng=True,native_quality_compensation_AGC_recomputed=True,HQ_shadow_in_C01=True,new_physical_snapshots_per_As_step=64,all_As_supervised=True,confidence_gate=False,formal_IQ_changed=False,new_deployment_parameters=0)
 c['source_promotion']=dict(S_minus_C_midlow_min_pp=.5,S_minus_N_fullVclean_min_pp=-.1,S_minus_originalAs_midlow_min_pp=0.,source_failure_skips_formal=False,automatic_extension=False,on_fail='close fixedHQ+6/extra.25;no noisegap/weight/seed rescue')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed data/parent/HQ/CEcost/E30 differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 original=b.config(next(r for r in b.rows() if r['row_id']==c['parent_row_id']),30)
 if original!=reference_config(next(r for r in rows() if r['row_id']==c['row_id']),30):raise ValueError('Pinned originalP43config differs')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled:raise ValueError('Original tail L/U/teacherargs differ')
 return a
