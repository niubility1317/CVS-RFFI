"""Two fixed ownAs20+10 tails: remove As RDrop KL, then add same-IQ CSIL KD."""
from copy import deepcopy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-sameIQ-CSIL-e30-s2026092701-m2-r02'
RELEASE='cvs_FcrossRX_sameIQ_CSIL_E30_20261019_r02'
OWNER='codex/root/sameIQ-CSIL';LOCAL='local_artifacts/F_sameIQ_distill_E30_20261019'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=2220
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
def resolved(v):
 if isinstance(v,dict):return {k:resolved(x) for k,x in v.items()}
 if isinstance(v,list):return [resolved(x) for x in v]
 return v.replace('PROJECT',PROJECT) if isinstance(v,str) else v
_REFERENCE=resolved(read(Path(__file__).with_name('reference_config.json')))
globals().update(_REFERENCE['constants'])
SOURCE=Path(SOURCE);FORMAL_VIEWS=Path(FORMAL_VIEWS);TRUTH=Path(TRUTH);PANEL=Path(PANEL);SOURCE_CACHE=Path(SOURCE_CACHE);BASE=Path(PROJECT)/'runs'/RUN
TRAIN_SCENES=tuple(_REFERENCE['configs']['B01']['30']['channel']['train_scenes'])
CHANNEL_SEEDS=dict(_REFERENCE['configs']['B01']['30']['channel']['seeds'])
TEACHER=Path(PROJECT)/'paper_reproduction/runs/20260930-phase1-native-baselines-practical-manysig-m5-r01/csil-s2026092701'
TEACHER_QUAL=Path(PROJECT)/'runs/20261019-diagnostic-FcrossRX-sameIQ-experts-s2026092701-m2-r02/csil/comparison.json'
def rows():return [dict(arm='B01',row_id=a+'-s'+str(SEED),model_seed=SEED,distill_mode=m) for a,m in [('C01','CE'),('D01','CSIL')]]
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only original B01 source reference')
 return deepcopy(_REFERENCE['configs']['B01'][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed ownAs20+10 two-arm matrix')
 c=reference_config(row,30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/'B01-s2026092701/stage20'
 c.update(schema='F_sameIQ_CSIL30_v1',run_id=RUN,**row,launch_owner=OWNER,method='same_fullAs_CSIL_class_KD',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict original As20 student+ownEMA; reset AdamW/cos10/Ucache',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=True,training_recipe='original6L768/CE01/ordinary+rawKL/CROSS_RX/oldU; remove only selectedAs KL; add KD only D01',source_selection='fixedFINAL30;all7 formal even sourcefailure',budget_authorization='2 independent sameAs20+10 tails,totalE30;4440 new updates',cost_claim='768 studentL+64 frozenCSIL/step; native U student/EMA unchanged; teacher shadow computed in control',baseline_reference='original P43 As30/NoAs30 fixed results reused')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='OriginalL - .25AsKL64 + D01*.25*6*mean(teacherCorrect*KL(q5||p5))',pair_weights=[.5,.5,.25],same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['crossview'].update(mode='ORD128',ordinary_anchors=128,additional_anchors=0,additional_view='none',schedule='cumulativeE21..30/.1')
 c['sameIQ_teacher']=dict(mode=row['distill_mode'],method='csil',checkpoint=str(TEACHER/'last.pt'),seed=SEED,selection='fixed_last_epoch200',historical_updates=9800,qualified_source_ref=str(TEACHER_QUAL),teacher_logits_scale=5,student_deploy_scale=30,student_KD_temperature=6,KD_multiplier=6,KD_outer_weight=.25,teacher_target='detached native softmax',teacher_correct_mask='L TX argmax equals legal true label; mean over all64, no accepted-only renormalization',CE_temperature=1,CE_smoothing=.01,all64_CE=True,As_KL_removed_both=True,teacher_NN_per_step=64,control_teacher_shadow=True,extra_student_NN=0,extra_U_NN=0,teacher_not_registered_in_student=True,new_deploy_parameters=0)
 c['source_promotion']=dict(D_minus_C_midlow_min_pp=.5,D_minus_originalAs_low_min_pp=0,D_minus_NoAs_fullVclean_min_pp=-.1,automatic_extension=False,source_failure_skips_formal=False,on_fail='close fixed teacher/KD coefficient; no T/lambda/seed rescue')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed teacher/data/defaulthead/E30 matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 original=b.config(next(r for r in b.rows() if r['row_id']=='B01-s2026092701'),30)
 if original!=reference_config(next(r for r in rows() if r['row_id']==c['row_id']),30):raise ValueError('Pinned original P43 config differs')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled:raise ValueError('Original L/U/EMA args differ')
 return a
