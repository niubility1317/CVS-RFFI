"""Complete original ordinary draw is retained;only final10 L return IQ becomes raw."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-LRaw-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_LRaw_E30_20261018_r01'
OWNER='codex/root/practical-LRaw';LOCAL='local_artifacts/F_LRaw_E30_20261018'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
REFERENCE_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=2220
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660

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
ARMS=('B00','B01')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only actual original own20/30 references')
 return deepcopy(_REFERENCE['configs'][row['arm']][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only ownoriginal20+10 fixed seed01 rows')
 c=reference_config(row,30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/row['row_id']/'stage20';sat=row['arm']=='B01'
 c.update(schema='FcrossRX_LRaw30_v1',run_id=RUN,**row,launch_owner=OWNER,method='last10_ordinary_L_raw_return',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict original ownFINAL20 student and ownEMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=sat,training_recipe='OriginalP43 .5RDordinary128+.5RDmixed128+.25RDraw128+.1ORD128CROSS_RX;complete ordinary L draw discarded,return pre-call raw clone in final10 only',source_selection='fixedFINAL30;source failure still completes all7 formal;no automatic extension',budget_authorization='two ownoriginal20+10,totalE30 each;4440newupdates;reuse originalNoAs30 andAs30 readouts',cost_claim='6L/768NN and1.25Lscale eachstep;1704960newLNN each;142080As B01/0 B00;full native ordinary computation retained,no claimed saving',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='Original P43 matched own20 initializers and originalNoAs30/As30 frozen readouts',benchmark_context='Prior PA-only bypass returned post-general-channel prefix;this fixed whole ordinary return is pre-call raw.Original As already uses raw input,so not a duplicated-As-damage repair.Whole-training RNG equivalence unverified;first20 ordinary retained')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(raw ordinary slot128)+.5RD(mixed128)+.25RD(raw128)+.1CROSS_RX(raw ordinary slot128)',pair_weights=[.5,.5,.25],second_pair='existing raw ordinary64+completeAs(raw64)' if sat else 'existing raw ordinary64+raw64',same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0,ordinary_computation_retained=True,ordinary_return='pre-call raw clone',U_recipe_unchanged=True)
 c['factorial'].update(absent_slot='NoAs selected raw64',As_input='unchanged complete practical(raw64) before ordinary')
 c['crossview'].update(mode='ORD128',ordinary_anchors=128,additional_anchors=0,additional_view='none',actual_anchor_input='pre-call raw128 returned in ordinary slot',schedule='same original cumulativeE21..30/full0.1')
 c['lraw']=dict(scope='unique native ordinary L factory,visible L128 labels0..5,labels/no_paFalse signature',rule='clone original input;run original augmentor once on independent clone;discard generated IQ;return pre-call raw clone',first20_ordinary_retained=True,whole_trajectory_NoAug=False,full_native_aug_computation=True,aug_call_RNG_endstate_preserved='focused same-call check;fulltraining RNG not claimed',whole_training_RNG_matched=False,ordinary_calls_per_step=1,ordinary_packets_per_step=128,actual_ordinary_raw_MSE=0.,RD_same_generated_IQ=True,U_weak_strong_unchanged=True,U_augmentor_wrapped=False,teacher_unchanged=True,EMA_unchanged=True,new_losses=0,new_parameters=0,new_NN=0,formal_test_changed=False)
 c['source_promotion']=dict(fullV_newAs_minus_newNoAs_clean_min_pp=-.1,panel_midlow_newAs_minus_originalAs_min_pp=.5,source_failure_skips_formal=False,automatic_extension=False,on_fail='close fixedrecipe;no E50/switch/phase/seed rescue')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed two-row LRaw/data/parent/seed/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 row=next(r for r in rows() if r['row_id']==c['row_id']);original=b.config(next(r for r in b.rows() if r['row_id']==c['row_id']),30)
 if original!=reference_config(row,30):raise ValueError('Pinned actual P43 recipe differs from saved reference')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled:raise ValueError('Original tail L/RDrop/U args differ')
 return a
