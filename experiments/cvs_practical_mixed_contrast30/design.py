"""One fixed E20+10 continuation; SupCon input membership is the sole new factor."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-mixedContrast-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_mixedContrast_E30_20261018_r01'
OWNER='codex/root/practical-mixedContrast';LOCAL='local_artifacts/F_mixedContrast_E30_20261018'
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

ARMS=('B00','B01')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only actual original B00/B01 references')
 return deepcopy(_REFERENCE['configs'][row['arm']][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only ownoriginal20+10 fixed seed01 rows')
 c=reference_config(row,30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/row['row_id']/'stage20';sat=row['arm']=='B01'
 c.update(schema='FcrossRX_mixedContrast30_v1',run_id=RUN,**row,launch_owner=OWNER,method='mixed128_singlephysical_CROSS_RX',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict original ownFINAL20 student and EMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=sat,training_recipe='.5RDordinary128+.5RDmixed128+.25RDraw128+.1CROSS_RX(existingmixed128);one anchor per labeled physical input;oldU unchanged',source_selection='fixedFINAL30;future extension eligibility only;source failure still completes all7 formal',budget_authorization='two ownoriginal20+10,totalE30 each;4440newupdates;reuse original20 and originalE30 frozen readouts',cost_claim='6L/768NN and1.25Lscale eachstep;1704960LNN pernewrow;As142080or0;extraNN0',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='Original P43 NoAs/As20 initializers and E30 readouts;no weighted parent,unchanged original .25/.25 mixed supervision',benchmark_context='SAT192 source failure retained;new hypothesis changes view membership at fixed128 anchors/single physical weighting,not claim of novel satellite SupCon')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.5RD(mixed128)+.25RD(raw128)+.1CROSS_RX(existingmixed128)',pair_weights=[.5,.5,.25],second_pair='existing ordinary64+selected64;no new IQ or forward',same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['factorial'].update(absent_slot='original NoAs returns raw64 on selected mask',As_input='unchanged fullpractical(raw64) before ordinary')
 c['crossview'].update(mode='MIXED128',ordinary_anchors=64,additional_anchors=0,additional_view='replace ordinary128 by existing mixed128,never concatenate',schedule='same original cumulativeE21..30/full0.1',total_anchors=128,anchors_per_physical_input=1,used_pass_maps='corresponding existing mixed first/second RD pass',raw_pair_maps_used=0)
 c['mixed_contrast']=dict(mode='MIXED128',total_anchors=128,selected64_input='As_raw' if sat else 'raw',anchors_per_input_position=1,physical_ID_uniqueness_required=False,extra_forwards=0,new_losses=0,existing_coefficient=.1,temperature=.2,levels=['time_down','t2.pool'],relation='sameTX differentRX positive;otherTX negative;sameTX sameRX excluded',normalization='unchanged detached current128 descriptor moments thenL2',prior_SAT192='ordinary128 plus additional satellite64;source failed;not repeated')
 c['source_promotion']=dict(fullV_newAs_minus_originalAs_clean_min_pp=-.1,panel_midlow_newAs_minus_originalAs_min_pp=.5,source_failure_skips_formal=False,automatic_extension=False)
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed two-row mixed128 contrast/data/parent/seed/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 row=next(r for r in rows() if r['row_id']==c['row_id']);original=b.config(next(r for r in b.rows() if r['row_id']==c['row_id']),30)
 if original!=reference_config(row,30):raise ValueError('Pinned actual P43 recipe differs from saved reference')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher:raise ValueError('Original tail L/RDrop/U args differ')
 return a
