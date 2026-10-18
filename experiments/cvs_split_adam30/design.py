"""Original own20+10 recipe; one fixed rest/As AdamW-state factor."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-splitAdam-e30-s2026092701-m3-r01'
RELEASE='cvs_FcrossRX_splitAdam_E30_20261019_r01'
OWNER='codex/root/splitAdam30';LOCAL='local_artifacts/F_splitAdam_E30_20261019_r01'
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
ARMS=('N01','C01','S01')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def reference_row(row):
 if row not in rows():raise ValueError('Only fixed N01/C01/S01 seed01 rows')
 a='B00' if row['arm']=='N01' else 'B01'
 return dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED)
def reference_config(row,segment=30):
 if segment not in (20,30):raise ValueError('Only original own20 or its10epoch segment')
 return deepcopy(_REFERENCE['configs'][reference_row(row)['arm']][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only ownoriginal20+10 fixed seed01 rows')
 c=reference_config(row,30);ref=reference_row(row);sat=row['arm']!='N01';mode='SPLIT' if row['arm']=='S01' else 'MEASURE';parent=Path(PROJECT)/'runs'/REFERENCE_RUN/ref['row_id']/'stage20'
 c.update(schema='FcrossRX_splitAdam30_v1',run_id=RUN,**row,launch_owner=OWNER,method='view_separate_AdamW_states',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],parent_row_id=ref['row_id'],model_initialization='strict original ownFINAL20 student and EMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=sat,optimizer_mode=mode,amp=False,training_recipe='.5RDordinary128+.5RDmixed128+.25RDraw128+.1CROSS_RX(ordinary128);oldU unchanged;optimizer state factor only',source_selection='fixedFINAL30;source failure still completes all7 formal;no automatic E50',budget_authorization='three ownoriginal20+10,totalE30 each;6660newupdates;original20 reused',cost_claim='6L/768NN perstep;As64or0;two extra VJPs eachrow;zero added NN',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='Original P43 ownNoAs20/ownAs20 initializers;new matched MEASURE rows;no test-selected parent')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.5RD(mixed128)+.25RD(raw128)+.1CROSS_RX(ordinary128)+unchanged native U',pair_weights=[.5,.5,.25],same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['moment_mode']=mode
 c['split_adam']=dict(mode=mode,As_loss='.25RD(masked64 of original mixed128);NoAs graph-connected zero',rest_loss='native total L+U minus As_loss',VJP_calls_per_update=2,extra_NN=0,new_losses=0,new_U=False,all_parameters=True,adaptive_weights=[.8,.2],adaptive_weight_interpretation='fixed normalized optimizer displacements;not preservation of original effective gradient weights',state='independent rest/As first and second moments only in SPLIT',clipping='same single original physical total-gradient clip5;common factor for both VJPs',decay='once original AdamW decay including CENTER_UNITS physical Sinc hook',EMA='once original native successful update',amp=False)
 c['source_promotion']=dict(panel_midlow_S_minus_C_min_pp=.5,fullV_clean_S_minus_N_min_pp=-.1,source_failure_skips_formal=False,automatic_extension=False,ratio_threshold_seed_optimizer_rescue=False)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed three-row splitAdam/data/parent/seed/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 row=next(r for r in rows() if r['row_id']==c['row_id']);ref=reference_row(row);original=b.config(next(r for r in b.rows() if r['row_id']==ref['row_id']),30)
 if original!=reference_config(row,30):raise ValueError('Pinned actual P43 recipe differs from saved reference')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';a.amp=False
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or a.amp:raise ValueError('Original tail L/RDrop/U args or fixed AMP differs')
 return a
