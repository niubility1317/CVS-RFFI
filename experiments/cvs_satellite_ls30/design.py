"""Fixed ownAs20+10:only selected As CE target smoothing .01 to .05."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-AsLS-e30-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_AsLS_E30_20261018_r01'
OWNER='codex/root/practical-AsLS';LOCAL='local_artifacts/F_AsLS_E30_20261018'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
REFERENCE_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=2220
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660
OLD_EPS=.01;SATELLITE_EPS=.05

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
ARMS=('B01',)
def rows():return [dict(arm='B01',row_id='B01-s'+str(SEED),model_seed=SEED)]
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only actual original ownAs20/30 reference')
 return deepcopy(_REFERENCE['configs']['B01'][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only ownoriginalAs20+10 fixed seed01')
 c=reference_config(row,30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/row['row_id']/'stage20'
 c.update(schema='FcrossRX_satelliteLS30_v1',run_id=RUN,**row,launch_owner=OWNER,method='As_selected_CE_label_smoothing',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict original ownAsFINAL20 student and EMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=True,training_recipe='OriginalP43 .5RDordinary128+.5RDmixed128+.25RDraw128+.1ORD128CROSS_RX;only selectedAs64 CE eps .05 instead of .01',source_selection='fixedFINAL30;source failure still completes all7 formal;no automatic extension',budget_authorization='one ownoriginalAs20+10,totalE30;2220newupdates;reuse frozen originalNoAs30 andAs30 readouts',cost_claim='6L/768NN and1.25Lscale eachstep;1704960newLNN;142080As;0extraNN but4extraCE ops/256logitpackets perstep',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='Original P43 ownAs20 initializer and originalNoAs30/As30 frozen readouts;original mixed .25/.25 group weights',benchmark_context='CE_T2 and CE_HALF changed temperature or weight;this fixed mechanism changes only target entropy,not their closed recipes')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.5RD(mixed128 with As64 CE05/ordinary64 CE01)+.25RD(raw128)+.1CROSS_RX(ordinary128)',pair_weights=[.5,.5,.25],second_pair='existing ordinary64+completeAs(raw64);same original mask',same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['factorial'].update(absent_slot='originalNoAs selected raw64;readout reused,not newly trained',As_input='unchanged complete practical(raw64) before ordinary')
 c['crossview'].update(mode='ORD128',ordinary_anchors=128,additional_anchors=0,additional_view='none',schedule='same original cumulativeE21..30/full0.1')
 c['satellite_label_smoothing']=dict(old_epsilon=OLD_EPS,epsilon=SATELLITE_EPS,native_label_smoothing=OLD_EPS,selected_packets_per_step=64,selected_positions='original alternating mask (position+local_step)%2!=0',scope='only CE targets of existing mixed selectedAs64 two passes',mixed_CE_normalization='mean across128 then mean across2passes;original outer.5',delta_formula='.25*(sum_selected(CE05_pass0-CE01_pass0)+sum_selected(CE05_pass1-CE01_pass1))/128',selected_total_CE_weight=.25,mixed_KL_unchanged=True,common_raw_U_SupCon_unchanged=True,same_IQ_within_RDpair=True,confidence_filter=False,all_six_classes_same_formula=True,extra_NN=0,extra_CE_operator_calls_per_step=4,extra_CE_logit_packets_per_step=256,baseline_mixed_CE_operator_widths=[128,128],extra_CE_operator_widths=[64,64,64,64],actual_mixed_CE_operator_widths=[128,128,64,64,64,64],explicit_CUDA_synchronize=False,new_losses=0,new_U_views=0,formal_test_changed=False)
 c['source_promotion']=dict(fullV_newAs_minus_originalNoAs_clean_min_pp=-.1,panel_midlow_newAs_minus_originalAs_min_pp=.5,source_failure_skips_formal=False,automatic_extension=False,on_fail='close fixed.05;no E50/epsilon/seed rescue')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed one-row satelliteLS/data/parent/seed/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 row=next(r for r in rows() if r['row_id']==c['row_id']);original=b.config(next(r for r in b.rows() if r['row_id']==c['row_id']),30)
 if original!=reference_config(row,30):raise ValueError('Pinned actual P43 recipe differs from saved reference')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled:raise ValueError('Original tail L/RDrop/U args differ')
 return a
