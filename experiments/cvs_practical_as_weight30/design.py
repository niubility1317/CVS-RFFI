"""One fixed E20+10 continuation; temporal organization is the sole new factor."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_AsWeight_E30_20261018_r01'
OWNER='codex/root/practical-Asweight';LOCAL='local_artifacts/F_As_weight_E30_20261018'
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
 if row not in rows() or segment not in (20,30):raise ValueError('Only original B00/B01 fixed20+10 reference')
 return deepcopy(_REFERENCE['configs'][row['arm']][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed B00/B01 E20+10 weighted mixed supervision')
 c=reference_config(row,30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/row['row_id']/'stage20';sat=row['arm']=='B01'
 c.update(schema='FcrossRX_practical_Asweight30_v1',run_id=RUN,**row,launch_owner=OWNER,method='mixed_slot_supervision_weight',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict own original FINAL20 student and EMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=sat,training_recipe='.5RDordinary128+.125RDmixedOrd64+.375RDmixedSelected64+.25RDraw128+.1ORD128 CROSS_RX;existing64 mask/IQ/6L/order/U unchanged',source_selection='fixed FINAL30;source rule decides only future E50 or seed expansion;all7 formal automatic even on source failure',budget_authorization='two original ownNoAs20/As20+10;totalE30 each;4440newupdates;reuse original FINAL30 controls',cost_claim='6L/768NN and1.25Lscale every step;eachnewrow1704960LNN,As row142080As andNoAs0;parents20/readouts separate',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='reused P43 originalNoAs30/As30 plus newmatchedNoAsweighted;all same seed/data/6L/own20 lineage',benchmark_context='Already exposed benchmark background retained;one fixed.375 supervision allocation from source organization evidence,no target selection or grid')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.125RD(mixedOrd64)+.375RD(mixedSelected64)+.25RD(raw128)+.1CROSS_RX(ordinary128)',pair_weights=[.5,.5,.25],second_pair='existing mixed128 two forwards;group means64+64,CE and symmetricKL same weights',same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['factorial'].update(absent_slot='original NoAs identity raw64 on selected mask;not ordinary',As_input='unchanged complete practical(raw before ordinary)')
 c['crossview'].update(mode='ORD128',additional_anchors=0,additional_view='none',schedule='original cumulative21..30;full0.1')
 c['mixed_weight']=dict(ordinary_group_weight=.125,selected_group_weight=.375,total_pair_weight=.5,primary_ordinary_weight=.5,raw_weight=.25,total_L_scale=1.25,selected_count=64,selected_IQ='existing full practical raw64' if sat else 'original NoAs identity returns raw64',mask='(batch_position+local_step)%2 !=0;offset4440 even',CE_and_symmetricKL_both_weighted=True,all_existing_As_supervised=True,new_losses=0,extra_NN=0)
 c['source_promotion']=dict(fullV_newAs_minus_originalAs_clean_min_pp=-.1,panel_midlow_newAs_minus_originalAs_min_pp=.5,NoAs_weighted_control_and_interaction_reported=True,source_failure_skips_formal=False,automatic_extension=False)
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed two-row weight/data/parent/seed/budget/head/physics differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 row=next(r for r in rows() if r['row_id']==c['row_id']);original=b.config(next(r for r in b.rows() if r['row_id']==c['row_id']),30)
 if original!=reference_config(row,30):raise ValueError('Pinned actual P43 recipe differs from saved reference')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher:raise ValueError('Original tail L/RDrop/U args differ')
 return a
