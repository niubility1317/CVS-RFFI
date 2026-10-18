"""One fixed E20+10 continuation; temporal organization is the sole new factor."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-stepALT-e30-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_stepALT_E30_20261018_r01'
OWNER='codex/root/practical-stepALT';LOCAL='local_artifacts/F_As_stepALT_E30_20261018'
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

def rows():return [dict(arm='B01',row_id='B01-s2026092701',model_seed=SEED)]
def reference_config(segment=30):return deepcopy(_REFERENCE['configs'][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only one fixed original As20 +10 stepALT row')
 c=reference_config(30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/row['row_id']/'stage20'
 c.update(schema='FcrossRX_practical_stepALT30_v1',run_id=RUN,**row,launch_owner=OWNER,method='step_alternating_full_practical',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict original As fixed FINAL20 student and ownEMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=True,training_recipe='.5RDordinary128+.5RDmixed128+.25RDraw128+.1ORD128 CROSS_RX; mixed all128As on odd cumulative steps,allordinary on even steps;original U unchanged',source_selection='fixed FINAL30;source readout descriptive,automatic all7 formal even on scientific failure',budget_authorization='one original ownAs20+10;totalE30;2220newupdates;no new controls',cost_claim='6L/768NN and1.25Lscale every step;142080As in new10 same total as originalAs30 tail;parent20 and readouts separate',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='reuse original P43 same seed B00 NoAs30 and B01 As30;same parentAs20 for mechanism comparison;no target selection',benchmark_context='Already exposed benchmark background retained;candidate fixed from temporal-organization hypothesis and source exposure evidence;target scores do not choose method or seed')
 c['factorial'].update(partition='cumulative step odd:all128As;even:all128ordinary;deterministic no labels or quality',operator_marginal_packet_fraction=.5,absent_slot='same generated ordinary IQ, no channel call',Ag=False,As=True)
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.5RD(mixed128)+.25RD(raw128)+.1CROSS_RX(ordinary128)',pair_weights=[.5,.5,.25],second_pair='128As on oddstep,128sameordinary on evenstep',same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['crossview'].update(mode='ORD128',additional_anchors=0,additional_view='none',schedule='original cumulative21..30;full0.1')
 c['step_alternate']=dict(on='odd cumulative step,128 full practical',off='even cumulative step,0 practical;reuse ordinary',first_cumulative_step=4441,last_cumulative_step=6660,As_packets_per_epoch=14208,As_packets_total=142080,same_per_physical_exposure=False,scene_prior='unchanged uniform three scenarios;realized exposure recorded',no_new_losses=True,no_new_U=True,extra_NN=0)
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed single stepALT data/parent/seed/budget/head/physics differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 original=b.config(next(r for r in b.rows() if r['row_id']==c['row_id']),30)
 if original!=reference_config(30):raise ValueError('Pinned P43 actual recipe differs from saved reference')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing)!=(10,0,10,128,.01) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher:raise ValueError('Original tail L/RDrop/U args differ')
 return a
