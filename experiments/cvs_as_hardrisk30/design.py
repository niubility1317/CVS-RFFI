"""Fixed ownAs20+10:only selected As CE target smoothing .01 to .05."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-withinTXHard-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_withinTXHard_E30_20261019_r01'
OWNER='codex/root/practical-withinTXHard';LOCAL='local_artifacts/F_withinTXHard_E30_20261019'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
REFERENCE_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=2220
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660
RISK_GAMMA=.25;RISK_SEED=20261019091

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

def rows():return [dict(arm='B01',row_id=a+'-s'+str(SEED),model_seed=SEED,risk_mode=m) for a,m in [('H01','HARD'),('R01','SHUFFLE')]]
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only original B01 As20/30 reference')
 return deepcopy(_REFERENCE['configs']['B01'][str(segment)])
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed ownAs20+10/risk2rows')
 c=reference_config(row,30);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/'B01-s2026092701'/'stage20'
 c.update(schema='FcrossRX_withinTXHard30_v1',run_id=RUN,**row,launch_owner=OWNER,method='within_TX_bounded_CE_risk',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict original B01 sourceFINAL20 student and ownEMA',native_scratch_only=False,baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,segment=30,parent_epochs=20,parent_optimizer_steps=4440,epoch_offset=20,step_offset=4440,cumulative_epochs=30,cumulative_updates=6660,new_Ag=False,Ag_mode='NONE',As=True,training_recipe='OriginalP43 sixL/RDrop/raw/CROSS_RX/oldU;only withinTX selectedAs CE bounded weights',source_selection='fixedFINAL30;all7 formal even sourcefailure;future E50 sourceonly',budget_authorization='Two ownAs20+10 tails,totalE30 each;4440new updates;originalNoAs30/As30 reused',cost_claim='6L768 and oldU unchanged;zero extraNN/VJP;two extraCE64 operators eachstep',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),baseline_reference='Original P43 B00/B01 own20 scratch ancestors and fixedFINAL30 readouts,not targetranked')
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RDordinary+.5RDmixed with bounded AsCE correction+.25RDraw+.1ORD128CROSS_RX',pair_weights=[.5,.5,.25],same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['crossview'].update(mode='ORD128',ordinary_anchors=128,additional_anchors=0,additional_view='none',schedule='same cumulativeE21..30/.1')
 c['within_TX_risk']=dict(mode=row['risk_mode'],gamma=.25,top_fraction=.5,top_count='ceil(n_TX/2)',weight_min=.75,weight_max=1.25,weight_normalization='eachTX sumweights equals originaln;allselected64 supervised',score='detached average of two sameIQ selectedAs CE01 losses',HARD='weights .75+.25*n/k on topk,others .75',SHUFFLE='same class-specific weight histogram,private CPU generator permutation withinTX',seed=RISK_SEED,selected_packets=64,selected_total_nominal_CE_weight=.25,correction_formula='.25*sum_pass sum_selected((w-1)*CE01)/128',rank_differentiated=False,KL_unchanged=True,ordinary_raw_U_SupCon_unchanged=True,physical_channel_unchanged=True,new_losses=0,extra_NN=0,extra_VJP=0,extra_CE_operator_calls=2,extra_CE_logit_packets=128,confidence_filter=False)
 c['source_promotion']=dict(HARD_minus_originalNoAs_fullVclean_min_pp=-.1,HARD_minus_originalAs_midlow_min_pp=.5,HARD_minus_SHUFFLE_midlow_must_be_positive=True,source_failure_skips_formal=False,automatic_extension=False,on_fail='close fixedgamma/tophalf;no E50/seed/ratio rescue')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed withinTX risk/data/parent/seed/E30 scope differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 original=b.config(next(r for r in b.rows() if r['row_id']=='B01-s2026092701'),30)
 if original!=reference_config(next(r for r in rows() if r['row_id']==c['row_id']),30):raise ValueError('Pinned P43 original config differs')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled:raise ValueError('Original tail L/RDrop/U args differ')
 return a
