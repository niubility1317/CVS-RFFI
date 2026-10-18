"""Source-qualified satellite-dropout recipe, fixed own FINAL30 to FINAL50 budget."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-satelliteDropout-budget-e50-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01'
OWNER='codex/root/satellite-budget62';LOCAL='local_artifacts/F_satellite_dropout_budget_E50_20261018'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
PARENT_RUN='20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
SATELLITE_COMMIT='735e171fc064094f227fe6807d75836a0b3b0e3a'
BUDGET_REFERENCE_RUN='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
ARMS=('AS_CE','AS_DET');SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=4440
PARENT_EPOCHS=30;EPOCH_OFFSET=30;STEP_OFFSET=6660;CUMULATIVE_EPOCHS=50;CUMULATIVE_UPDATES=11100

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
_runtime=ROOT/'runtime.json';PROJECT=read(_runtime)['project'] if _runtime.is_file() else 'PROJECT'
def resolved(v):
 if isinstance(v,dict):return {k:resolved(x) for k,x in v.items()}
 if isinstance(v,list):return [resolved(x) for x in v]
 return v.replace('PROJECT',PROJECT) if isinstance(v,str) else v
_REFERENCE=resolved(read(Path(__file__).with_name('reference_config.json')))
globals().update(_REFERENCE['constants'])
SOURCE=Path(SOURCE);FORMAL_VIEWS=Path(FORMAL_VIEWS);TRUTH=Path(TRUTH);PANEL=Path(PANEL);SOURCE_CACHE=Path(SOURCE_CACHE);BASE=Path(PROJECT)/'runs'/RUN

def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def logical_rows():
 root=Path(PROJECT)/'runs';ref=root/REFERENCE_RUN;parent=root/PARENT_RUN;budget=root/BUDGET_REFERENCE_RUN
 return [('NOAS30',ref/'B00-s2026092701',True),('RD30',ref/'B01-s2026092701',True),('CE30',parent/'AS_CE-s2026092701',True),('DET30',parent/'AS_DET-s2026092701',True),('NOAS50',budget/'B00-s2026092701',True),('RD50',budget/'B01-s2026092701',True),('CE50',BASE/'AS_CE-s2026092701',False),('DET50',BASE/'AS_DET-s2026092701',False)]
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed satellite AS_CE/AS_DET seed01 own FINAL30 to FINAL50')
 c=deepcopy(_REFERENCE['configs']['30']);parent=Path(PROJECT)/'runs'/PARENT_RUN/row['row_id']/'source'
 c.update(schema='FcrossRX_satellite_budget62_v1',stage='fullsource_satellite_dropout_budget_extension',method='matched_satellite_dropout_E30_to_E50',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='own_fixed_FINAL30_student_and_EMA',native_scratch_only=False,baseline_reuse=False,epochs=20,label_epochs=0,pseudo_epochs=20,updates=UPDATES,segment=30,parent_epochs=30,parent_segment_epochs=10,parent_segment_updates=2220,epoch_offset=EPOCH_OFFSET,step_offset=STEP_OFFSET,cumulative_epochs=CUMULATIVE_EPOCHS,cumulative_updates=CUMULATIVE_UPDATES,new_Ag=False,Ag_mode='NONE',As=True,satellite_mode=row['arm'],training_recipe='Unchanged SAT18 .5ordinaryRD+.5mixedCEwithordinaryhalfKL+.25rawRD+CROSS_RX;AS_DET bypass existingdropout actual64Asrows withfullmaskRNG;strictownFINAL30+20U/resetAdamW/cos20',budget_authorization='User permits source-supported good recipe totalE50;2own30+20 rows/new40modelE/8880updates;no seed expansion/noE70',cost_claim='Pernewrow20epochs/4440updates/6271922nativeNN+12600source+1176002formal=7460524NN;own20+10 parentcost separate;4NN compoundrealCP smoke separate',source_selection='FixedFINAL50;source DETvsNoAs50 clean>=-.1pp,DETvsCE50 pooledmidlow>=+.5pp andlow>=0;allnewrows formal regardless;noautomaticseedextension',baseline_reference='Reuse P43 NoAs/RD30,SAT18 CE/DET30,budget50 NoAs/RD50;matched sixLNN/defaulthead/U/data/fullphysics;no old repredict/rescore/retrain',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),benchmark_context='Prior exposed benchmark retained;E50 promotion follows completed SAT18 sourcePASS before readingformal;no target-based selection')
 c['factorial'].update(Ag=False,As=True)
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.5[CE(mixed128)+.5KL(ordinaryhalf)]+.25RD(raw128)+.1CROSS_RX(ordinary)',same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['crossview'].update(mode='ORD128',additional_anchors=0,additional_view='none',schedule='native segment epoch+30;full0.1 at E31..50')
 c['satellite_dropout']=dict(mode=row['arm'],mask='Actual alternating64 physicalsatellite rows from existingstep parity',AS_CE='Originalstochasticdropout;only satelliteKLoff',AS_DET='Same CE,existingnn.Dropout output bypass onlysatellite rows;still draw nativefullmask to match RNG',ordinary_raw_U='Originaldropout/RDrop unchanged',new_parameters=0,new_deployment_modules=0,extra_forwards=0,all_labeled_satellite_supervised=True)
 c['optimizer_extension']=dict(optimizer='AdamW',reset=True,lr=.0002,weight_decay=.0001,lr_schedule='cosine20',lr_min=.000001,EMA='strict own FINAL30 EMA;localcounter0..4440,cumulative6660..11100;startup_averageFalse',ordinary_augmentation='freeze nativeepoch13 using own scratch20 args/epochs20;not FINAL30 args/epochs10',no_new_U=True)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed E50 data/parent/budget/head/physics/dropout differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_satellite_dropout18 import design as b
 a=b.make_args(b.config(next(r for r in b.rows() if r['arm']==c['arm']),30),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';a.epochs=20;a.label_epochs=0;a.pseudo_epochs=20;a.lr=.0002;a.weight_decay=.0001;a.from_scratch=False;a.a1_scratch_only=False;a.a1_ema_startup_average=False
 if a.label_smoothing!=.01 or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or a.batch_size!=128:raise ValueError('Original L/RDrop/U arguments differ')
 return a
