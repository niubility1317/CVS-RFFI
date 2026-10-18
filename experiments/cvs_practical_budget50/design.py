"""Fixed original practical recipe, paired E30-to-E50 continuation; no new mechanism."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_As_budget_E50_20261018_r01'
OWNER='codex/root/practical-budget50';LOCAL='local_artifacts/F_As_budget_E50_20261018'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
ARMS=('B00','B01');SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=4440
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
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed B00/B01 seed01 cumulative FINAL50 continuation')
 c=deepcopy(_REFERENCE['configs']['30']);parent=Path(PROJECT)/'runs'/REFERENCE_RUN/row['row_id']/'source';sat=row['arm']=='B01'
 c.update(schema='FcrossRX_practical_budget50_v1',stage='fullsource_practical_budget_extension',method='matched_E30_to_E50_budget',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='own_fixed_FINAL30_student_and_EMA',native_scratch_only=False,baseline_reuse=False,epochs=20,label_epochs=0,pseudo_epochs=20,updates=UPDATES,segment=30,parent_epochs=30,parent_segment_epochs=10,parent_segment_updates=2220,epoch_offset=EPOCH_OFFSET,step_offset=STEP_OFFSET,cumulative_epochs=CUMULATIVE_EPOCHS,cumulative_updates=CUMULATIVE_UPDATES,new_Ag=False,Ag_mode='NONE',As=sat,training_recipe='OriginalP43 .5ordinaryRD+.5mixedRD+.25rawRD+CROSS_RX;ownFINAL30 student/EMA+20U,resetAdamW/cos20;full satellite KL/dropout retained',budget_authorization='User permits well-supported originalAs totalE50;fixedseed01 NoAs/As each30+20,not failed dropout rescue',cost_claim='Pernewrow20epochs/4440updates/6271922nativeNN+12600source+1176002formal=7460524NN;parent30cost separately;2newrows only',source_selection='FixedFINAL50;source fullV/panel comparison descriptive, no target selection or automaticseedextension',baseline_reference='Matched B00/B01 sourcefixedFINAL30 lineage, continuation20 each;unchanged6L/defaulthead/U/data/physics;original30 readouts reused for budgetinteraction',goal=dict(primary='fixed practical low/midurban',stage_clean_drop_max_absolute_pp=.1,final_repeated_clean_gain=True,target_feedback=False,goal_complete=False),benchmark_context='Prior exposedbenchmark context retained;E50 extension selected from three-seed legal source evidence;not target score based')
 c['factorial'].update(Ag=False,As=sat)
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640,objective='.5RD(ordinary128)+.5RD(mixed128)+.25RD(raw128)+.1CROSS_RX(ordinary)',same_IQ_within_RDpair=True,augmentation_redraws_in_second_forward=0)
 c['crossview'].update(mode='ORD128',additional_anchors=0,additional_view='none',schedule='native segment epoch+30;full0.1 at E31..50')
 c['optimizer_extension']=dict(optimizer='AdamW',reset=True,lr=.0002,weight_decay=.0001,lr_schedule='cosine20',lr_min=.000001,EMA='strict own FINAL30 EMA;localcounter0..4440,cumulative6660..11100;startup_averageFalse',ordinary_augmentation='freeze nativeepoch13 using own scratch20 args/epochs20;not FINAL30 args/epochs10',no_new_U=True)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed E50 data/parent/budget/head/physics differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 a=b.make_args(b.config(next(r for r in b.rows() if r['arm']==c['arm']),30),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';a.epochs=20;a.label_epochs=0;a.pseudo_epochs=20;a.lr=.0002;a.weight_decay=.0001;a.from_scratch=False;a.a1_scratch_only=False;a.a1_ema_startup_average=False
 if a.label_smoothing!=.01 or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or a.batch_size!=128:raise ValueError('Original L/RDrop/U arguments differ')
 return a
