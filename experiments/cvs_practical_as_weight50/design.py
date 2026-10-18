"""Budget-only original weighted FINAL30+20; no recipe/physics change."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_practical_as_weight30 import design as old
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_AsWeight_E50_20261018_r01'
OWNER='codex/root/practical-Asweight50';LOCAL='local_artifacts/F_As_weight_E50_20261018'
PARENT_RUN=old.RUN;PARENT_RELEASE=old.RELEASE;PARENT_COMMIT='3befd47c8041d81f9430eda53562d0821624ff04'
REFERENCE_RUN=old.REFERENCE_RUN;REFERENCE_COMMIT=old.REFERENCE_COMMIT;REFERENCE_RELEASE=old.REFERENCE_RELEASE
ORIGINAL50_RUN='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
SEED=old.SEED;ARMS=old.ARMS;RX=FIT=old.RX;STEPS=222;UPDATES=4440
EPOCH_OFFSET=30;STEP_OFFSET=6660;CUMULATIVE_EPOCHS=50;CUMULATIVE_UPDATES=11100
read,write=old.read,old.write
PROJECT=old.PROJECT;SOURCE=old.SOURCE;FORMAL_VIEWS=old.FORMAL_VIEWS;TRUTH=old.TRUTH;PANEL=old.PANEL;SOURCE_CACHE=old.SOURCE_CACHE;FULL_FP32_POLICY=old.FULL_FP32_POLICY;CLASSES=old.CLASSES;SCENES=old.SCENES;VIEWS=old.VIEWS
BASE=Path(PROJECT)/'runs'/RUN
rows=old.rows

def parent_config(c):return old.config(next(r for r in old.rows() if r['row_id']==c['row_id']))
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed weighted seed01 ownFINAL30+20')
 c=deepcopy(old.config(row));parent=Path(PROJECT)/'runs'/PARENT_RUN/row['row_id']/'source'
 c.update(schema='FcrossRX_practical_Asweight50_v1',run_id=RUN,**row,launch_owner=OWNER,method='weighted_practical_budget_extension',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict own weighted FINAL30 student and EMA',epochs=20,label_epochs=0,pseudo_epochs=20,updates=UPDATES,parent_epochs=30,parent_segment_epochs=10,parent_segment_updates=2220,parent_optimizer_steps=6660,epoch_offset=30,step_offset=6660,cumulative_epochs=50,cumulative_updates=11100,source_selection='fixed FINAL50;source future seed eligibility only;all7 formal mandatory for both rows',budget_authorization='source-qualified unchangedweighted recipe own30+20,total50;two new rows only,not50extra',cost_claim='perrow4440updates/3409920LNN/6271922nativeNN;As284160 orNoAs0;readouts/sharedsmoke andown30parents separate',baseline_reference='8logical original30/50 andweighted30/50;only two new weighted50rows;old frozen predictions reused',benchmark_context='E50 chosen from fixed source Asweight30 criteria before formalread;no target feedback or selective seed rescue')
 c['training_mechanism'].update(expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640)
 c['crossview']['schedule']='original cumulativeE31..50/full0.1'
 c['mixed_weight']['mask']='(batch_position+local_step)%2 !=0;offset6660 even'
 c['source_promotion']=dict(fullV_As50_minus_ownAs30_clean_min_pp=0.,panel_midlow_As50_minus_ownAs30_min_pp=.5,fixed_future_seeds=[2026092702,2026092703],NoAs_and_interactions_reported=True,source_failure_skips_formal=False,automatic_extension=False)
 c['optimizer_extension']=dict(optimizer='AdamW',reset=True,lr=.0002,weight_decay=.0001,schedule='cosine20',lr_min=.000001,EMA='own weighted FINAL30 EMA;local0..4440,cumulative6660..11100;cache reset',ordinary_augmentation='own original20 args freezeepoch13;not native weighted30 args epochs10')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed own-weighted30+20 data/seed/head/recipe/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(parent_config(c),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';a.epochs=20;a.label_epochs=0;a.pseudo_epochs=20;a.lr=.0002;a.weight_decay=.0001;a.from_scratch=False;a.a1_scratch_only=False;a.a1_ema_startup_average=False
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.seed,a.label_smoothing)!=(20,0,20,128,SEED,.01) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher:raise ValueError('Original weighted E50 native/U/defaulthead args differ')
 return a
