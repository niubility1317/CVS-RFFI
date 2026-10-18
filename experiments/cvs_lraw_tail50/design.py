"""Source-selected budget extension only:own LRawFINAL30+20,not fifty new epochs."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_lraw_tail30 import design as old
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_LRaw_E50_20261018_r01'
OWNER='codex/root/practical-LRaw50';LOCAL='local_artifacts/F_LRaw_E50_20261018'
PARENT_RUN=old.RUN;PARENT_RELEASE=old.RELEASE;PARENT_COMMIT='d14450bf1b0d12cc281df1f4d0e79e8760e48282'
REFERENCE_RUN=old.REFERENCE_RUN;REFERENCE_COMMIT=old.REFERENCE_COMMIT;REFERENCE_RELEASE=old.REFERENCE_RELEASE
ORIGINAL50_RUN='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
ORIGINAL50_RELEASE='cvs_FcrossRX_As_budget_E50_20261018_r01';ORIGINAL50_COMMIT='0f399c910324e21a8b1a60346d879be5c2964a9c'
SEED=old.SEED;ARMS=old.ARMS;RX=FIT=old.RX;STEPS=222;UPDATES=4440
EPOCH_OFFSET=30;STEP_OFFSET=6660;CUMULATIVE_EPOCHS=50;CUMULATIVE_UPDATES=11100
read,write=old.read,old.write
PROJECT=old.PROJECT;SOURCE=old.SOURCE;FORMAL_VIEWS=old.FORMAL_VIEWS;TRUTH=old.TRUTH;PANEL=old.PANEL;SOURCE_CACHE=old.SOURCE_CACHE;FULL_FP32_POLICY=old.FULL_FP32_POLICY;CLASSES=old.CLASSES;SCENES=old.SCENES;VIEWS=old.VIEWS
TRAIN_SCENES=old.TRAIN_SCENES;CHANNEL_SEEDS=old.CHANNEL_SEEDS
BASE=Path(PROJECT)/'runs'/RUN
rows=old.rows

def parent_config(c):return old.config(next(r for r in old.rows() if r['row_id']==c['row_id']))
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed LRaw seed01 ownFINAL30+20')
 c=deepcopy(old.config(row));parent=Path(PROJECT)/'runs'/PARENT_RUN/row['row_id']/'source'
 c.update(schema='FcrossRX_LRaw50_v1',run_id=RUN,**row,launch_owner=OWNER,method='LRaw_budget_extension',output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict own LRaw FINAL30 student and ownEMA',epochs=20,label_epochs=0,pseudo_epochs=20,updates=UPDATES,parent_epochs=30,parent_segment_epochs=10,parent_segment_updates=2220,parent_optimizer_steps=6660,epoch_offset=30,step_offset=6660,cumulative_epochs=50,cumulative_updates=11100,source_selection='fixed FINAL50;source comparison descriptive only;all7 formal mandatory for both rows',budget_authorization='User permits source-supported unchanged LRaw totalE50;two own30+20 rows only,not50additional;no seed expansion',cost_claim='perrow4440updates/3409920LNN/1134000existingUstudent/1134000ownteacherEval;As284160 B01 or0 B00;shared1540smoke/parent30/source/formal costs separate',baseline_reference='8logical original30/50 andLRaw30/50;only two new LRaw50rows;old frozen readouts reused',benchmark_context='E50 selected from predeclared LRaw30 source AND before formalread;no target feedback or selective rescue;oldordinary20 thenraw30,not fullNoAug')
 c['training_mechanism'].update(expected_L_forward_packets=UPDATES*768,original_L_identity_forwards=UPDATES*128,extra_L_identity_forwards=UPDATES*640)
 c['crossview']['schedule']='original cumulativeE31..50/full0.1'
 c['lraw'].update(first20_ordinary_retained=True,raw_return_epochs_cumulative=30,whole_trajectory_NoAug=False,ordinary_packets_new=UPDATES*128)
 c.pop('source_promotion',None)
 c['optimizer_extension']=dict(optimizer='AdamW',reset=True,lr=.0002,weight_decay=.0001,schedule='cosine20',lr_min=.000001,EMA='own LRaw FINAL30 EMA;local0..4440,cumulative6660..11100;cache reset',ordinary_augmentation='strict originalown20 args/base freeze13;not native LRaw30 args epochs10')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed ownLRaw30+20/data/seed/head/recipe/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(parent_config(c),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';a.epochs=20;a.label_epochs=0;a.pseudo_epochs=20;a.lr=.0002;a.weight_decay=.0001;a.from_scratch=False;a.a1_scratch_only=False;a.a1_ema_startup_average=False
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.seed,a.label_smoothing)!=(20,0,20,128,SEED,.01) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_unlabeled or not a.use_ema_teacher or a.freeze_backbone or a.phase1_lr_schedule!='cosine' or a.phase1_lr_min!=.000001:raise ValueError('Original LRaw E50 native/U/defaulthead/cos20 args differ')
 return a
